# =======================
# Step 0: Import Libraries
# =======================
import os
import json
import time
import random
import joblib  # For saving and loading scaler

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report, roc_curve,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Enable anomaly detection for better debugging
torch.autograd.set_detect_anomaly(True)

# =======================
# Step 1: Set Random Seeds for Reproducibility
# =======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# =======================
# Step 2: Load Datasets
# =======================
def load_dataset(filepath):
    try:
        df = pd.read_csv(filepath).drop(columns=['Employee ID'])
        print(f"Dataset '{filepath}' loaded successfully with shape {df.shape}.")
        return df
    except Exception as e:
        print(f"Error loading dataset '{filepath}': {e}")
        raise

# Replace 'train.csv' and 'test.csv' with the full paths to your files
dataset_directory = r""
df_train = load_dataset(os.path.join(dataset_directory, 'train.csv'))
df_test = load_dataset(os.path.join(dataset_directory, 'test.csv'))

# =======================
# Step 3: Define Column Groups
# =======================
categorical_cols = [
    'Job Role', 'Work-Life Balance', 'Job Satisfaction', 'Performance Rating',
    'Education Level', 'Marital Status', 'Job Level', 'Company Size',
    'Company Reputation', 'Employee Recognition'
]
binary_cols = ['Gender', 'Overtime', 'Remote Work', 'Leadership Opportunities', 'Innovation Opportunities']
target_col = 'Attrition'
continuous_cols = [
    'Age', 'Distance from Home', 'Monthly Income', 'Company Tenure',
    'Years at Company', 'Number of Promotions', 'Number of Dependents'
]

print("\nDefined feature groups:")
print(f"Categorical Columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Binary Columns ({len(binary_cols)}): {binary_cols}")
print(f"Continuous Columns ({len(continuous_cols)}): {continuous_cols}")
print(f"Target Column: {target_col}\n")

# =======================
# Step 4: Define Data Preparation Functions
# =======================
def encode_and_map_target(df, encoders=None, num_categorical=None, fit=False):
    """
    Encodes categorical and binary features and maps the target variable.
    """
    if encoders is None:
        encoders = {}
    if num_categorical is None:
        num_categorical = []

    # Define custom mappings for binary features
    binary_mappings = {
        'Gender': {'Male': 1, 'Female': 0},
        'Overtime': {'Yes': 1, 'No': 0},
        'Remote Work': {'Yes': 1, 'No': 0},
        'Leadership Opportunities': {'Yes': 1, 'No': 0},
        'Innovation Opportunities': {'Yes': 1, 'No': 0}
    }

    # Encode binary features
    for feature in binary_cols:
        if feature in df.columns:
            mapping = binary_mappings.get(feature, {'Yes': 1, 'No': 0})  # Default mapping
            df[feature] = df[feature].map(mapping).fillna(0).astype(int)
            print(f"Converted '{feature}' to numeric with mapping: {mapping}. Unmapped values set to 0.")
        else:
            print(f"Warning: Binary feature '{feature}' not found in DataFrame.")

    # Encode target variable
    if df[target_col].dtype == 'object' or isinstance(df[target_col].iloc[0], str):
        df[target_col] = df[target_col].map({'Left': 1, 'Stayed': 0}).fillna(0).astype(int)
        print("Mapped 'Attrition' to binary values.")
    else:
        print(f"'{target_col}' is already numeric. Skipping mapping.")

    # Encode categorical features
    for col in categorical_cols:
        if col not in df.columns:
            print(f"Warning: Categorical feature '{col}' not found in DataFrame.")
            continue

        if fit:
            n_unique = df[col].nunique()
            # Reserve one index for unseen categories
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=n_unique)
            encoder.fit(df[[col]])
            encoders[col] = encoder
            num_categorical.append(n_unique + 1)  # +1 for 'Unknown'
            print(f"Encoded '{col}': {n_unique} unique categories (+1 for 'Unknown').")
        else:
            if col not in encoders:
                raise ValueError(f"Encoder for column '{col}' not provided.")
            encoder = encoders[col]

        df[col] = encoder.transform(df[[col]]).astype(int)

        # Verify encoding
        min_val = df[col].min()
        max_val = df[col].max()
        expected_max = num_categorical[categorical_cols.index(col)] - 1 if fit else encoders[col].categories_[0].size
        print(f"Feature '{col}' - Min: {min_val}, Max: {max_val}, Expected Max: {expected_max}")
        assert df[col].min() >= 0, f"Error: Feature '{col}' has negative indices after encoding."
        assert df[col].max() <= expected_max, f"Error: Feature '{col}' has indices exceeding the encoding size."

    return df, encoders, num_categorical

# =======================
# Step 5: Global Preprocessing - Encoding
# =======================
# Apply global encoding to training data
df_train_encoded, encoders_global, num_categorical_global = encode_and_map_target(df_train.copy(), fit=True)

# Apply global encoding to test data using the same encoders
df_test_encoded, _, _ = encode_and_map_target(
    df_test.copy(), 
    encoders=encoders_global,
    num_categorical=num_categorical_global.copy(),
    fit=False
)

# Clean Column Names
def clean_column_names(df):
    """
    Cleans column names by stripping whitespace.
    """
    df.columns = df.columns.str.strip()
    return df

df_train_encoded = clean_column_names(df_train_encoded)
df_test_encoded = clean_column_names(df_test_encoded)

# Verify column match
def verify_column_match(df_train, df_test):
    """
    Verifies that the training and testing DataFrames have the same columns.
    """
    # Get sorted lists of columns for both DataFrames
    train_columns = sorted(df_train.columns)
    test_columns = sorted(df_test.columns)

    # Compare columns
    if train_columns == test_columns:
        print("The columns in the training and testing datasets match.")
    else:
        print("The columns in the training and testing datasets do NOT match.")
        print("\nColumns in training data but not in test data:", set(train_columns) - set(test_columns))
        print("Columns in test data but not in training data:", set(test_columns) - set(train_columns))

verify_column_match(df_train_encoded, df_test_encoded)
print()

# =======================
# Step 6: Define PyTorch Dataset
# =======================
class AttritionDataset(Dataset):
    def __init__(self, df, categorical_cols, binary_cols, continuous_cols, target_col):
        self.categorical = df[categorical_cols].values.astype(np.int64)
        self.binary = df[binary_cols].values.astype(np.float32)  # Include binary features
        self.continuous = df[continuous_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)

        # Print unique target classes to verify they contain both 0 and 1
        unique_targets = np.unique(self.targets)
        print("Unique target classes in AttritionDataset:", unique_targets)

        # Check for NaNs or Infs in the data
        if np.isnan(self.categorical).any():
            raise ValueError("NaN values found in categorical features.")
        if np.isnan(self.binary).any():
            raise ValueError("NaN values found in binary features.")
        if np.isnan(self.continuous).any():
            raise ValueError("NaN values found in continuous features.")
        if np.isnan(self.targets).any():
            raise ValueError("NaN values found in target values.")

        if np.isinf(self.categorical).any():
            raise ValueError("Inf values found in categorical features.")
        if np.isinf(self.binary).any():
            raise ValueError("Inf values found in binary features.")
        if np.isinf(self.continuous).any():
            raise ValueError("Inf values found in continuous features.")
        if np.isinf(self.targets).any():
            raise ValueError("Inf values found in target values.")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            'categorical': torch.tensor(self.categorical[idx], dtype=torch.long),
            'binary': torch.tensor(self.binary[idx], dtype=torch.float32),  # Include binary tensor
            'continuous': torch.tensor(self.continuous[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], dtype=torch.float32)
        }

# =======================
# Step 7: Define SAINT Model with Intersample and Intrasample Attention
# =======================
class SAINT(nn.Module):
    def __init__(self, num_categorical, num_binary, num_continuous, embedding_dim=16, hidden_dim=64, num_heads=2, num_layers=2, dropout=0.1):
        super(SAINT, self).__init__()

        # Embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, embedding_dim) for num_categories in num_categorical
        ])

        # Linear layer for binary features
        self.binary_linear = nn.Linear(num_binary, embedding_dim)

        # Linear layer for continuous features
        self.continuous_linear = nn.Linear(num_continuous, embedding_dim)

        # Total number of tokens (categorical + binary + continuous)
        self.num_tokens = len(num_categorical) + 1 + 1  # +1 for binary, +1 for continuous

        # Positional Embedding
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.num_tokens, embedding_dim))

        # Transformer Encoder for Intrasample Attention
        intra_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(intra_encoder_layer, num_layers=num_layers)

        # Transformer Encoder for Intersample Attention
        inter_d_model = self.num_tokens * embedding_dim
        inter_encoder_layer = nn.TransformerEncoderLayer(d_model=inter_d_model, nhead=num_heads, dropout=dropout)
        self.inter_sample_transformer = nn.TransformerEncoder(inter_encoder_layer, num_layers=num_layers)

        # Classification Head
        self.fc = nn.Sequential(
            nn.Linear(inter_d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, categorical, binary, continuous):
        batch_size = categorical.size(0)

        # Process categorical features
        embedded_categorical = [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        embedded_categorical = torch.stack(embedded_categorical, dim=1)  # [batch_size, num_categorical, embedding_dim]

        # Process binary features
        binary = self.binary_linear(binary).unsqueeze(1)  # [batch_size, 1, embedding_dim]

        # Process continuous features
        continuous = self.continuous_linear(continuous).unsqueeze(1)  # [batch_size, 1, embedding_dim]

        # Concatenate all tokens
        tokens = torch.cat([embedded_categorical, binary, continuous], dim=1)  # [batch_size, num_tokens, embedding_dim]

        # Add positional embedding for intrasample attention
        tokens = tokens + self.positional_embedding  # [batch_size, num_tokens, embedding_dim]

        # Intrasample Attention (within sample)
        tokens = tokens.permute(1, 0, 2).contiguous()  # [num_tokens, batch_size, embedding_dim]
        tokens = self.transformer_encoder(tokens)  # [num_tokens, batch_size, embedding_dim]
        tokens = tokens.permute(1, 0, 2).contiguous()  # [batch_size, num_tokens, embedding_dim]

        # Flatten tokens for intersample attention
        tokens = tokens.reshape(batch_size, -1)  # [batch_size, num_tokens * embedding_dim]

        # Intersample Attention (across samples)
        tokens = tokens.unsqueeze(1)  # [batch_size, 1, num_tokens * embedding_dim]
        tokens = tokens.permute(1, 0, 2).contiguous()  # [1, batch_size, num_tokens * embedding_dim]
        tokens = self.inter_sample_transformer(tokens)  # [1, batch_size, num_tokens * embedding_dim]
        tokens = tokens.permute(1, 0, 2).squeeze(1)  # [batch_size, num_tokens * embedding_dim]

        # Classification Head
        x = self.fc(tokens)  # [batch_size, 1]
        return torch.sigmoid(x).squeeze()

# =======================
# Step 8: Define Training and Evaluation Functions
# =======================
def train_model(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        categorical = batch['categorical'].to(device)
        binary = batch['binary'].to(device)
        continuous = batch['continuous'].to(device)
        targets = batch['target'].to(device)

        optimizer.zero_grad()
        outputs = model(categorical, binary, continuous)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * categorical.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate_model_nn(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation/test set.
    """
    model.eval()
    running_loss = 0.0
    preds = []
    targets_list = []
    with torch.no_grad():
        for batch in dataloader:
            categorical = batch['categorical'].to(device)
            binary = batch['binary'].to(device)
            continuous = batch['continuous'].to(device)
            targets = batch['target'].to(device)

            outputs = model(categorical, binary, continuous)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * categorical.size(0)

            preds.extend(outputs.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, np.array(preds), np.array(targets_list)

# =======================
# Step 9: Define Hyperparameter Grid
# =======================
param_grid = {
    'embedding_dim': [16, 32, 64],      # Dimensionality of embeddings for categorical features
    'hidden_dim': [32, 64, 128],         # Number of units in the hidden layers
    'num_heads': [2, 4, 6],           # Number of attention heads in each transformer layer
    'num_layers': [2, 4],          # Number of layers in the transformer encoder
    'dropout': [0.1, 0.5],           # Dropout rate for regularization
    'learning_rate': [1e-3],    # Learning rate for the optimizer
    'batch_size': [100],        # Number of samples per batch
    'epochs': [1, 3]                # Number of complete passes through the dataset
}

# =======================
# Step 10: Implement Nested Cross-Validation
# =======================
# Separate features and target
X = df_train_encoded.drop(columns=[target_col]).copy()
y = df_train_encoded[target_col].copy()

# Check distribution of target
print("Target distribution in the entire training set:")
print(y.value_counts(normalize=True))
print()

# Initialize outer and inner cross-validation strategies with stratification
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Initialize containers to collect metrics
validation_scores = []
validation_metrics = []
best_params = None
best_score = -np.inf
traintest_time_best_model = None
best_model = None
best_val_preds = None  # To store the best validation predictions
best_val_targets = None  # To store the best validation targets

# Device configuration - Ensure GPU is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure that a GPU is available and CUDA is installed.")
device = torch.device("cuda")
print(f"Using device: {device}\n")

# Start Nested Cross-Validation
for outer_fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), 1):
    print(f"=== Outer Fold {outer_fold} ===")
    X_train_outer, X_val_outer = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train_outer, y_val_outer = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

    # Ensure both classes are present in outer training and validation sets
    unique_train_outer = y_train_outer.unique()
    unique_val_outer = y_val_outer.unique()
    print(f"  Outer Fold {outer_fold} - Training target classes: {unique_train_outer}")
    print(f"  Outer Fold {outer_fold} - Validation target classes: {unique_val_outer}")

    # Handle Missing Values in Continuous Columns
    for col in continuous_cols:
        if X_train_outer[col].isnull().any():
            print(f"  Warning: Missing values found in '{col}'. Filling with median.")
            median_value = X_train_outer[col].median()
            X_train_outer[col].fillna(median_value, inplace=True)
            X_val_outer[col].fillna(median_value, inplace=True)

    # =====================
    # Inner Cross-Validation for Hyperparameter Tuning
    # =====================
    inner_scores = []
    inner_params = []
    print("  Starting Inner CV for Hyperparameter Tuning...")

    for params in ParameterGrid(param_grid):
        print(f"    Evaluating params: {params}")
        fold_scores = []

        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train_outer, y_train_outer), 1):
            # Split inner training and validation data
            X_train_inner = X_train_outer.iloc[inner_train_idx].copy()
            y_train_inner = y_train_outer.iloc[inner_train_idx].copy()
            X_val_inner = X_train_outer.iloc[inner_val_idx].copy()
            y_val_inner = y_train_outer.iloc[inner_val_idx].copy()

            # Handle Missing Values in Continuous Columns
            for col in continuous_cols:
                if X_train_inner[col].isnull().any():
                    print(f"    Warning: Missing values found in '{col}'. Filling with median.")
                    median_value = X_train_inner[col].median()
                    X_train_inner[col].fillna(median_value, inplace=True)
                    X_val_inner[col].fillna(median_value, inplace=True)

            # Scale continuous features within the inner fold
            X_train_inner_scaled = X_train_inner.copy()
            X_val_inner_scaled = X_val_inner.copy()

            scaler_inner = StandardScaler()
            X_train_inner_scaled[continuous_cols] = scaler_inner.fit_transform(X_train_inner[continuous_cols])
            X_train_inner_scaled[continuous_cols] = X_train_inner_scaled[continuous_cols].astype(np.float32)
            X_val_inner_scaled[continuous_cols] = scaler_inner.transform(X_val_inner[continuous_cols])
            X_val_inner_scaled[continuous_cols] = X_val_inner_scaled[continuous_cols].astype(np.float32)

            # Combine features and target
            df_train_inner_scaled = X_train_inner_scaled.copy()
            df_train_inner_scaled[target_col] = y_train_inner
            df_val_inner_scaled = X_val_inner_scaled.copy()
            df_val_inner_scaled[target_col] = y_val_inner

            # Create datasets
            try:
                train_dataset_inner = AttritionDataset(
                    df_train_inner_scaled,
                    categorical_cols=categorical_cols,
                    binary_cols=binary_cols,
                    continuous_cols=continuous_cols,
                    target_col=target_col
                )
                val_dataset_inner = AttritionDataset(
                    df_val_inner_scaled,
                    categorical_cols=categorical_cols,
                    binary_cols=binary_cols,
                    continuous_cols=continuous_cols,
                    target_col=target_col
                )
            except ValueError as ve:
                print(f"    ValueError during dataset creation: {ve}")
                print("    Skipping this inner fold due to dataset error.")
                continue

            # Create dataloaders
            train_loader_inner = DataLoader(
                train_dataset_inner,
                batch_size=params['batch_size'],
                shuffle=True,
                num_workers=0
            )
            val_loader_inner = DataLoader(
                val_dataset_inner,
                batch_size=params['batch_size'],
                shuffle=False,
                num_workers=0
            )

            # Initialize model
            model = SAINT(
                num_categorical=num_categorical_global,
                num_binary=len(binary_cols),
                num_continuous=len(continuous_cols),
                embedding_dim=params['embedding_dim'],
                hidden_dim=params['hidden_dim'],
                num_heads=params['num_heads'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            ).to(device)

            # Define loss and optimizer
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

            # Training Loop
            try:
                for epoch in range(params['epochs']):
                    train_loss = train_model(model, train_loader_inner, criterion, optimizer, device)
                    # Optionally: Print training loss
            except Exception as e:
                print(f"    Exception during training: {e}")
                print("    Skipping this hyperparameter set due to error.")
                continue

            # Evaluation on Inner Validation Set
            try:
                val_loss, val_preds, val_targets = evaluate_model_nn(model, val_loader_inner, criterion, device)
                # Check if both classes are present in val_targets
                unique_val_targets = np.unique(val_targets)
                if len(unique_val_targets) < 2:
                    print(f"    Warning: Inner Fold {inner_fold} Validation set contains only one class. Skipping ROC AUC computation.")
                    continue
                val_auc = roc_auc_score(val_targets, val_preds)
                fold_scores.append(val_auc)
            except ValueError as ve:
                print(f"    ValueError during evaluation: {ve}")
                print("    Skipping this hyperparameter set due to error.")
                continue

        if fold_scores:
            avg_val_auc = np.mean(fold_scores)
            print(f"    Average AUC for params {params}: {avg_val_auc:.4f}")
            inner_scores.append(avg_val_auc)
            inner_params.append(params)
        else:
            print(f"    No valid folds for params {params}")

    # Select Best Hyperparameters Based on Inner CV
    if not inner_scores:
        print("  No valid hyperparameter sets found in Inner CV. Skipping this Outer Fold.\n")
        continue  # Skip to next outer fold

    # Select the hyperparameters with the highest average AUC
    best_idx = np.argmax(inner_scores)
    best_params_fold = inner_params[best_idx]
    best_auc_fold = inner_scores[best_idx]

    print(f"  Best AUC in Inner CV: {best_auc_fold:.4f} with params: {best_params_fold}")

    # Train Model with Best Hyperparameters on Outer Training Fold
    start_time = time.time()

    # Scale Continuous Features Within Outer Fold
    X_train_outer_scaled = X_train_outer.copy()
    X_val_outer_scaled = X_val_outer.copy()

    scaler_outer = StandardScaler()
    X_train_outer_scaled[continuous_cols] = scaler_outer.fit_transform(X_train_outer[continuous_cols])
    X_train_outer_scaled[continuous_cols] = X_train_outer_scaled[continuous_cols].astype(np.float32)
    X_val_outer_scaled[continuous_cols] = scaler_outer.transform(X_val_outer[continuous_cols])
    X_val_outer_scaled[continuous_cols] = X_val_outer_scaled[continuous_cols].astype(np.float32)

    # Combine features and target
    df_train_outer_scaled = X_train_outer_scaled.copy()
    df_train_outer_scaled[target_col] = y_train_outer
    df_val_outer_scaled = X_val_outer_scaled.copy()
    df_val_outer_scaled[target_col] = y_val_outer

    # Create Datasets and Dataloaders with Scaled Data
    try:
        train_dataset_outer = AttritionDataset(
            df_train_outer_scaled,
            categorical_cols=categorical_cols,
            binary_cols=binary_cols,
            continuous_cols=continuous_cols,
            target_col=target_col
        )
        val_dataset_outer = AttritionDataset(
            df_val_outer_scaled,
            categorical_cols=categorical_cols,
            binary_cols=binary_cols,
            continuous_cols=continuous_cols,
            target_col=target_col
        )
    except ValueError as ve:
        print(f"  ValueError during outer fold dataset creation: {ve}")
        print("  Skipping this Outer Fold due to dataset error.\n")
        continue

    train_loader_outer = DataLoader(
        train_dataset_outer,
        batch_size=best_params_fold['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader_outer = DataLoader(
        val_dataset_outer,
        batch_size=best_params_fold['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # Initialize Model with Best Hyperparameters
    model_outer = SAINT(
        num_categorical=num_categorical_global,
        num_binary=len(binary_cols),
        num_continuous=len(continuous_cols),
        embedding_dim=best_params_fold['embedding_dim'],
        hidden_dim=best_params_fold['hidden_dim'],
        num_heads=best_params_fold['num_heads'],
        num_layers=best_params_fold['num_layers'],
        dropout=best_params_fold['dropout']
    ).to(device)

    # Define loss and optimizer
    criterion_outer = nn.BCELoss()
    optimizer_outer = torch.optim.Adam(model_outer.parameters(), lr=best_params_fold['learning_rate'])

    # Training Loop on Outer Fold
    try:
        for epoch in range(best_params_fold['epochs']):
            train_loss_outer = train_model(model_outer, train_loader_outer, criterion_outer, optimizer_outer, device)
            # Optionally: Print training loss
    except Exception as e:
        print(f"    Exception during training on Outer Fold {outer_fold}: {e}")
        print("    Skipping this Outer Fold due to training error.\n")
        continue

    end_time = time.time()
    training_time = end_time - start_time

    # Evaluate Model on Outer Validation Fold
    try:
        val_loss_outer, val_preds_outer, val_targets_outer = evaluate_model_nn(model_outer, val_loader_outer, criterion_outer, device)
        unique_val_targets_outer = np.unique(val_targets_outer)
        if len(unique_val_targets_outer) < 2:
            print(f"  Warning: Outer Fold {outer_fold} Validation set contains only one class. Skipping ROC AUC computation.\n")
            continue
        val_auc_outer = roc_auc_score(val_targets_outer, val_preds_outer)
        cm_outer = confusion_matrix(val_targets_outer, (val_preds_outer > 0.5).astype(int))
        cr_outer = classification_report(val_targets_outer, (val_preds_outer > 0.5).astype(int), output_dict=True)

        print(f"  Outer Fold {outer_fold} AUC: {val_auc_outer:.4f}")
        print(f"  Training Time for Fold {outer_fold}: {training_time:.2f} seconds\n")

        validation_scores.append(val_auc_outer)
        validation_metrics.append({
            'AUC': val_auc_outer,
            'Confusion Matrix': cm_outer.tolist(),
            'Classification Report': cr_outer
        })

        # Update best model if current fold has the best AUC
        if val_auc_outer > best_score:
            best_score = val_auc_outer
            best_params = best_params_fold
            traintest_time_best_model = training_time
            best_model = model_outer
            best_scaler = scaler_outer  # Save the scaler
            # Save validation predictions and targets for plotting
            best_val_preds = val_preds_outer
            best_val_targets = val_targets_outer
    except ValueError as ve:
        print(f"  ValueError during evaluation of Outer Fold {outer_fold}: {ve}")
        print("  ROC AUC score is not defined as the validation set contains only one class.")
        print("  Skipping this fold due to evaluation error.\n")
    except RuntimeError as e:
        print(f"  RuntimeError during evaluation of Outer Fold {outer_fold}: {e}")
        print("  Skipping this fold due to evaluation error.\n")

# =======================
# Step 11: Train the Best Model on the Entire Training Set
# =======================
if best_model is not None:
    print("=== Training the Best Model on the Entire Training Set and Evaluating on Test Set ===")
    
    # Start the timer before Step 11 begins
    traintest_start_time = time.time()
    
    # -----------------------
    # Step 11: Training
    # -----------------------
    print("=== Training the Best Model on the Entire Training Set ===")
    
    # Scale continuous features on the entire training set
    df_train_full_scaled = df_train_encoded.copy()
    scaler_full = StandardScaler()
    df_train_full_scaled[continuous_cols] = scaler_full.fit_transform(df_train_full_scaled[continuous_cols])
    df_train_full_scaled[continuous_cols] = df_train_full_scaled[continuous_cols].astype(np.float32)

    # Combine features and target
    df_train_full_scaled[target_col] = df_train_encoded[target_col]

    # Create Dataset and Dataloader
    try:
        train_dataset_full = AttritionDataset(
            df_train_full_scaled,
            categorical_cols=categorical_cols,
            binary_cols=binary_cols,
            continuous_cols=continuous_cols,
            target_col=target_col
        )
    except ValueError as ve:
        print(f"ValueError during full training dataset creation: {ve}")
        print("Skipping training on the entire training set due to dataset error.")
        exit()

    train_loader_full = DataLoader(
        train_dataset_full,
        batch_size=best_params['batch_size'],
        shuffle=True,
        num_workers=0
    )

    # Initialize the Best Model with Best Hyperparameters
    model_full = SAINT(
        num_categorical=num_categorical_global,
        num_binary=len(binary_cols),
        num_continuous=len(continuous_cols),
        embedding_dim=best_params['embedding_dim'],
        hidden_dim=best_params['hidden_dim'],
        num_heads=best_params['num_heads'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    ).to(device)

    # Define loss and optimizer
    criterion_full = nn.BCELoss()
    optimizer_full = torch.optim.Adam(model_full.parameters(), lr=best_params['learning_rate'])

    # Training Loop on Entire Training Set
    try:
        for epoch in range(best_params['epochs']):
            train_loss_full = train_model(model_full, train_loader_full, criterion_full, optimizer_full, device)
            print(f"  Epoch {epoch+1}/{best_params['epochs']} - Training Loss: {train_loss_full:.4f}")
    except Exception as e:
        print(f"Exception during training on Entire Training Set: {e}")
        print("Skipping further steps due to error.")
        exit()

    # -----------------------
    # Step 12: Evaluation on Training Set and Test Set
    # -----------------------
    # Evaluate the model on the training set
    print("=== Evaluating on the Training Set ===")
    try:
        train_loss_full, train_preds_full, train_targets_full = evaluate_model_nn(model_full, train_loader_full, criterion_full, device)
        train_auc_full = roc_auc_score(train_targets_full, train_preds_full)
        train_cm = confusion_matrix(train_targets_full, (train_preds_full > 0.5).astype(int))
        train_cr = classification_report(train_targets_full, (train_preds_full > 0.5).astype(int), output_dict=True)

        print(f"Training Set AUC: {train_auc_full:.4f}")
        print("Confusion Matrix:")
        print(train_cm)
        print("Classification Report:")
        print(classification_report(train_targets_full, (train_preds_full > 0.5).astype(int)))
    except Exception as e:
        print(f"Exception during training set evaluation: {e}")
        print("Skipping further steps due to evaluation error.")

    # -----------------------
    # Step 12: Evaluation on Test Set
    # -----------------------
    print("=== Evaluating on the Test Set ===")
    
    # Handle Missing Values in Continuous Columns
    for col in continuous_cols:
        if df_test_encoded[col].isnull().any():
            print(f"Warning: Missing values found in '{col}'. Filling with median.")
            median_value = df_test_encoded[col].median()
            df_test_encoded[col].fillna(median_value, inplace=True)

    # Scale continuous features using the scaler fitted on the entire training set
    df_test_scaled = df_test_encoded.copy()
    df_test_scaled[continuous_cols] = scaler_full.transform(df_test_scaled[continuous_cols])
    df_test_scaled[continuous_cols] = df_test_scaled[continuous_cols].astype(np.float32)
    
    # Combine features and target
    df_test_scaled[target_col] = df_test_encoded[target_col]
    
    # Create Dataset and Dataloader
    try:
        test_dataset = AttritionDataset(
            df_test_scaled,
            categorical_cols=categorical_cols,
            binary_cols=binary_cols,
            continuous_cols=continuous_cols,
            target_col=target_col
        )
    except ValueError as ve:
        print(f"ValueError during test dataset creation: {ve}")
        print("Skipping test set evaluation due to dataset error.")
        exit()

    test_loader = DataLoader(
        test_dataset,
        batch_size=best_params['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Evaluate the Model on Test Set
    try:
        test_loss, test_preds, test_targets = evaluate_model_nn(model_full, test_loader, criterion_full, device)

        # Check unique classes in test_targets
        unique_test_targets, counts_test_targets = np.unique(test_targets, return_counts=True)
        print("Unique classes in test_targets and their counts:")
        print(dict(zip(unique_test_targets, counts_test_targets)))

        if len(unique_test_targets) < 2:
            raise ValueError("Only one class present in y_true. ROC AUC score is not defined in that case.")

        test_auc = roc_auc_score(test_targets, test_preds)
        test_cm = confusion_matrix(test_targets, (test_preds > 0.5).astype(int))
        test_cr = classification_report(test_targets, (test_preds > 0.5).astype(int), output_dict=True)

        print(f"\nTest Set Evaluation:")
        print(f"AUC: {test_auc:.4f}")
        print("Confusion Matrix:")
        print(test_cm)
        print("Classification Report:")
        print(classification_report(test_targets, (test_preds > 0.5).astype(int)))
        
        # Save predicted probabilities and true labels to a CSV file
        save_path = r"" # set path to where objects should be saved
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        test_predictions = pd.DataFrame({
            'y_test': test_targets,
            'y_pred_proba': test_preds
        })
        test_predictions.to_csv(os.path.join(save_path, 'test_predictions_SAINT.csv'), index=False)
        print(f"Saved predicted probabilities and true labels for the test set to '{os.path.join(save_path, 'test_predictions_SAINT.csv')}'.\n")

    except ValueError as ve:
        print(f"ValueError during evaluation: {ve}")
        print("ROC AUC score is not defined as the test set contains only one class.")
        # Optionally, handle the exception by using alternative metrics
        print("Skipping ROC AUC and related metrics due to error.")
    except Exception as e:
        print(f"Exception during test set evaluation: {e}")
        print("Skipping ROC AUC and related metrics due to error.")

    # Stop the timer after Step 12 completes
    traintest_end_time = time.time()
    traintest_time = traintest_end_time - traintest_start_time
    print(f"Total time for Training and Testing (Steps 11 & 12): {traintest_time:.2f} seconds.\n")

# =======================
# Step 13: Collect and Save Metrics
# =======================
metrics = {
    'Validation Scores per Fold': validation_scores,
    'Validation Metrics Summary': {
        'AUC Mean': np.mean(validation_scores) if validation_scores else None,
        'AUC Std': np.std(validation_scores) if validation_scores else None
    },
    'Test Set': {
        'ROC-AUC': test_auc if 'test_auc' in locals() else None,
        'Precision': precision_score(test_targets, (test_preds > 0.5).astype(int), zero_division=0) if 'test_preds' in locals() else None,
        'Recall': recall_score(test_targets, (test_preds > 0.5).astype(int), zero_division=0) if 'test_preds' in locals() else None,
        'F1-Score': f1_score(test_targets, (test_preds > 0.5).astype(int), zero_division=0) if 'test_preds' in locals() else None,
        'Confusion Matrix': test_cm.tolist() if 'test_cm' in locals() else None,
        'Classification Report': test_cr if 'test_cr' in locals() else None
    },
    'traintest_time_seconds': traintest_time if 'traintest_time' in locals() else None,
    'Best Hyperparameters': best_params
}

# Ensure the output directory exists
os.makedirs(save_path, exist_ok=True)

# Save metrics to JSON
try:
    with open(os.path.join(save_path, 'metrics_SAINT.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print("\nSaved performance metrics to 'outputs_SAINT/metrics_SAINT.json'.")
except Exception as e:
    print(f"Error saving metrics to JSON: {e}")

# =======================
# Step 14: Plot ROC AUC and Confusion Matrix
# =======================
# Compute ROC curves
from sklearn.metrics import roc_curve

# Training ROC curve
train_fpr, train_tpr, _ = roc_curve(train_targets_full, train_preds_full)
# Validation ROC curve (from best outer fold)
if best_val_preds is not None and best_val_targets is not None:
    val_fpr, val_tpr, _ = roc_curve(best_val_targets, best_val_preds)
else:
    val_fpr, val_tpr = (None, None)
# Test ROC curve
if 'test_preds' in locals() and 'test_targets' in locals():
    test_fpr, test_tpr, _ = roc_curve(test_targets, test_preds)
else:
    test_fpr, test_tpr = (None, None)

# Plot ROC Curves
plt.figure(figsize=(8,6))
plt.plot(train_fpr, train_tpr, label=f'Training AUC = {train_auc_full:.4f}')
if val_fpr is not None and val_tpr is not None:
    plt.plot(val_fpr, val_tpr, label=f'Validation AUC = {best_score:.4f}')
if test_fpr is not None and test_tpr is not None:
    plt.plot(test_fpr, test_tpr, label=f'Test AUC = {test_auc:.4f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'roc_curves_SAINT.png'))
plt.show()

# Plot Confusion Matrix if available
if 'test_cm' in locals() and test_cm is not None:
    plt.figure(figsize=(6,5))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix_SAINT.png'))
    plt.show()

# =======================
# Step 15: Performance Metrics Plotting Function
# =======================
def plot_performance_metrics(metrics_train, metrics_validation, metrics_test, save_path):
    """
    Plots a bar chart comparing performance metrics between Training, Validation, and Test Sets.
    """
    metrics_names = ['ROC-AUC', 'Precision', 'Recall', 'F1-Score']
    train_values = [metrics_train.get(m, 0) for m in metrics_names]
    validation_values = [metrics_validation.get(m, 0) if metrics_validation.get(m, None) is not None else 0 for m in metrics_names]
    test_values = [metrics_test.get(m, 0) if metrics_test.get(m, None) is not None else 0 for m in metrics_names]

    x = np.arange(len(metrics_names))  # label locations
    width = 0.25  # width of the bars

    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width, train_values, width, label='Training Set', color='skyblue')
    rects2 = ax.bar(x, validation_values, width, label='Validation Set', color='lightgreen')
    rects3 = ax.bar(x + width, test_values, width, label='Test Set', color='salmon')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Performance Metrics by Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()

    # Attach a text label above each bar, displaying its height
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'performance_metrics_SAINT.png'))
    plt.show()
    print("Displayed Performance Metrics Bar Chart with Training, Validation, and Test Sets.")

# Collect performance metrics
# Training Set Metrics
train_auc_full = train_auc_full if 'train_auc_full' in locals() else None
metrics_train = {
    'ROC-AUC': train_auc_full,
    'Precision': precision_score(train_targets_full, (train_preds_full > 0.5).astype(int), zero_division=0),
    'Recall': recall_score(train_targets_full, (train_preds_full > 0.5).astype(int), zero_division=0),
    'F1-Score': f1_score(train_targets_full, (train_preds_full > 0.5).astype(int), zero_division=0)
}

# Validation Set Metrics
if best_val_preds is not None and best_val_targets is not None:
    metrics_validation = {
        'ROC-AUC': best_score,
        'Precision': precision_score(best_val_targets, (best_val_preds > 0.5).astype(int), zero_division=0),
        'Recall': recall_score(best_val_targets, (best_val_preds > 0.5).astype(int), zero_division=0),
        'F1-Score': f1_score(best_val_targets, (best_val_preds > 0.5).astype(int), zero_division=0)
    }
else:
    metrics_validation = {
        'ROC-AUC': None,
        'Precision': None,
        'Recall': None,
        'F1-Score': None
    }

# Test Set Metrics
if 'test_auc' in locals() and test_auc is not None:
    metrics_test = {
        'ROC-AUC': test_auc,
        'Precision': precision_score(test_targets, (test_preds > 0.5).astype(int), zero_division=0),
        'Recall': recall_score(test_targets, (test_preds > 0.5).astype(int), zero_division=0),
        'F1-Score': f1_score(test_targets, (test_preds > 0.5).astype(int), zero_division=0)
    }
else:
    metrics_test = {
        'ROC-AUC': None,
        'Precision': None,
        'Recall': None,
        'F1-Score': None
    }

# Plot Performance Metrics
plot_performance_metrics(metrics_train, metrics_validation, metrics_test, save_path)

# =======================
# Step 16: Perform SHAP Analysis on Test Set
# =======================
def model_predict(data, model, categorical_cols, binary_cols, continuous_cols, scaler):
    """
    Predict function for SHAP, compatible with categorical, binary, and continuous inputs.
    Applies scaling to continuous features before prediction.
    """
    model.eval()
    with torch.no_grad():
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Ensure data types are float32
        data = data.astype(np.float32)
        
        # Separate categorical, binary, and continuous features
        categorical_data = torch.tensor(
            data[:, :len(categorical_cols)],
            dtype=torch.long
        ).to(next(model.parameters()).device)
        
        binary_data = torch.tensor(
            data[:, len(categorical_cols):len(categorical_cols) + len(binary_cols)],
            dtype=torch.float32
        ).to(next(model.parameters()).device)
        
        continuous_raw = data[:, len(categorical_cols) + len(binary_cols):]
        # Apply scaling to continuous features
        continuous_scaled = scaler.transform(continuous_raw)
        continuous_data = torch.tensor(
            continuous_scaled,
            dtype=torch.float32
        ).to(next(model.parameters()).device)
    
        # Predict using the model
        outputs = model(categorical_data, binary_data, continuous_data)
        
        # Ensure outputs is numpy array and at least 1D
        outputs_np = outputs.cpu().numpy()
        outputs_np = np.atleast_1d(outputs_np)
        return outputs_np

def perform_shap_analysis_nn(model, X, feature_names, categorical_cols, binary_cols, continuous_cols, title_suffix, scaler, batch_size=10):
    """
    Performs SHAP analysis on a neural network model using SamplingExplainer with batch processing.
    """
    # Convert input data to numpy array if it is a DataFrame
    if isinstance(X, pd.DataFrame):
        X = X.values.astype(np.float32)
    else:
        X = X.astype(np.float32)

    # Limit the number of samples to speed up SHAP calculations
    num_samples = 50  # Adjust this number based on available memory and computation time
    if X.shape[0] > num_samples:
        X_sample = shap.sample(X, num_samples, random_state=42)
    else:
        X_sample = X

    # Initialize SHAP SamplingExplainer with custom model_predict function
    explainer = shap.SamplingExplainer(
        lambda data: model_predict(data, model, categorical_cols, binary_cols, continuous_cols, scaler),
        X_sample  # Using sample as background
    )
    print("Initialized SHAP SamplingExplainer.\n")

    # Compute SHAP values in batches to reduce memory usage
    shap_values = []
    num_batches = int(np.ceil(X_sample.shape[0] / batch_size))
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, X_sample.shape[0])
        X_batch = X_sample[start_idx:end_idx]
        shap_values_batch = explainer.shap_values(X_batch)
        shap_values.append(shap_values_batch)
        print(f"Calculated SHAP values for batch {i+1}/{num_batches}.")

    # Concatenate SHAP values from all batches
    shap_values = np.concatenate(shap_values, axis=0)
    print("Calculated SHAP values for all batches.\n")

    # Aggregate SHAP values by feature
    aggregated_shap = {feature: shap_values[:, i] for i, feature in enumerate(feature_names)}

    # Plot aggregated SHAP values
    plot_aggregated_shap(aggregated_shap, title_suffix)

def plot_aggregated_shap(aggregated_shap, title_suffix):
    """
    Plots the aggregated SHAP values for original features.
    """
    # Calculate mean absolute SHAP values for importance
    aggregated_mean = {feature: np.mean(np.abs(shap_vals)) for feature, shap_vals in aggregated_shap.items()}

    # Sort features by mean SHAP values in descending order
    sorted_features = sorted(aggregated_mean, key=aggregated_mean.get, reverse=True)
    sorted_values = [aggregated_mean[feature] for feature in sorted_features]

    # Plot Mean |SHAP value|
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_values, y=sorted_features, palette='viridis')
    plt.xlabel('Mean |SHAP value|')
    plt.title(f'Aggregated SHAP Values - {title_suffix}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'aggregated_shap_mean_{title_suffix.lower().replace(" ", "_")}.png'))
    plt.show()
    print(f"Displayed aggregated SHAP mean plot for {title_suffix}.\n")

    # Calculate mean signed SHAP values for positive/negative contributions
    aggregated_mean_signed = {feature: np.mean(shap_vals) for feature, shap_vals in aggregated_shap.items()}

    # Sort features by signed SHAP values in descending order
    sorted_features_signed = sorted(aggregated_mean_signed, key=aggregated_mean_signed.get, reverse=True)
    sorted_values_signed = [aggregated_mean_signed[feature] for feature in sorted_features_signed]

    # Plot Mean SHAP value with positive and negative contributions
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_values_signed, y=sorted_features_signed, palette='coolwarm')
    plt.xlabel('Mean SHAP value')
    plt.title(f'Aggregated SHAP Values (Signed) - {title_suffix}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'aggregated_shap_signed_{title_suffix.lower().replace(" ", "_")}.png'))
    plt.show()
    print(f"Displayed aggregated SHAP signed plot for {title_suffix}.\n")

# Combine feature names from all column groups
feature_names = categorical_cols + binary_cols + continuous_cols

# Perform SHAP analysis
if best_model is not None:
    print("Performing SHAP analysis on test set...\n")
    perform_shap_analysis_nn(
        model=model_full,                            # The best model trained in the pipeline
        X=df_test_scaled[feature_names],             # Scaled and encoded test data
        feature_names=feature_names,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
        continuous_cols=continuous_cols,
        title_suffix="Test Set",
        scaler=scaler_full,                          # Pass the scaler here
        batch_size=10                                # Adjust batch size based on memory
    )
else:
    print("No best model available for SHAP analysis.\n")
