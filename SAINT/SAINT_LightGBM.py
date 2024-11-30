# =======================
# Step 0: Define Output Directory, Device, and Import Libraries
# =======================
import os
import json
import time
import random

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

from lightgbm import LGBMClassifier  # Use LightGBM

# Enable anomaly detection for better debugging
torch.autograd.set_detect_anomaly(True)

# Define the output directory
output_dir = r""
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory set to: {output_dir}\n")

# Define Device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

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

# Adjusted file paths
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
            df[feature] = df[feature].map(mapping).fillna(0).astype(np.int32)
            print(f"Converted '{feature}' to numeric with mapping: {mapping}. Unmapped values set to 0.")
        else:
            print(f"Warning: Binary feature '{feature}' not found in DataFrame.")

    # Encode target variable
    if df[target_col].dtype == 'object' or isinstance(df[target_col].iloc[0], str):
        df[target_col] = df[target_col].map({'Left': 1, 'Stayed': 0}).fillna(0).astype(np.int32)
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

        df[col] = encoder.transform(df[[col]]).astype(np.int32)

        # Verify encoding
        min_val = df[col].min()
        max_val = df[col].max()
        expected_max = num_categorical[categorical_cols.index(col)] - 1 if fit else encoders[col].categories_[0].size
        print(f"Feature '{col}' - Min: {min_val}, Max: {max_val}, Expected Max: {expected_max}")
        assert df[col].min() >= 0, f"Error: Feature '{col}' has negative indices after encoding."
        assert df[col].max() <= expected_max, f"Error: Feature '{col}' has indices exceeding the embedding size."

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
        self.categorical = df[categorical_cols].values.astype(np.int32)
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
        return torch.sigmoid(x).squeeze(), tokens  # Also return tokens (embeddings)

# =======================
# Step 8: Define Training and Evaluation Functions
# =======================
def train_model_epoch(model, dataloader, criterion, optimizer, device):
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
        outputs, _ = model(categorical, binary, continuous)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * categorical.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate_model_epoch(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation/test set.
    """
    model.eval()
    running_loss = 0.0
    preds = []
    targets_list = []
    embeddings_list = []
    with torch.no_grad():
        for batch in dataloader:
            categorical = batch['categorical'].to(device)
            binary = batch['binary'].to(device)
            continuous = batch['continuous'].to(device)
            targets = batch['target'].to(device)

            outputs, embeddings = model(categorical, binary, continuous)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * categorical.size(0)

            preds.extend(outputs.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
            embeddings_list.append(embeddings.cpu().numpy())
    epoch_loss = running_loss / len(dataloader.dataset)
    embeddings_array = np.vstack(embeddings_list)
    return epoch_loss, np.array(preds), np.array(targets_list), embeddings_array

# =======================
# Step 9: Set SAINT Hyperparameters
# =======================
saint_params = {
    'embedding_dim': 16,     # Dimensionality of embeddings for categorical features
    'hidden_dim': 64,        # Number of units in the hidden layers
    'num_heads': 2,          # Number of attention heads in each transformer layer
    'num_layers': 2,         # Number of layers in the transformer encoder
    'dropout': 0.1,          # Dropout rate for regularization
    'learning_rate': 1e-3,   # Learning rate for the optimizer
    'batch_size': 100,        # Number of samples per batch
    'epochs': 1              # Number of complete passes through the dataset
}

# =======================
# Step 10: Prepare Data for SAINT
# =======================
# Separate features and target
X = df_train_encoded.drop(columns=[target_col]).copy()
y = df_train_encoded[target_col].copy()

# Handle Missing Values in Continuous Columns
for col in continuous_cols:
    if X[col].isnull().any():
        print(f"Warning: Missing values found in '{col}'. Filling with median.")
        median_value = X[col].median()
        X[col].fillna(median_value, inplace=True)

# Scale continuous features for training data
scaler = StandardScaler()
X[continuous_cols] = scaler.fit_transform(X[continuous_cols])

# Scale continuous features for test data
df_test_scaled = df_test_encoded.copy()
for col in continuous_cols:
    if df_test_scaled[col].isnull().any():
        print(f"Warning: Missing values found in '{col}'. Filling with median.")
        median_value = df_test_scaled[col].median()
        df_test_scaled[col].fillna(median_value, inplace=True)

df_test_scaled[continuous_cols] = scaler.transform(df_test_scaled[continuous_cols])
df_test_scaled[continuous_cols] = df_test_scaled[continuous_cols].astype(np.float32)

print("Scaled continuous features for both training and test datasets.\n")



# =======================
# Step 11: Nested Cross-Validation for SAINT and LightGBM
# =======================
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# LightGBM hyperparameter grid
lgbm_param_grid = {
    'learning_rate': [0.1, 0.05],
    'num_leaves': [5, 10, 20],
    'n_estimators': [200, 300, 400],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'reg_alpha': [0.1],
    'reg_lambda': [5],
    'min_child_samples': [1, 3],
    'min_split_gain': [0.1]
}

# Initialize variables to collect results
train_preds_all = []
train_targets_all = []
val_preds_all = []
val_targets_all = []

validation_scores_per_fold = []
best_model_training_time = None
cross_val_start_time = time.time()  # Start timing cross-validation

best_val_auc = -np.inf
best_model_state = None
best_hyperparams = None

# Initialize the scaler (already fitted)
# scaler = StandardScaler()  # Already fitted and saved

print(f"Using device: {device}\n")

for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), 1):
    print(f"===== Fold {fold} =====")
    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

    # Prepare datasets
    df_train_cv = X_train_cv.copy()
    df_train_cv[target_col] = y_train_cv
    df_val_cv = X_val_cv.copy()
    df_val_cv[target_col] = y_val_cv

    train_dataset = AttritionDataset(
        df_train_cv,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
        continuous_cols=continuous_cols,
        target_col=target_col
    )
    val_dataset = AttritionDataset(
        df_val_cv,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
        continuous_cols=continuous_cols,
        target_col=target_col
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=saint_params['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=saint_params['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # Initialize SAINT model
    model = SAINT(
        num_categorical=num_categorical_global,
        num_binary=len(binary_cols),
        num_continuous=len(continuous_cols),
        embedding_dim=saint_params['embedding_dim'],
        hidden_dim=saint_params['hidden_dim'],
        num_heads=saint_params['num_heads'],
        num_layers=saint_params['num_layers'],
        dropout=saint_params['dropout']
    ).to(device)

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=saint_params['learning_rate'])

    # Train SAINT model
    start_time = time.time()
    for epoch in range(saint_params['epochs']):
        train_loss = train_model_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds_saint, val_targets_saint, _ = evaluate_model_epoch(model, val_loader, criterion, device)
        val_auc_saint = roc_auc_score(val_targets_saint, val_preds_saint)
        print(f"Epoch {epoch+1}/{saint_params['epochs']} - SAINT Validation AUC: {val_auc_saint:.4f}")
    training_time = time.time() - start_time

    # Extract embeddings for train and validation sets
    _, train_preds_saint, train_targets_saint, train_embeddings = evaluate_model_epoch(model, train_loader, criterion, device)
    _, val_preds_saint, val_targets_saint, val_embeddings = evaluate_model_epoch(model, val_loader, criterion, device)

    # Nested cross-validation for LightGBM hyperparameter tuning
    lgbm_best_auc = -np.inf
    lgbm_best_params = None
    for params in ParameterGrid(lgbm_param_grid):
        aucs = []
        for inner_train_idx, inner_val_idx in inner_cv.split(train_embeddings, train_targets_saint):
            X_train_inner, X_val_inner = train_embeddings[inner_train_idx], train_embeddings[inner_val_idx]
            y_train_inner, y_val_inner = train_targets_saint[inner_train_idx], train_targets_saint[inner_val_idx]

            lgbm_model = LGBMClassifier(
                **params,
                random_state=42,
                n_jobs=-1
            )
            lgbm_model.fit(X_train_inner, y_train_inner)
            val_preds = lgbm_model.predict_proba(X_val_inner)[:, 1]
            val_auc = roc_auc_score(y_val_inner, val_preds)
            aucs.append(val_auc)

        mean_auc = np.mean(aucs)
        if mean_auc > lgbm_best_auc:
            lgbm_best_auc = mean_auc
            lgbm_best_params = params

    print(f"Best LightGBM params for fold {fold}: {lgbm_best_params} with AUC: {lgbm_best_auc:.4f}")

    # Train LightGBM model with best params on current fold's training data
    lgbm_model = LGBMClassifier(
        **lgbm_best_params,
        random_state=42,
        n_jobs=-1
    )
    lgbm_model.fit(train_embeddings, train_targets_saint)

    # Evaluate on validation set
    val_preds_lgbm = lgbm_model.predict_proba(val_embeddings)[:, 1]
    val_auc_lgbm = roc_auc_score(val_targets_saint, val_preds_lgbm)
    print(f"LightGBM Validation AUC for fold {fold}: {val_auc_lgbm:.4f}\n")

    validation_scores_per_fold.append(val_auc_lgbm)

    # Save best model
    if val_auc_lgbm > best_val_auc:
        best_val_auc = val_auc_lgbm
        best_model_state = {
            'saint_model_state': model.state_dict(),
            'lgbm_model': lgbm_model,
            'scaler': scaler  # Keep the scaler in memory
        }
        best_model_training_time = training_time
        best_hyperparams = {
            'embedding_dim': saint_params['embedding_dim'],
            'hidden_dim': saint_params['hidden_dim'],
            'num_heads': saint_params['num_heads'],
            'num_layers': saint_params['num_layers'],
            'dropout': saint_params['dropout'],
            'learning_rate': saint_params['learning_rate'],
            'batch_size': saint_params['batch_size'],
            'epochs': saint_params['epochs'],
            'lgbm_params': lgbm_best_params
        }

    # Collect predictions and targets
    train_preds_all.extend(train_preds_saint)
    train_targets_all.extend(train_targets_saint)
    val_preds_all.extend(val_preds_lgbm)
    val_targets_all.extend(val_targets_saint)

cross_val_time_end = time.time()
total_cross_val_time = cross_val_time_end - cross_val_start_time
print(f"Total cross-validation time: {total_cross_val_time:.2f} seconds\n")

# =======================
# Step 12: Train Best Model on Full Training Data and Predict on Test Set
# =======================
# Load best hyperparameters
best_lgbm_params = best_hyperparams['lgbm_params']

# Initialize SAINT model with best hyperparameters
final_model = SAINT(
    num_categorical=num_categorical_global,
    num_binary=len(binary_cols),
    num_continuous=len(continuous_cols),
    embedding_dim=best_hyperparams['embedding_dim'],
    hidden_dim=best_hyperparams['hidden_dim'],
    num_heads=best_hyperparams['num_heads'],
    num_layers=best_hyperparams['num_layers'],
    dropout=best_hyperparams['dropout']
).to(device)

# Load SAINT model state
final_model.load_state_dict(best_model_state['saint_model_state'])

# Define loss and optimizer
final_criterion = nn.BCELoss()
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_hyperparams['learning_rate'])

# Prepare full training dataset
full_train_dataset = AttritionDataset(
    df_train_encoded.copy(),
    categorical_cols=categorical_cols,
    binary_cols=binary_cols,
    continuous_cols=continuous_cols,
    target_col=target_col
)
full_train_loader = DataLoader(
    full_train_dataset,
    batch_size=best_hyperparams['batch_size'],
    shuffle=True,
    num_workers=0
)

# Prepare LightGBM model with best hyperparameters
final_lgbm_model = LGBMClassifier(
    **best_lgbm_params,
    random_state=42,
    n_jobs=-1
)

# Measure time for training on full data and prediction
final_training_start_time = time.time()

# Train SAINT model on full training data
print("Training SAINT model on full training data...")
for epoch in range(best_hyperparams['epochs']):
    train_loss = train_model_epoch(final_model, full_train_loader, final_criterion, final_optimizer, device)
    print(f"Epoch {epoch+1}/{best_hyperparams['epochs']} - Training Loss: {train_loss:.4f}")

# Extract embeddings for full training data
_, full_train_preds_saint, full_train_targets_saint, full_train_embeddings = evaluate_model_epoch(final_model, full_train_loader, final_criterion, device)

# Train LightGBM on full training embeddings
print("Training LightGBM model on full training embeddings...")
final_lgbm_model.fit(full_train_embeddings, full_train_targets_saint)

# Prepare test dataset
# Handle Missing Values in Continuous Columns (already handled and scaled)
test_dataset = AttritionDataset(
    df_test_encoded.copy(),
    categorical_cols=categorical_cols,
    binary_cols=binary_cols,
    continuous_cols=continuous_cols,
    target_col=target_col
)
test_loader = DataLoader(
    test_dataset,
    batch_size=best_hyperparams['batch_size'],
    shuffle=False,
    num_workers=0
)

# Extract embeddings for test set
_, test_preds_saint_full, test_targets_saint_full, test_embeddings_full = evaluate_model_epoch(final_model, test_loader, final_criterion, device)
print(f"Extracted embeddings for test set. Shape: {test_embeddings_full.shape}")

# Predict on test set using LightGBM
test_preds_lgbm_full = final_lgbm_model.predict_proba(test_embeddings_full)[:, 1]

# Measure end time
final_training_end_time = time.time()
final_training_time = final_training_end_time - final_training_start_time

print(f"Final training and prediction time: {final_training_time:.2f} seconds")

# =======================
# Step 13: Save Test Predictions
# =======================
# Save LightGBM predictions and true labels to a CSV file
test_predictions_SAINT_LightGBM = pd.DataFrame({
    'y_test': test_targets_saint_full,
    'y_pred_proba_LightGBM': test_preds_lgbm_full
})
test_predictions_SAINT_LightGBM.to_csv(os.path.join(output_dir, 'test_predictions_SAINT_LightGBM.csv'), index=False)
print(f"Saved LightGBM predicted probabilities and true labels for the test set to '{os.path.join(output_dir, 'test_predictions_SAINT_LightGBM.csv')}'.\n")

# =======================
# Step 14: Plot ROC Curves
# =======================
# Compute ROC curves
train_fpr, train_tpr, _ = roc_curve(train_targets_all, train_preds_all)
val_fpr, val_tpr, _ = roc_curve(val_targets_all, val_preds_all)
test_fpr, test_tpr, _ = roc_curve(test_targets_saint_full, test_preds_lgbm_full)

# Plot ROC Curves
plt.figure(figsize=(8,6))
plt.plot(train_fpr, train_tpr, label=f'Training AUC = {roc_auc_score(train_targets_all, train_preds_all):.4f}')
plt.plot(val_fpr, val_tpr, label=f'Validation AUC = {roc_auc_score(val_targets_all, val_preds_all):.4f}')
plt.plot(test_fpr, test_tpr, label=f'Test AUC = {roc_auc_score(test_targets_saint_full, test_preds_lgbm_full):.4f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roc_curves_SAINT_LightGBM.png'))
plt.show()
print("Displayed ROC Curves for Training, Validation, and Test Sets.\n")

# =======================
# Step 15: Collect and Save Metrics
# =======================
# Training Set Metrics
train_precision = precision_score(train_targets_all, (np.array(train_preds_all) > 0.5).astype(int))
train_recall = recall_score(train_targets_all, (np.array(train_preds_all) > 0.5).astype(int))
train_f1 = f1_score(train_targets_all, (np.array(train_preds_all) > 0.5).astype(int))
metrics_train = {
    'ROC-AUC': roc_auc_score(train_targets_all, train_preds_all),
    'Precision': train_precision,
    'Recall': train_recall,
    'F1-Score': train_f1
}

# Validation Set Metrics
val_precision = precision_score(val_targets_all, (np.array(val_preds_all) > 0.5).astype(int))
val_recall = recall_score(val_targets_all, (np.array(val_preds_all) > 0.5).astype(int))
val_f1 = f1_score(val_targets_all, (np.array(val_preds_all) > 0.5).astype(int))
metrics_validation = {
    'ROC-AUC': roc_auc_score(val_targets_all, val_preds_all),
    'Precision': val_precision,
    'Recall': val_recall,
    'F1-Score': val_f1
}

# Test Set Metrics
if 'Attrition' in df_test_scaled.columns:
    test_precision = precision_score(test_targets_saint_full, (test_preds_lgbm_full > 0.5).astype(int))
    test_recall = recall_score(test_targets_saint_full, (test_preds_lgbm_full > 0.5).astype(int))
    test_f1 = f1_score(test_targets_saint_full, (test_preds_lgbm_full > 0.5).astype(int))
    test_cm = confusion_matrix(test_targets_saint_full, (test_preds_lgbm_full > 0.5).astype(int))
    test_cr = classification_report(test_targets_saint_full, (test_preds_lgbm_full > 0.5).astype(int), output_dict=True)
    metrics_test = {
        'ROC-AUC': roc_auc_score(test_targets_saint_full, test_preds_lgbm_full),
        'Precision': test_precision,
        'Recall': test_recall,
        'F1-Score': test_f1,
        'Confusion Matrix': test_cm.tolist(),
        'Classification Report': test_cr
    }
else:
    metrics_test = {
        'ROC-AUC': None,
        'Precision': None,
        'Recall': None,
        'F1-Score': None,
        'Confusion Matrix': None,
        'Classification Report': None
    }

# Collect Validation Metrics Summary
validation_metrics_summary = {
    'AUC Mean': np.mean(validation_scores_per_fold),
    'AUC Std': np.std(validation_scores_per_fold)
}

# Collect Final Training and Prediction Time
final_model_time = final_training_time  # Time measured during final training and prediction

# Save Metrics to JSON
metrics = {
    'Validation Scores per Fold': validation_scores_per_fold,
    'Validation Metrics Summary': validation_metrics_summary,
    'Test Set': {
        'AUC': metrics_test['ROC-AUC'],
        'Confusion Matrix': metrics_test['Confusion Matrix'],
        'Classification Report': metrics_test['Classification Report']
    },
    'Best Model Training Time (seconds)': best_model_training_time,
    'Final Model Training and Prediction Time (seconds)': final_model_time,
    'Total Cross-Validation Time (seconds)': total_cross_val_time,
    'Best Hyperparameters': best_hyperparams
}

# Save metrics to JSON
try:
    with open(os.path.join(output_dir, 'metrics_SAINT_LightGBM.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved performance metrics to '{os.path.join(output_dir, 'metrics_SAINT_LightGBM.json')}'.")
except Exception as e:
    print(f"Error saving metrics to JSON: {e}")

# =======================
# Step 16: Plot Performance Metrics
# =======================
def plot_performance_metrics(metrics_train, metrics_validation, metrics_test, output_directory):
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
    plt.savefig(os.path.join(output_directory, 'performance_metrics_SAINT_LightGBM.png'))
    plt.show()
    print("Displayed Performance Metrics Bar Chart with Training, Validation, and Test Sets.")

plot_performance_metrics(metrics_train, metrics_validation, metrics_test, output_dir)

# =======================
# Step 17: Perform SHAP Analysis on Test Set (LightGBM Model)
# =======================
def perform_shap_analysis_lgbm(model, X, feature_names, title_suffix, output_directory):
    """
    Performs SHAP analysis on a LightGBM model.
    """
    # Create a TreeExplainer for the LightGBM model
    explainer = shap.TreeExplainer(model)
    print("Initialized SHAP TreeExplainer for LightGBM model.\n")

    # Limit the number of samples to speed up SHAP calculations
    num_samples = 1000  # Adjust based on memory and computation time
    if X.shape[0] > num_samples:
        X_sample = shap.sample(X, num_samples, random_state=42)
    else:
        X_sample = X

    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)
    print("Calculated SHAP values for the LightGBM model.\n")

    # Aggregate SHAP values by feature
    aggregated_shap = {feature: shap_values[:, i] for i, feature in enumerate(feature_names)}

    # Plot aggregated SHAP values
    plot_aggregated_shap(aggregated_shap, title_suffix, output_directory)

def plot_aggregated_shap(aggregated_shap, title_suffix, output_directory):
    """
    Plots the aggregated SHAP values for original features.
    """
    # Calculate mean absolute SHAP values for importance
    aggregated_mean = {feature: np.mean(np.abs(shap_vals)) for feature, shap_vals in aggregated_shap.items()}

    # Sort features by mean SHAP values in descending order
    sorted_features = sorted(aggregated_mean, key=aggregated_mean.get, reverse=True)
    sorted_values = [aggregated_mean[feature] for feature in sorted_features]

    # Select top 20 most influential features
    top_n = 20
    sorted_features_top = sorted_features[:top_n]
    sorted_values_top = sorted_values[:top_n]

    # Plot Mean |SHAP value| for top 20 features
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_values_top, y=sorted_features_top, palette='viridis')
    plt.xlabel('Mean |SHAP value|')
    plt.title(f'Aggregated SHAP Values - Top {top_n} Features ({title_suffix})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'aggregated_shap_mean_top_{top_n}_{title_suffix.lower().replace(" ", "_")}.png'))
    plt.show()
    print(f"Displayed aggregated SHAP mean plot for top {top_n} features in {title_suffix}.\n")

    # Calculate mean signed SHAP values for positive/negative contributions
    aggregated_mean_signed = {feature: np.mean(shap_vals) for feature, shap_vals in aggregated_shap.items()}

    # Sort features by signed SHAP values in descending order
    sorted_features_signed = sorted(aggregated_mean_signed, key=aggregated_mean_signed.get, reverse=True)
    sorted_values_signed = [aggregated_mean_signed[feature] for feature in sorted_features_signed]

    # Select top 10 positive and top 10 negative features
    top_n_signed = 10
    top_positive_features = sorted_features_signed[:top_n_signed]
    top_positive_values = sorted_values_signed[:top_n_signed]

    top_negative_features = sorted_features_signed[-top_n_signed:]
    top_negative_values = sorted_values_signed[-top_n_signed:]

    # Combine positive and negative features
    top_features_signed = top_positive_features + top_negative_features
    top_values_signed = top_positive_values + top_negative_values

    # Plot Mean SHAP value with positive and negative contributions
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_values_signed, y=top_features_signed, palette='coolwarm')
    plt.xlabel('Mean SHAP value')
    plt.title(f'Aggregated SHAP Values (Signed) - Top {top_n_signed} Positive & Negative Features ({title_suffix})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, f'aggregated_shap_signed_top_{2*top_n_signed}_{title_suffix.lower().replace(" ", "_")}.png'))
    plt.show()
    print(f"Displayed aggregated SHAP signed plot for top {top_n_signed} positive and negative features in {title_suffix}.\n")

# Prepare data for SHAP analysis
# Use the embeddings from the final LightGBM model as input
# Since LightGBM uses the embeddings directly, feature names correspond to the embedding dimensions
embedding_feature_names = [f'Embedding_{i}' for i in range(X.shape[1])]

# Perform SHAP analysis on LightGBM model
print("Performing SHAP analysis on LightGBM model...\n")
perform_shap_analysis_lgbm(
    model=final_lgbm_model,
    X=test_embeddings_full,
    feature_names=embedding_feature_names,
    title_suffix="Test Set",
    output_directory=output_dir
)

# =======================
# Step 18: Conclusion
# =======================
print("SAINT-LightGBM pipeline with nested cross-validation and timing adjustments has been successfully executed. Results have been saved to the specified output directory.")
