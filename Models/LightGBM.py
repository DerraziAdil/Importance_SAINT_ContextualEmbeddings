# =======================
# Step 1: Load Required Libraries
# =======================
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import json
import time
import random
import warnings
import os  # Import os module for path handling

# Suppress warnings for cleaner output (use cautiously)
warnings.filterwarnings("ignore", category=UserWarning)

# =======================
# Step 2: Set Random Seeds for Reproducibility
# =======================
def set_random_seeds(seed=42):
    """
    Sets random seeds for various libraries to ensure reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    # LightGBM uses the 'random_state' parameter for reproducibility

set_random_seeds(42)

# =======================
# Step 3: Load Datasets
# =======================
def load_dataset(filepath):
    """
    Loads the dataset from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        if 'Employee ID' in df.columns:
            df = df.drop(columns=['Employee ID'])
            print(f"Dataset '{filepath}' loaded and 'Employee ID' column dropped successfully.")
        else:
            print(f"Dataset '{filepath}' loaded successfully. 'Employee ID' column not found; skipping drop.")
        return df
    except Exception as e:
        print(f"Error loading dataset '{filepath}': {e}")
        raise

# Replace 'train.csv' and 'test.csv' with the full paths to your files
dataset_directory = r""
df_train = load_dataset(os.path.join(dataset_directory, 'train.csv'))
df_test = load_dataset(os.path.join(dataset_directory, 'test.csv'))

# =======================
# Step 4: Data Preparation (One-Hot Encoding)
# =======================
def prepare_data(df, encoder=None):
    """
    Prepares the data for modeling by one-hot encoding categorical variables,
    mapping binary features to numeric, and identifying continuous numeric features.
    """
    # Initialize binary and categorical features 
    binary_features = ['Overtime', 'Remote Work', 'Leadership Opportunities', 'Innovation Opportunities']
    categorical_features = [
        'Job Role', 'Work-Life Balance', 'Job Satisfaction', 'Performance Rating',
        'Education Level', 'Marital Status', 'Job Level', 'Company Size',
        'Company Reputation', 'Employee Recognition'
    ]

    # Convert binary categorical features to numeric
    for feature in binary_features:
        if feature in df.columns:
            df[feature] = df[feature].map({'Yes': 1, 'No': 0})
            print(f"Converted '{feature}' to numeric.")
        else:
            print(f"Warning: Binary feature '{feature}' not found in the dataset.")

    # Convert 'Gender' feature to numeric (Male: 1, Female: 0)
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        print("Converted 'Gender' to numeric.\n")
    else:
        print("Warning: 'Gender' column not found in the dataset.\n")

    # One-hot encode categorical features
    if encoder is None:
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        categorical_encoded = encoder.fit_transform(df[categorical_features])
        print("Fitted OneHotEncoder and transformed categorical features.\n")
    else:
        categorical_encoded = encoder.transform(df[categorical_features])
        print("Transformed categorical features using the pre-fitted OneHotEncoder.\n")

    # Define feature names after encoding
    try:
        feature_names = encoder.get_feature_names_out(categorical_features)
    except AttributeError:
        feature_names = encoder.get_feature_names(categorical_features)
        print("Using 'get_feature_names' instead of 'get_feature_names_out' due to older scikit-learn version.\n")

    # Create a DataFrame for the encoded categorical features
    categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=feature_names).astype(int)
    print("Converted one-hot encoded columns to integer type.\n")

    # Drop original categorical columns and concatenate encoded columns
    df = df.drop(columns=categorical_features).reset_index(drop=True)
    df = pd.concat([df, categorical_encoded_df], axis=1)
    print("Dropped original categorical columns and concatenated encoded columns.\n")

    # Define features and target
    if 'Attrition' not in df.columns:
        raise ValueError("Target column 'Attrition' not found in the dataset.")
    X = df.drop(columns=['Attrition'])  # Features
    y = df['Attrition'].map({'Left': 1, 'Stayed': 0})  # Target

    # Handle missing target values
    if y.isnull().sum() > 0:
        print("Warning: Some 'Attrition' values could not be mapped and are NaN. Filling NaN with 0.")
        y = y.fillna(0).astype(int)
        print(f"Unique values in 'Attrition' after filling NaN: {y.unique()}\n")

    # Identify continuous numeric features for scaling
    # Exclude binary features and one-hot encoded features
    onehot_features = feature_names.tolist()
    continuous_features = X.select_dtypes(include=['float64', 'int64']).columns.difference(binary_features + onehot_features).tolist()
    print(f"Identified continuous features for scaling: {continuous_features}\n")

    # Create feature mapping for SHAP aggregation
    feature_mapping = {}
    for feature in X.columns:
        if feature in onehot_features:
            # Extract base feature name before the first underscore
            base_name = feature.split('_')[0]
            feature_mapping[feature] = base_name
        else:
            feature_mapping[feature] = feature
    print("Created feature mapping for SHAP aggregation.\n")

    return X, y, encoder, binary_features, categorical_features, continuous_features, feature_mapping

# Prepare training data
X_train, y_train, encoder, binary_features, categorical_features, continuous_features, feature_mapping = prepare_data(df_train)

# Prepare testing data using the fitted encoder
X_test, y_test, _, _, _, _, _ = prepare_data(
    df_test, 
    encoder=encoder
)

# =======================
# Step 5: SHAP Analysis Functions
# =======================

def aggregate_shap_values(shap_values, feature_names, feature_mapping):
    """
    Aggregates SHAP values by their original feature names.

    Parameters:
    - shap_values (numpy.ndarray): SHAP values for all samples and features.
    - feature_names (list): List of feature names corresponding to SHAP values.
    - feature_mapping (dict): Mapping from encoded feature names to original feature names.

    Returns:
    - aggregated_shap (dict): Dictionary mapping original feature names to aggregated SHAP values.
    """
    aggregated_shap = {}
    for i, feature in enumerate(feature_names):
        base_name = feature_mapping.get(feature, feature)
        if base_name not in aggregated_shap:
            aggregated_shap[base_name] = shap_values[:, i]
        else:
            aggregated_shap[base_name] += shap_values[:, i]
    print("Aggregated SHAP values.\n")
    return aggregated_shap

def plot_aggregated_shap(aggregated_shap, title_suffix, output_dir):
    """
    Plots the aggregated SHAP values for original features.

    Parameters:
    - aggregated_shap (dict): Dictionary mapping original feature names to aggregated SHAP values.
    - title_suffix (str): Suffix to add to the plot titles.
    - output_dir (str): Directory to save the plots.
    """
    # Calculate mean absolute SHAP values for importance
    aggregated_mean = {feature: np.mean(np.abs(shap_vals)) 
                       for feature, shap_vals in aggregated_shap.items()}

    # Sort features by mean SHAP values in descending order
    sorted_features = sorted(aggregated_mean, key=aggregated_mean.get, reverse=True)
    sorted_values = [aggregated_mean[feature] for feature in sorted_features]

    # Plot Mean |SHAP value|
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_values, y=sorted_features, palette='viridis')
    plt.xlabel('Mean |SHAP value|')
    plt.title(f'Aggregated SHAP Values - {title_suffix}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'aggregated_SHAP_LightGBM_{title_suffix}.png'))
    plt.close()
    print(f"Saved aggregated SHAP mean plot for {title_suffix}.\n")

    # Calculate mean signed SHAP values for positive/negative contributions
    aggregated_mean_signed = {feature: np.mean(shap_vals) 
                              for feature, shap_vals in aggregated_shap.items()}

    # Sort features by signed SHAP values in descending order
    sorted_features_signed = sorted(aggregated_mean_signed, key=aggregated_mean_signed.get, reverse=True)
    sorted_values_signed = [aggregated_mean_signed[feature] for feature in sorted_features_signed]

    # Plot Mean SHAP value with positive and negative contributions
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_values_signed, y=sorted_features_signed, palette='coolwarm')
    plt.xlabel('Mean SHAP value')
    plt.title(f'Aggregated SHAP Values (Signed) - {title_suffix}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'aggregated_signed_SHAP_LightGBM_{title_suffix}.png'))
    plt.close()
    print(f"Saved aggregated SHAP signed plot for {title_suffix}.\n")

def perform_shap_analysis(model, X, feature_mapping, output_dir):
    """
    Performs SHAP analysis on the dataset using TreeExplainer.

    Parameters:
    - model (Pipeline): Trained model pipeline.
    - X (pd.DataFrame): Features to explain (e.g., test set).
    - feature_mapping (dict): Mapping from encoded feature names to original feature names.
    - output_dir (str): Directory to save SHAP plots.
    """
    # Sample a subset for SHAP to reduce computation
    X_shap = X.sample(frac=0.5, random_state=42)
    print(f"Sampled 50% for SHAP analysis: {X_shap.shape[0]} samples.\n")
    
    # Access the LightGBM model within the pipeline
    lgbm_model = model.named_steps['lgbm']
    
    # Access the preprocessor and transform the dataset
    preprocessor = model.named_steps['preprocessor']
    X_transformed = preprocessor.transform(X_shap)  # Transform X for SHAP explanation

    # Initialize SHAP TreeExplainer with the trained LightGBM model
    explainer = shap.TreeExplainer(lgbm_model)
    print("Initialized SHAP TreeExplainer.\n")

    # Calculate SHAP values for the sampled set
    shap_values = explainer.shap_values(X_transformed)
    print("Calculated SHAP values.\n")

    # For binary classification, shap_values is a list with two arrays
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use the SHAP values for the positive class

    # Aggregate SHAP values by original features
    aggregated_shap = aggregate_shap_values(shap_values, X_shap.columns, feature_mapping)
    print("Aggregated SHAP values by original features.\n")

    # Plot aggregated SHAP values
    plot_aggregated_shap(aggregated_shap, "LightGBM", output_dir)

# =======================
# Step 6: Nested Cross-Validation and Model Training Function
# =======================
def perform_nested_cross_validation(X, y, param_grid, continuous_features, output_dir):
    """
    Performs nested cross-validation for hyperparameter tuning and model evaluation.
    
    Parameters:
    - X (pd.DataFrame): Features.
    - y (pd.Series): Target variable.
    - param_grid (dict): Hyperparameter grid for GridSearchCV.
    - continuous_features (list): List of continuous feature names.
    - output_dir (str): Directory to save outputs.

    Returns:
    - best_model (Pipeline): Trained model with best hyperparameters.
    - best_params (dict): Best hyperparameters found.
    - validation_scores (list): List of validation metrics per fold.
    - avg_validation_metrics (dict): Average validation metrics across folds.
    - best_val_model (Pipeline): Best model based on validation ROC-AUC.
    - best_val_X_val (pd.DataFrame): Validation features for the best model.
    - best_val_y_val (pd.Series): Validation targets for the best model.
    - training_time_best_model (float): Time taken to train the best model.
    """
    # Set up cross-validation strategies
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Lists to hold validation scores
    validation_scores = []

    # Variables to keep track of the best model and its parameters
    best_val_roc_auc = 0
    best_val_model = None
    best_val_X_val = None
    best_val_y_val = None
    best_params = None

    print("Starting Nested Cross-Validation for Hyperparameter Tuning...")
    start_time = time.time()
    for fold, (train_ix, val_ix) in enumerate(outer_cv.split(X, y), 1):
        print(f"\n=== Outer Fold {fold} ===")
        X_train_cv, X_val_cv = X.iloc[train_ix], X.iloc[val_ix]
        y_train_cv, y_val_cv = y.iloc[train_ix], y.iloc[val_ix]

        # Scale continuous features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), continuous_features)
            ],
            remainder='passthrough'  # Keep binary and one-hot encoded features as is
        )
        preprocessor.fit(X_train_cv)
        print("  Preprocessor fitted on the training data.")

        # Create pipeline with preprocessor and LightGBM classifier
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('lgbm', LGBMClassifier(
                random_state=42
            ))
        ])

        # Set up GridSearchCV
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        # Run GridSearchCV
        print("  Starting GridSearchCV...")
        grid_search.fit(X_train_cv, y_train_cv)
        print("  GridSearchCV completed.")

        # Get the best model and its parameters
        best_model_cv = grid_search.best_estimator_
        best_params_cv = grid_search.best_params_

        # Evaluate on validation set
        y_val_proba = best_model_cv.predict_proba(X_val_cv)[:, 1]
        y_val_pred = best_model_cv.predict(X_val_cv)

        val_roc_auc = roc_auc_score(y_val_cv, y_val_proba)
        val_precision = precision_score(y_val_cv, y_val_pred, zero_division=0)
        val_recall = recall_score(y_val_cv, y_val_pred, zero_division=0)
        val_f1 = f1_score(y_val_cv, y_val_pred, zero_division=0)

        validation_scores.append({
            'roc_auc': val_roc_auc,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1
        })

        print(f"  Fold {fold} Metrics:")
        print(f"    ROC-AUC: {val_roc_auc:.4f}")
        print(f"    Precision: {val_precision:.4f}")
        print(f"    Recall: {val_recall:.4f}")
        print(f"    F1-Score: {val_f1:.4f}")

        # Update the best model if this fold's model is better
        if val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = val_roc_auc
            best_val_model = best_model_cv
            best_val_X_val = X_val_cv.copy()
            best_val_y_val = y_val_cv.copy()
            best_params = best_params_cv

    end_time = time.time()
    print(f"\nNested Cross-Validation Time: {end_time - start_time:.2f} seconds")

    # Calculate average validation metrics
    avg_validation_metrics = {
        'ROC-AUC': np.mean([score['roc_auc'] for score in validation_scores]),
        'Precision': np.mean([score['precision'] for score in validation_scores]),
        'Recall': np.mean([score['recall'] for score in validation_scores]),
        'F1-Score': np.mean([score['f1'] for score in validation_scores])
    }

    print("\nAverage Validation Metrics:")
    for metric, value in avg_validation_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Now, train the final model with best hyperparameters on entire training data
    print("\nTraining the best model on the entire training set...")

    # Scale continuous features on the full training data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_features)
        ],
        remainder='passthrough'
    )
    preprocessor.fit(X)
    print("Preprocessor fitted on the full training data.")

    # Extract the hyperparameters for LGBMClassifier
    lgbm_best_params = {k.replace('lgbm__', ''): v for k, v in best_params.items()}

    # Create pipeline with the best hyperparameters
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('lgbm', LGBMClassifier(
            **lgbm_best_params,
            random_state=42
        ))
    ])

    # Start the timer for steps 11 and 12 before training and evaluation
    traintest_start_time = time.time()

    # Fit the final model and measure the time taken
    fit_start_time = time.time()
    pipeline.fit(X, y)
    fit_end_time = time.time()
    training_time_best_model = fit_end_time - fit_start_time
    print(f"Training Time for Best Model: {training_time_best_model:.2f} seconds")

    best_model = pipeline

    # Stop the timer after training and before evaluation
    traintest_end_time = time.time()
    # Calculate the total time for training and testing
    traintest_time = traintest_end_time - traintest_start_time
    print(f"Total time for Training and Testing (Steps 11 & 12): {traintest_time:.2f} seconds.\n")

    return (
        best_model, 
        best_params, 
        validation_scores, 
        avg_validation_metrics, 
        best_val_model, 
        best_val_X_val, 
        best_val_y_val, 
        traintest_time
    )

# =======================
# Step 7: Evaluation Metrics Function
# =======================
def evaluate_model(model, X, y, dataset_type='Dataset'):
    """
    Evaluates the model on the provided dataset and returns performance metrics.

    Parameters:
    - model (Pipeline): Trained model pipeline.
    - X (pd.DataFrame): Features.
    - y (pd.Series): Target variable.
    - dataset_type (str): Descriptor for the dataset (e.g., 'Training Set').

    Returns:
    - metrics (dict): Dictionary containing evaluation metrics.
    """
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    # Calculate metrics
    roc_auc = roc_auc_score(y, y_pred_proba)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred)

    # Print classification report
    print(f"\nClassification Report for {dataset_type}:")
    print(classification_report(y, y_pred, zero_division=0))

    # Return metrics
    return {
        'ROC-AUC': roc_auc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': cm.tolist()  # Convert to list for JSON serialization
    }

# =======================
# Step 8: ROC Curve Plotting Function
# =======================
def plot_roc_curves(y_train, y_proba_train, y_test, y_proba_test, best_val_model, X_val, y_val, output_dir):
    """
    Plots ROC curves for Training, Best Validation, and Test Sets.

    Parameters:
    - y_train (pd.Series): Training target.
    - y_proba_train (numpy.ndarray): Training predicted probabilities.
    - y_test (pd.Series): Test target.
    - y_proba_test (numpy.ndarray): Test predicted probabilities.
    - best_val_model (Pipeline): Best validation model.
    - X_val (pd.DataFrame): Validation features.
    - y_val (pd.Series): Validation targets.
    - output_dir (str): Directory to save the ROC plot.
    """
    # ROC for Training Set
    fpr_train, tpr_train, _ = roc_curve(y_train, y_proba_train)
    roc_auc_train = roc_auc_score(y_train, y_proba_train)
    
    # ROC for Best Validation Set
    y_val_proba = best_val_model.predict_proba(X_val)[:, 1]
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_proba)
    roc_auc_val = roc_auc_score(y_val, y_val_proba)
    
    # ROC for Test Set
    fpr_test, tpr_test, _ = roc_curve(y_test, y_proba_test)
    roc_auc_test = roc_auc_score(y_test, y_proba_test)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, label=f'Training Set (AUC = {roc_auc_train:.2f})', color='blue')
    plt.plot(fpr_val, tpr_val, label=f'Best Validation Set (AUC = {roc_auc_val:.2f})', color='green')
    plt.plot(fpr_test, tpr_test, label=f'Test Set (AUC = {roc_auc_test:.2f})', color='red')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Training, Best Validation, and Test Sets')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ROC_LightGBM.png'))
    plt.close()
    print("Saved ROC Curves.\n")

# =======================
# Step 9: Performance Metrics Plotting Function
# =======================
def plot_performance_metrics(metrics_train, metrics_validation, metrics_test, output_dir):
    """
    Plots a bar chart comparing performance metrics between Training, Validation, and Test Sets.

    Parameters:
    - metrics_train (dict): Training set metrics.
    - metrics_validation (dict): Validation set metrics.
    - metrics_test (dict): Test set metrics.
    - output_dir (str): Directory to save the metrics plot.
    """
    metrics_names = ['ROC-AUC', 'Precision', 'Recall', 'F1-Score']
    train_values = [metrics_train.get(m, 0) for m in metrics_names]
    validation_values = [metrics_validation.get(m, 0) for m in metrics_names]
    test_values = [metrics_test.get(m, 0) for m in metrics_names]

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
    plt.savefig(os.path.join(output_dir, 'Metrics_LightGBM.png'))
    plt.close()
    print("Saved Performance Metrics Bar Chart.\n")

# =======================
# Step 10: Main Execution Flow
# =======================
def main():
    """
    Main function to execute the machine learning pipeline:
    - Train and tune the model using nested cross-validation.
    - Evaluate the model on training and test sets.
    - Perform comprehensive SHAP analysis on the test set.
    - Store performance metrics.
    """
    # Define the output directory
    output_dir = r''
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory set to: {output_dir}\n")

    # Define hyperparameter grid 
    param_grid = {
        'lgbm__learning_rate': [0.05],
        'lgbm__num_leaves': [1, 3, 5],
        'lgbm__n_estimators': [300],
        'lgbm__subsample': [0.8],
        'lgbm__colsample_bytree': [0.8],
        'lgbm__reg_alpha': [0.1],
        'lgbm__reg_lambda': [5],
        'lgbm__min_child_samples': [1, 3, 5],
        'lgbm__min_split_gain': [0.1]
    }

    # Start the timer before Step 11 and 12
    traintest_start_time = time.time()

    # Perform nested cross-validation on the training data
    best_model, best_params, validation_scores, avg_validation_metrics, best_val_model, best_val_X_val, best_val_y_val, traintest_time = perform_nested_cross_validation(
        X_train, y_train, param_grid, continuous_features, output_dir
    )

    # Evaluate the model on the training set (Step 12)
    print("\nEvaluating the best model on the Training Set...")
    metrics_train = evaluate_model(best_model, X_train, y_train, dataset_type='Training Set')

    # Evaluate the model on the test set (Step 12)
    print("\nEvaluating the best model on the Test Set...")
    metrics_test = evaluate_model(best_model, X_test, y_test, dataset_type='Test Set')

    # Stop the timer after Step 12 completes
    traintest_end_time = time.time()
    traintest_time = traintest_end_time - traintest_start_time
    print(f"Total time for Training and Testing (Steps 11 & 12): {traintest_time:.2f} seconds.\n")

    # Calculate statistical summaries for validation scores
    validation_df = pd.DataFrame(validation_scores)
    validation_stats = validation_df.describe().to_dict()

    # Save Metrics to JSON
    metrics = {
        'Training Set': metrics_train,
        'Validation Set': avg_validation_metrics,
        'Validation Scores per Fold': validation_scores,
        'Validation Metrics Summary': validation_stats,
        'Test Set': metrics_test,
        'traintest_time': traintest_time,  # Updated to record the combined time
        'Best Hyperparameters': best_params
    }

    try:
        with open(os.path.join(output_dir, 'metrics_LightGBM.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        print("Saved performance metrics to 'metrics_LightGBM.json'.")
    except Exception as e:
        print(f"Error saving metrics to JSON: {e}")

    # Perform SHAP analysis on the test set
    print("\nPerforming SHAP analysis on the Test Set...")
    perform_shap_analysis(best_model, X_test, feature_mapping, output_dir)

    # Plot ROC Curves and Performance Metrics
    y_proba_train = best_model.predict_proba(X_train)[:, 1]
    y_proba_test = best_model.predict_proba(X_test)[:, 1]

    # Save predicted probabilities and true labels to a CSV file
    test_predictions = pd.DataFrame({
        'y_test': y_test.values,
        'y_pred_proba': y_proba_test
    })
    test_predictions.to_csv(os.path.join(output_dir, 'test_predictions_LightGBM.csv'), index=False)
    print("Saved predicted probabilities and true labels for the test set to 'test_predictions_LightGBM.csv'.\n")

    # Plot ROC Curves with Best Validation Model's ROC
    plot_roc_curves(y_train, y_proba_train, y_test, y_proba_test, best_val_model, best_val_X_val, best_val_y_val, output_dir)

    # Plot Performance Metrics Bar Chart
    plot_performance_metrics(metrics_train, avg_validation_metrics, metrics_test, output_dir)

if __name__ == "__main__":
    main()
