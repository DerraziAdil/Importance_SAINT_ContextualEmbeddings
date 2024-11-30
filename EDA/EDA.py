# =======================
# Step 1: Import Libraries
# =======================
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better aesthetics
sns.set(style='whitegrid')

# =======================
# Step 2: Set Random Seeds for Reproducibility (Optional for EDA)
# =======================
def set_random_seeds(seed=42):
    """
    Sets random seeds for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)

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

# Replace with actual file paths
df_train = load_dataset('Attrition (75K)/train.csv')
df_test = load_dataset('Attrition (75K)/test.csv')

# =======================
# Step 4: Perform EDA Function
# =======================
def perform_eda(df, dataset_name='Dataset'): 
    """
    Performs exploratory data analysis on the dataset.
    
    Parameters:
    - df (pd.DataFrame): The dataframe to analyze.
    - dataset_name (str): A name to identify the dataset (e.g., 'Training Set', 'Test Set').
    """
    print(f"\n=======================\nPerforming EDA on {dataset_name}\n=======================")
    # Summary statistics
    print(f"\nSummary Statistics for {dataset_name}:")
    print(df.describe(include='all'))
    
    # Missing values
    missing_values = df.isnull().sum()
    print(f"\nMissing values per column in {dataset_name}:\n{missing_values}")
    
    # Visualize class distribution of the target variable if it exists
    if 'Attrition' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x='Attrition')
        plt.title(f'Class Distribution of Attrition in {dataset_name}')
        plt.xlabel('Attrition')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'class_distribution_attrition_{dataset_name}.png')
        plt.close()
        print(f"Saved class distribution plot as 'class_distribution_attrition_{dataset_name}.png'.\n")
    
    # Identify categorical features
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    if 'Attrition' in categorical_features:
        categorical_features.remove('Attrition')  # Exclude the target variable
    
    # Identify binary features (non-object type with two unique values)
    binary_features = [col for col in df.columns if df[col].nunique() == 2 and col not in categorical_features + ['Attrition']]
    
    # Combine categorical and binary features
    cat_bin_features = categorical_features + binary_features
    
    # Print value counts for categorical and binary features
    for col in cat_bin_features:
        print(f"\nValue counts for '{col}' in {dataset_name}:")
        print(df[col].value_counts())
        print("-" * 40)
    
    # Correlation heatmap for numerical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if numerical_features:
        plt.figure(figsize=(12, 10))
        corr_matrix = df[numerical_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='viridis', linewidths=0.5)
        plt.title(f'Correlation Heatmap for {dataset_name}')
        plt.tight_layout()
        plt.savefig(f'correlation_heatmap_{dataset_name}.png')
        plt.close()
        print(f"Saved correlation heatmap as 'correlation_heatmap_{dataset_name}.png'.\n")
    else:
        print(f"No numerical features to plot correlation heatmap for {dataset_name}.\n")
        
    # Boxplot analysis for outliers in numerical features
    for feature in numerical_features:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[feature])
        plt.title(f'Boxplot of {feature} in {dataset_name}')
        plt.xlabel(feature)
        plt.tight_layout()
        plt.savefig(f'boxplot_{feature}_{dataset_name}.png')
        plt.close()
        # Removed print statement about saving boxplots

# Perform EDA on the training data
perform_eda(df_train, dataset_name='Training Set')

# Perform EDA on the test data
perform_eda(df_test, dataset_name='Test Set')
