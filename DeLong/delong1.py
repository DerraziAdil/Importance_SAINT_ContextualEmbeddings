# ==========================
# DeLong's Test Implementation
# ==========================
import os
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from itertools import combinations
from scipy.stats import spearmanr

def load_predictions(file_path):
    """
    Load predictions from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: y_true and y_pred_proba arrays.
    """
    df = pd.read_csv(file_path)
    if 'y_test' not in df.columns or 'y_pred_proba' not in df.columns:
        raise ValueError(f"CSV file {file_path} must contain 'y_test' and 'y_pred_proba' columns.")
    y_true = df['y_test'].values
    y_pred_proba = df['y_pred_proba'].values
    return y_true, y_pred_proba

def compute_roc_auc(y_true, y_pred_proba):
    """
    Compute ROC-AUC score.

    Args:
        y_true (np.array): True binary labels.
        y_pred_proba (np.array): Predicted probabilities.

    Returns:
        float: ROC-AUC score.
    """
    return roc_auc_score(y_true, y_pred_proba)

# DeLong's Test Functions
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    # Note: +1 is due to Python using 0-based indexing
    T2[J] = T + 1
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def calc_pvalue(aucs, sigma):
    """Computes p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       p-value (float)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    p_value = 2 * scipy.stats.norm.sf(z)  # two-tailed
    return p_value.item()  # Ensure p_value is a scalar float

def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1]), "Ground truth must be binary (0 and 1)."
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count

def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes p-value for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    Returns:
       p-value (float)
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    p_value = calc_pvalue(aucs, delongcov)
    return p_value

# ==========================
# Main Script with DeLong's Test
# ==========================
def main():
    """
    Main function to perform DeLong's comparisons between multiple models.
    """
    # List of prediction files (ensure paths are correct)
    prediction_files = [
        'test_predictions_SAINT_XGBoost.csv',
        'test_predictions_SAINT_LightGBM.csv',
        'test_predictions_LightGBM.csv',
        'test_predictions_SAINT.csv',
        'test_predictions_XGBoost.csv'
    ]

    # Dictionary to store predictions and true labels
    model_predictions = {}

    # Load predictions
    for file in prediction_files:
        if not os.path.exists(file):
            print(f"File {file} does not exist. Skipping.")
            continue
        model_name = file.replace('test_predictions_', '').replace('.csv', '')
        try:
            y_true, y_pred_proba = load_predictions(file)
            model_predictions[model_name] = {
                'y_true': y_true,
                'y_pred_proba': y_pred_proba,
                'roc_auc': compute_roc_auc(y_true, y_pred_proba)
            }
            print(f"Loaded {file} successfully. AUC: {model_predictions[model_name]['roc_auc']:.4f}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not model_predictions:
        print("No valid prediction files loaded. Exiting.")
        return

    # Print ROC-AUC scores
    print("\nROC-AUC Scores:")
    for model_name, data in model_predictions.items():
        print(f"{model_name}: {data['roc_auc']:.4f}")
    print("\n")

    # Compare models using DeLong's test
    model_names = list(model_predictions.keys())
    results = []

    print("Statistical Comparison of ROC-AUC Scores (DeLong's Test):")
    for model_i, model_j in combinations(model_names, 2):
        y_true_i = model_predictions[model_i]['y_true']
        y_true_j = model_predictions[model_j]['y_true']
        y_pred_i = model_predictions[model_i]['y_pred_proba']
        y_pred_j = model_predictions[model_j]['y_pred_proba']

        # Ensure that the true labels are the same
        if not np.array_equal(y_true_i, y_true_j):
            print(f"Mismatch in true labels between {model_i} and {model_j}. Skipping comparison.")
            continue
        y_true = y_true_i

        try:
            # Compute AUCs
            auc1 = model_predictions[model_i]['roc_auc']
            auc2 = model_predictions[model_j]['roc_auc']

            # Perform DeLong's test
            p_value = delong_roc_test(y_true, y_pred_i, y_pred_j)

            # Handle cases where p_value might be extremely small or zero
            if p_value > 0:
                log10_p_value = np.log10(p_value)
            else:
                # Assign a very small value to represent p-value effectively zero
                log10_p_value = -300

            significance = 'Significant' if p_value < 0.05 else 'Not Significant'

            print(f"Comparison between {model_i} and {model_j}:")
            print(f"  AUC of {model_i}: {auc1:.4f}")
            print(f"  AUC of {model_j}: {auc2:.4f}")
            print(f"  p-value: {p_value:.6f} ({significance})")
            print(f"  log10(p-value): {log10_p_value:.2f}\n")

            # Store results
            results.append({
                'Model 1': model_i,
                'Model 2': model_j,
                'AUC 1': auc1,
                'AUC 2': auc2,
                'p-value': p_value,
                'log10(p-value)': log10_p_value,
                'Significance': significance
            })

            # Optional: Visualizing ROC Curves (Not part of DeLong's test)
            # You can plot ROC curves here if desired.

        except Exception as e:
            print(f"Error comparing {model_i} and {model_j}: {e}\n")

    if not results:
        print("No comparisons were made.")
        return

    # Create a DataFrame of results
    results_df = pd.DataFrame(results)

    # Reorder columns to place 'log10(p-value)' between 'p-value' and 'Significance'
    cols = ['Model 1', 'Model 2', 'AUC 1', 'AUC 2', 'p-value', 'log10(p-value)', 'Significance']
    results_df = results_df[cols]

    print("Summary of Statistical Tests:")
    print(results_df)

    # Optional: Save the results to a CSV file
    results_df.to_csv('delong_test_results.csv', index=False)
    print("Saved statistical test results to 'delong_test_results.csv'.")

if __name__ == "__main__":
    main()
