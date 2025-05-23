import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

def plot_correlation_heatmaps(df_orig, df_aug, numeric_features):
    """
    Plots side-by-side correlation heatmaps for numeric features in two DataFrames.

    Parameters:
    - df_orig: Original DataFrame
    - df_aug: Augmented DataFrame
    - numeric_features: List of column names with numeric data to include in the heatmap
    """
    if not numeric_features:
        print("No numeric features provided for correlation heatmaps.")
        return

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap for original data
    sns.heatmap(df_orig[numeric_features].corr(), annot=True, cmap='coolwarm', ax=axes[0])
    axes[0].set_title('Original Data Correlation')

    # Heatmap for augmented data
    sns.heatmap(df_aug[numeric_features].corr(), annot=True, cmap='coolwarm', ax=axes[1])
    axes[1].set_title('Augmented Data Correlation')

    plt.tight_layout()
    plt.show()

def compare_numeric_feature(feature, df_orig, df_aug):
    """
    Compares the distribution of a numeric feature between two DataFrames using the KS test
    and visualizes it with histograms.

    Parameters:
    - feature: Name of the numeric feature to compare
    - df_orig: Original DataFrame
    - df_aug: Augmented DataFrame
    """
    # Skip if feature is not numeric
    if not pd.api.types.is_numeric_dtype(df_orig[feature]):
        print(f"Skipping '{feature}': not a numeric feature.")
        return

    # Perform Kolmogorov-Smirnov test
    ks_stat, p_val = ks_2samp(df_orig[feature].dropna(), df_aug[feature].dropna())
    print(f"\nFeature: {feature}")
    print(f"  KS Statistic: {ks_stat:.4f}")
    print(f"  P-value: {p_val:.4f} (High p-value indicates similar distributions)")

    # Plot histograms with KDE
    plt.figure(figsize=(10, 6))
    sns.histplot(df_orig[feature], label='Original', kde=True, color='blue', alpha=0.5)
    sns.histplot(df_aug[feature], label='Augmented', kde=True, color='orange', alpha=0.5)
    plt.title(f'Distribution Comparison: {feature}')
    plt.legend()
    plt.show()

def compute_js_divergence(df_orig, df_aug, num_bins=10):
    """
    Computes Jensen-Shannon divergence for each feature between two DataFrames.

    Parameters:
    - df_orig: Original DataFrame
    - df_aug: Augmented DataFrame
    - num_bins: Number of bins for discretizing numeric features (default=10)

    Returns:
    - dict: JS divergence values for each feature
    """
    js_divergences = {}

    for col in df_orig.columns:
        if pd.api.types.is_numeric_dtype(df_orig[col]):
            # Handle numeric features by discretizing into bins
            discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform')
            orig_binned = discretizer.fit_transform(df_orig[[col]].dropna()).flatten()
            aug_binned = discretizer.fit_transform(df_aug[[col]].dropna()).flatten()

            # Compute probability distributions
            orig_hist, _ = np.histogram(orig_binned, bins=num_bins, density=True)
            aug_hist, _ = np.histogram(aug_binned, bins=num_bins, density=True)

            # Calculate JS divergence
            js_div = jensenshannon(orig_hist, aug_hist)
        else:
            # Handle categorical features with label encoding
            le = LabelEncoder()
            orig_encoded = le.fit_transform(df_orig[col].astype(str).dropna())
            aug_encoded = le.fit_transform(df_aug[col].astype(str).dropna())

            # Compute probability distributions
            all_values = np.union1d(np.unique(orig_encoded), np.unique(aug_encoded))
            orig_counts, _ = np.histogram(orig_encoded, bins=len(all_values), density=True)
            aug_counts, _ = np.histogram(aug_encoded, bins=len(all_values), density=True)

            # Calculate JS divergence
            js_div = jensenshannon(orig_counts, aug_counts)

        js_divergences[col] = js_div if not np.isnan(js_div) else 0.0  # Handle NaN cases

    # Display results
    print("\nJensen-Shannon Divergence for Each Feature:")
    for feature, value in js_divergences.items():
        print(f"  {feature}: {value:.4f}")
    avg_js = np.mean(list(js_divergences.values()))
    print(f"\nAverage JS Divergence: {avg_js:.4f} (Lower values indicate higher similarity)")

    return js_divergences

def assess_data_similarity(df_orig, df_aug):
    """
    Assesses similarity between original and augmented DataFrames by comparing marginal
    and joint distributions.

    Parameters:
    - df_orig: Original DataFrame
    - df_aug: Augmented DataFrame
    """
    # Identify numeric and categorical features
    numeric_features = df_orig.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df_orig.select_dtypes(exclude=[np.number]).columns.tolist()

    print("Feature Types Identified:")
    print(f"  Numeric: {numeric_features}")
    print(f"  Categorical: {categorical_features}")

    # Compare marginal distributions for numeric features
    print("\n--- Marginal Distribution Comparison (Numeric Features) ---")
    if numeric_features:
        for feature in numeric_features:
            compare_numeric_feature(feature, df_orig, df_aug)
    else:
        print("No numeric features to compare.")

    # Compare joint distributions
    print("\n--- Joint Distribution Comparison ---")
    plot_correlation_heatmaps(df_orig, df_aug, numeric_features)

    # Compute JS divergence for all features
    print("\n--- Jensen-Shannon Divergence ---")
    compute_js_divergence(df_orig, df_aug)

if __name__ == "__main__":
    # Placeholder for synthetic data generation (assuming it exists elsewhere)
    try:
        df_orig, df_aug = generate_synthetic_data_best(n=1000, m=5, noise_scale=0.5, random_state=42)
        df_orig = pd.DataFrame(df_orig)
        df_aug = pd.DataFrame(df_aug)
    except NameError:
        print("Error: 'generate_synthetic_data_best' is not defined. Using dummy data instead.")
        df_orig = pd.DataFrame(np.random.randn(1000, 5), columns=[f"feat_{i}" for i in range(5)])
        df_aug = pd.DataFrame(np.random.randn(1000, 5), columns=[f"feat_{i}" for i in range(5)])

    # Ensure column consistency
    df_aug.columns = df_orig.columns

    # Run the assessment
    assess_data_similarity(df_orig, df_aug)
