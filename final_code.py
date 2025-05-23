# Install the required package within the notebook's environment.
from ucimlrepo import fetch_ucirepo
from scipy.stats import energy_distance
# Restart the kernel for changes to take effect.
# Once the kernel restarts, re-run the cell that imports 'sdmetrics'.
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import rankdata, ks_2samp, chi2_contingency

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.spatial.distance import cdist
import numpy as np

def multivariate_energy_distance(D, D2):
    """
    Calcule la distance d'énergie entre deux ensembles de données multivariés avec types mixtes.
    Les variables catégorielles sont encodées en one-hot et les variables numériques sont standardisées.

    Paramètres :
    - D  : np.array de taille (n1, m) (Données originales)
    - D2 : np.array de taille (n2, m) (Données générées)

    Retourne :
    - Distance d'énergie entre D et D2
    """
    
    # Get data types for each column in D
    D_types = [type(x) for x in D[0]]

    # Identify numerical and categorical columns based on data types
    num_cols_D = [i for i, x in enumerate(D_types) if np.issubdtype(x, np.floating)]  
    cat_cols_D = [i for i, x in enumerate(D_types) if not np.issubdtype(x, np.floating)]

    # Check if there are any categorical columns, if not, return early
    if not cat_cols_D:  # If cat_cols_D is empty
        # If there are no categorical features, you can still calculate the distance, 
        # but ensure that num_cols_D has features.
        if not num_cols_D:
          raise ValueError("Data must have at least one numerical or categorical feature for energy distance calculation.")
          
        num_columns_D = D[:, num_cols_D]
        num_columns_D2 = D2[:, num_cols_D]
        
        # Standardize numerical columns using StandardScaler
        scaler = StandardScaler()
        num_columns_D_standardized = scaler.fit_transform(num_columns_D)
        num_columns_D2_standardized = scaler.transform(num_columns_D2)

        # In this case, calculate the distance based on numerical data only
        dist_XX = cdist(num_columns_D_standardized, num_columns_D_standardized, metric='euclidean')
        dist_YY = cdist(num_columns_D2_standardized, num_columns_D2_standardized, metric='euclidean')
        dist_XY = cdist(num_columns_D_standardized, num_columns_D2_standardized, metric='euclidean')

        energy_dist = (2 * np.mean(dist_XY) - np.mean(dist_XX) - np.mean(dist_YY))
        return energy_dist

    # Extract numerical and categorical data (if cat_cols_D is not empty)
    num_columns_D = D[:, num_cols_D]
    cat_columns_D = D[:, cat_cols_D]

    num_columns_D2 = D2[:, num_cols_D]
    cat_columns_D2 = D2[:, cat_cols_D]

    # Standardize numerical columns using StandardScaler
    scaler = StandardScaler()
    num_columns_D_standardized = scaler.fit_transform(num_columns_D)
    num_columns_D2_standardized = scaler.transform(num_columns_D2)

    # Encode categorical columns using OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_columns_D_encoded = encoder.fit_transform(cat_columns_D.astype(str))
    cat_columns_D2_encoded = encoder.transform(cat_columns_D2.astype(str))

    # Combine standardized numerical data and encoded categorical data
    D_combined = np.hstack([num_columns_D_standardized, cat_columns_D_encoded])
    D2_combined = np.hstack([num_columns_D2_standardized, cat_columns_D2_encoded])

    D_combined = D_combined.astype(float)
    D2_combined = D2_combined.astype(float)

    # Calculate the energy distance (Euclidean distance based)
    dist_XX = cdist(D_combined, D_combined, metric='euclidean')
    dist_YY = cdist(D2_combined, D2_combined, metric='euclidean')
    dist_XY = cdist(D_combined, D2_combined, metric='euclidean')

    energy_dist = (2 * np.mean(dist_XY) - np.mean(dist_XX) - np.mean(dist_YY))
    return energy_dist


def permutation_test(D, D2, compute_stat, n_permutations=1000):
    """
    Test de permutation pour comparer les distributions de D et D2.

    Paramètres :
    - D : np.array (n1, m) - Données originales
    - D2 : np.array (n2, m) - Données générées
    - compute_stat : Fonction pour calculer la statistique (ici, distance d'énergie)
    - n_permutations : Nombre d'échantillons aléatoires générés

    Retourne :
    - Statistique observée
    - p-value du test
    """
    combined = np.vstack([D, D2])  # Concaténation des deux jeux de données
    observed_stat = compute_stat(D, D2)
    permuted_stats = []

    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_D = combined[:len(D)]
        perm_D2 = combined[len(D):]
        permuted_stats.append(compute_stat(perm_D, perm_D2))

    p_value = np.mean(np.array(permuted_stats) >= observed_stat)
    return observed_stat, p_value

def compare_distributions(original_data, generated_data):
    """
    Compares distributions of original and generated data using appropriate tests
    for continuous, integer, and categorical features.

    Parameters:
    - original_data: Original data (numpy array of shape (n_samples, n_features)).
    - generated_data: Generated data (numpy array of shape (n_samples, n_features)).

    Returns:
    - A dictionary containing:
        - 'pvalues': A list of p-values for each feature.
        - 'overall_pvalue': The overall p-value (combined using Fisher's method).
    """

    num_features = original_data.shape[1]
    pvalues = []

    # Compare marginal distributions for each feature
    for i in range(num_features):
        original_feature = original_data[:, i]
        generated_feature = generated_data[:, i]

        try:
          original_feature2 = original_feature.astype(float)

            # If convertible to float, check if it's actually integer
          if np.all(np.isclose(original_feature2, original_feature2.astype(int))):
            stat, p_value = ks_2samp(original_feature2, generated_feature.astype(float))
            print('int')
          else:
            print('cont')
            stat, p_value = ks_2samp(original_feature2, generated_feature.astype(float))

        except ValueError:
          print('cat')
          unique_values_original = np.unique(original_feature).astype(str)  # Convert to strings
          unique_values_generated = np.unique(generated_feature).astype(str)  # Convert to strings
          unique_values = np.union1d(unique_values_original, unique_values_generated)  # Combine unique values
          contingency_table = np.zeros((len(unique_values), 2), dtype=int)
          for j, value in enumerate(unique_values):
            contingency_table[j, 0] = np.sum(original_feature == value)
            contingency_table[j, 1] = np.sum(generated_feature == value)

          if np.any(np.sum(contingency_table, axis=0) == 0) or np.any(np.sum(contingency_table, axis=1) == 0):
            p_value = np.nan
          else:
            stat, p_value, dof, expected = chi2_contingency(contingency_table)

        pvalues.append(p_value)

    # Combine p-values using Fisher's method (for overall p-value)
    overall_pvalue = _combine_pvalues_fisher(pvalues)

    return {
        'pvalues': pvalues,
        'overall_pvalue': overall_pvalue
    }

# Rest of your code (including _combine_pvalues_fisher, plot_marginal_comparisons, etc.) remains the same.

def _combine_pvalues_fisher(pvalues):
    """Combines p-values using Fisher's method."""
    from scipy.stats import chi2
    # Handle NaNs (e.g., from non-applicable tests)
    valid_pvalues = [p for p in pvalues if not np.isnan(p)]
    if not valid_pvalues:
        return np.nan

    statistic = -2 * np.sum(np.log(valid_pvalues))
    degrees_of_freedom = 2 * len(valid_pvalues)
    overall_pvalue = chi2.sf(statistic, degrees_of_freedom)
    return overall_pvalue



def plot_marginal_comparisons(original_data, generated_data):
    """
    Plot histograms for marginal comparisons between original and generated data.
    """
    fig, axes = plt.subplots(1, original_data.shape[1], figsize=(15, 5))

    for i in range(original_data.shape[1]):
        axes[i].hist(original_data[:, i], bins=30, alpha=0.5, label='Original', density=True)
        axes[i].hist(generated_data[:, i], bins=30, alpha=0.5, label='Generated', density=True)
        axes[i].set_title(f'Feature {i+1}')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def generate_mixed_data(n_samples):
  """Generates a dataset with mixed data types and specified distributions."""

  # Generate 100 floating-point numbers (uniform distribution)
  float_data = np.random.normal(loc=0, scale=50, size=n_samples)
  # Generate 100 integers (uniform distribution between 1 and 10, for example)
  int_data = np.random.randint(1, 11, size=n_samples)

  # Generate 100 categorical values with specified probabilities
  categories = ['A', 'B', 'C', 'D']
  probabilities = [0.20, 0.05, 0.50, 0.25]  # Probabilities for A, B, C, D
  cat_data = np.random.choice(categories, size=n_samples, p=probabilities)

  # Combine the data into a single array
  data = np.column_stack([float_data, int_data, cat_data])

  # Convert to object dtype to handle mixed types
  data = data.astype(object)

  return data




def empirical_copula_transform_mixed(data):
    """
    Transform data to uniform margins using empirical CDF for continuous
    variables and frequency-based probabilities for categorical/discrete variables.
    Includes column type detection within the function.
    """

    u_data = np.zeros_like(data, dtype=float)

    for i in range(data.shape[1]):
        column = data[:, i]

        # Check if the column can be converted to float
        try:
            column_float = column.astype(float)

            # If convertible to float, check if it's actually integer
            if np.all(np.isclose(column_float, column_float.astype(int))):

              D_jittered = jitter(column_float.astype(int))
              u_data[:, i] = empirical_copula_transform_I(D_jittered)
            else:
              u_data[:, i] = empirical_copula_transform_C(column)

        except ValueError:

          u_data[:, i] = empirical_copula_transform_Cat(column)

    #copula_D=u_data
    return u_data






def inverse_marginals_mixed(data, u_data):
    """
    Transform uniform marginals back to original marginals using inverse CDF
    for continuous variables and reverse mapping for categorical/discrete variables.
    """
    transformed_data = np.zeros_like(u_data, dtype=object)
    for i in range(data.shape[1]):
        column = data[:, i]
        try:
          column_float = column.astype(float)
          if np.all(np.isclose(column_float, column_float.astype(int))):
            transformed_data[:, i]=inverse_empirical_copula_transform_I(u_data[:, i] , column.astype(int))
            transformed_data[:, i] = np.round(transformed_data[:, i].astype(float)).astype(int)
          else:
            transformed_data[:, i] = perturb_generated_data( column_float, u_data[:, i], noise_scale)

        except ValueError:
            transformed_data[:, i] = inverse_marginals_Cat(column, u_data[:, i])


    return transformed_data
def empirical_copula_transform_C(data):
    """
    Transform data to uniform margins using empirical CDF.
    """
    u_data = np.zeros_like(data, dtype=float)
    u_data = (rankdata(data, method="average") - 1) / (len(data) - 1)  # Empirical CDF
    return u_data
def empirical_copula_transform_I(data):
    """
    Transform data to uniform margins using empirical CDF.
    """
    u_data = np.zeros_like(data, dtype=float)
    u_data = (rankdata(data, method="average") - 1) / (len(data) - 1)  # Empirical CDF
    return u_data
def perturb_generated_data(original_data, sampled_u_data, noise_scale):
    """
    Perturb the generated data (sampled_u_data) to introduce more noise while maintaining the marginals and joint distribution.
    The noise is added in the uniform space, ensuring that the perturbed values respect the order of the original data.

    Assumes original_data and sampled_u_data are one-dimensional arrays.
    """

    perturbed_u_data = np.copy(sampled_u_data)

    # Sort original data (1D array)
    sorted_u_data = np.sort(original_data)

    for i in range(sampled_u_data.shape[0]):  # For each generated data point
        v = perturbed_u_data[i]  # Current value in uniform space

        # Find v1 and v2 (values just before and after v in the sorted original data)
        idx = np.searchsorted(sorted_u_data, v)  # Find index of v in sorted_u_data

        # Handle boundary cases (first and last elements)
        v1 = sorted_u_data[max(0, idx - 1)]
        v2 = sorted_u_data[min(len(sorted_u_data) - 1, idx)]

        if v1 == v2:  # Handle case where values are the same to avoid zero noise
            if idx > 0:
                v1 = sorted_u_data[idx - 1]
            if idx < len(sorted_u_data) - 1:
                v2 = sorted_u_data[idx + 1]

        # Calculate noise, ensuring it stays within bounds
        max_noise = min(abs(v2 - v), abs(v - v1))
        noise = np.random.uniform(-max_noise, max_noise)
        noise = noise_scale * noise

        # Add noise and clip to keep values in [0, 1]
        perturbed_u_data[i] = np.clip(v + noise, 0, 1)
    #copula_D2=perturbed_u_data
    #print("dim",copula_D2.shape)
    # Transform the perturbed uniform data back to the original space
    perturbed_data = inverse_marginals_C(original_data, perturbed_u_data)
    return perturbed_data
def inverse_marginals_C(data, u_data):
    """
    Transform uniform marginals back to original marginals using inverse CDF.

    Assumes data and u_data are one-dimensional arrays.
    """
    # Sort the original data (1D array)
    sorted_data = np.sort(data)

    # Create a uniform CDF for the sorted data
    cdf = np.linspace(0, 1, len(sorted_data))

    # Create an inverse CDF function using interpolation
    inv_cdf = interp1d(cdf, sorted_data, bounds_error=False, fill_value="extrapolate")

    # Apply the inverse CDF to the uniform data to get the transformed data
    transformed_data = inv_cdf(u_data)
    transformed_data = np.nan_to_num(transformed_data, nan=np.nanmin(data))  # or np.nanmax(data) for maximum


    return transformed_data
def jitter(data, scale=0.01):
    """
    Add small random noise to integer-valued data to break ties.
    """
    return data + np.random.uniform(-scale, scale, size=data.shape)
def inverse_empirical_copula_transform_I(copula_samples, original_data):
    """
    Transform copula samples back to the original feature space for a single feature.

    Parameters:
    - copula_samples: 1D NumPy array (uniform samples in copula space, values in [0, 1]).
    - original_data: 1D NumPy array (original data before transformation).

    Returns:
    - new_data: 1D NumPy array (values transformed back to the original feature space).
    """
    # Use quantiles of the original data to transform back
    new_data = np.quantile(original_data, copula_samples)

    return new_data

def empirical_copula_transform_Cat(data):
    # Assuming the data has only one categorical feature
    unique, counts = np.unique(data, return_counts=True)  # Step 1
    cum_probs = np.cumsum(counts) / len(data)  # Step 2
    mapping = dict(zip(unique, cum_probs))  # Step 3

    # Apply the mapping to the data and return the transformed values
    u_data = np.vectorize(mapping.get)(data)  # Step 4

    return u_data
def inverse_marginals_Cat(data, u_data):
    # Assuming data and u_data have only one categorical feature
    unique, counts = np.unique(data, return_counts=True)  # Step 1
    cum_probs = np.cumsum(counts) / len(data)  # Step 2
    inv_mapping = dict(zip(cum_probs, unique))  # Step 3

    # Apply the inverse mapping to the copula values (u_data)
    transformed_data = np.vectorize(inv_mapping.get)(u_data)  # Step 4

    return transformed_data
def sample_empirical_copula(u_data, n_samples):
    """
    Generate new samples by resampling from the empirical copula.
    """
    n = u_data.shape[0]
    indices = np.random.choice(n, size=n_samples, replace=True)  # Resample indices
    return u_data[indices]
def generate_new_data_mixed(data, n_samples):
    """
    Generate new data following the same multivariate distribution using the empirical copula.
    """
    # Step 1: Transform the data to uniform margins

    u_data = empirical_copula_transform_mixed(data)


    # Step 2: Sample from the empirical copula
    sampled_u_data = sample_empirical_copula(u_data, n_samples)
    #print("dim",sampled_u_data)
    #copula_D2=sampled_u_data
    # Step 3: Transform the sampled uniform data back to the original space
    new_data = inverse_marginals_mixed(data, sampled_u_data)
   # print("new data",new_data)
    return new_data



# fetch dataset

# Example usage
if __name__ == "__main__":
  data = pd.read_csv('ecoli.csv', header=None, delim_whitespace=True)

# Extract all features except the first and last columns
  data = data.iloc[:, 1:-1].values
  n_samples_generated=1000

  noise_scale=0.01

    # Generate new data using the empirical copula
  generated_data = generate_new_data_mixed(data, n_samples_generated)
  results = compare_distributions(data, generated_data)
  print("\nComparing distributions:")
  print(f"KS Test p-values for marginals: {results['pvalues']}")
  print(f"Overall KS Test p-value: {results['overall_pvalue']}")
    

# Normalize the numerical data
    
    # Plot marginal distributions
  energy_dist = multivariate_energy_distance(data, generated_data)
  print(f"Distance d'Énergie entre D1 et D2 : {energy_dist:.4f}")

    # Test statistique par permutation
  stat, p_val = permutation_test(data, generated_data, multivariate_energy_distance, n_permutations=1000)
  print(f"Test de permutation - Distance d'Énergie : {stat:.4f}, p-value : {p_val:.4f}")

    # Interprétation
  if p_val < 0.05:
    print("On rejette H0 : D1 et D2 ont des distributions significativement différentes.")
  else:
    print("On ne peut pas rejeter H0 : D1 et D2 semblent suivre la même distribution.")


    #plot_marginal_comparisons(data, generated_data)
