
from scipy.stats import energy_distance
# Restart the kernel for changes to take effect.
# Once the kernel restarts, re-run the cell that imports 'sdmetrics'.
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import rankdata, ks_2samp, chi2_contingency

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.spatial.distance import cdist
import numpy as np



# Rest of your code (including _combine_pvalues_fisher, plot_marginal_comparisons, etc.) remains the same.


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





  n_samples_generated=50

  noise_scale=0.01

    # Generate new data using the empirical copula
  generated_data = generate_new_data_mixed(data, n_samples_generated)


plt.figure(figsize=(8, 6))

plt.scatter(data[:, 0], data[:, 1], s=20, marker='*', color='darkblue', alpha=0.5, label='Original data')
plt.scatter(generated_data[:, 0], generated_data[:, 1], s=10, marker='o', color='red', alpha=0.6, label='Aug. data')

plt.title("original data vs. Augm. data")
plt.xlabel("Axe X")
plt.ylabel("Axe Y")
plt.legend()
plt.grid(True)
plt.show()