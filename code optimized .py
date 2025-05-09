#!pip install ucimlrepo
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.interpolate import interp1d
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Union, List, Dict, Optional


class EmpiricalCopulaGenerator:
    """A class to generate synthetic data using an empirical copula approach for mixed data types."""

    def __init__(
        self,
        noise_scale_continuous: float = 0.01,
        noise_scale_integer: float = 0.01,
        handle_missing: str = "impute",
        reduce_dims: Optional[int] = None,
    ):
        """
        Initialize the generator.

        Args:
            noise_scale_continuous (float): Scale of noise added to continuous data (default: 0.01).
            noise_scale_integer (float): Scale of noise added to integer data (default: 0.01).
            handle_missing (str): Strategy for handling missing values. Options: "impute" (default), "drop".
            reduce_dims (Optional[int]): Number of dimensions to reduce to using PCA (default: None).
        """
        self.noise_scale_continuous = noise_scale_continuous
        self.noise_scale_integer = noise_scale_integer
        self.handle_missing = handle_missing
        self.reduce_dims = reduce_dims
        self.column_types = None
        self.transform_params = {}
        self.pca = None

    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Handle missing values in the data."""
        if self.handle_missing == "impute":
            for i in range(data.shape[1]):
                column = data[:, i]
                if np.issubdtype(column.dtype, np.number):
                    # Impute numerical columns with mean
                    column[np.isnan(column)] = np.nanmean(column)
                else:
                    # Impute categorical columns with mode
                    # Convert column to string type to handle mixed types
                    column = column.astype(str)
                    unique, counts = np.unique(column[column != 'nan'], return_counts=True)  # Exclude 'nan' strings
                    mode = unique[np.argmax(counts)]
                    column[column == 'nan'] = mode  # Replace 'nan' strings with mode
                    data[:, i] = column  # Update the original data
        elif self.handle_missing == "drop":
            data = data[~pd.DataFrame(data).isnull().any(axis=1).to_numpy()]
            #data = data[~np.isnan(data).any(axis=1)]  # error with strings
        else:
            raise ValueError(f"Invalid handle_missing strategy: {self.handle_missing}")
        return data

    def _detect_column_types(self, data: np.ndarray) -> List[str]:
        """Detect whether each column is continuous, integer, or categorical."""
        column_types = []
        for i in range(data.shape[1]):
            column = data[:, i]
            try:
                column_float = column.astype(float)
                is_integer = np.all(np.isclose(column_float, column_float.astype(int)))
                column_types.append("integer" if is_integer else "continuous")
            except (ValueError, TypeError):
                column_types.append("categorical")
        return column_types

    def _transform_to_uniform(self, column: np.ndarray, col_type: str) -> np.ndarray:
        """Transform a single column to uniform margins based on its type."""
        if col_type == "continuous":
            return (rankdata(column, method="average") - 1) / (len(column) - 1)
        elif col_type == "integer":
            # Convert column to numeric if it contains strings
            try:
                column = column.astype(float)
            except ValueError:
                # Handle the case where the column contains non-numeric strings
                # You might want to impute or drop these values, or adjust the type detection logic
                pass  # For now, skip adding noise to this column
            jittered = column + np.random.uniform(-self.noise_scale_integer, self.noise_scale_integer, size=column.shape)
            return (rankdata(jittered, method="average") - 1) / (len(jittered) - 1)
        else:  # categorical
            unique, counts = np.unique(column, return_counts=True)
            cum_probs = np.cumsum(counts) / len(column)
            return np.array([cum_probs[np.where(unique == x)[0][0]] for x in column])

    def _inverse_transform(self, u_column: np.ndarray, original_column: np.ndarray, col_type: str) -> np.ndarray:
        """Transform uniform margins back to the original space."""
        if col_type == "continuous":
            sorted_data = np.sort(original_column)
            inv_cdf = interp1d(np.linspace(0, 1, len(sorted_data)), sorted_data, bounds_error=False, fill_value="extrapolate")
            transformed = inv_cdf(u_column)
            return self._perturb_continuous(original_column, u_column, transformed)
        elif col_type == "integer":
            return np.quantile(original_column, u_column, method='nearest').astype(int)
        else:  # categorical
            unique, counts = np.unique(original_column, return_counts=True)
            cum_probs = np.cumsum(counts) / len(original_column)
            return np.array([unique[np.searchsorted(cum_probs, u)] for u in u_column])

    def _perturb_continuous(self, original: np.ndarray, u_data: np.ndarray, transformed: np.ndarray) -> np.ndarray:
        """Add controlled noise to continuous data while preserving order."""
        perturbed_u = u_data.copy()
        sorted_orig = np.sort(original)
        for i in range(len(u_data)):
            v = perturbed_u[i]
            idx = np.searchsorted(sorted_orig, v)
            v1 = sorted_orig[max(0, idx - 1)]
            v2 = sorted_orig[min(len(sorted_orig) - 1, idx)]
            max_noise = min(abs(v2 - v), abs(v - v1)) if v1 != v2 else 0.01
            noise = np.random.uniform(-max_noise, max_noise) * self.noise_scale_continuous
            perturbed_u[i] = np.clip(v + noise, 0, 1)
        inv_cdf = interp1d(np.linspace(0, 1, len(sorted_orig)), sorted_orig, bounds_error=False, fill_value="extrapolate")
        return inv_cdf(perturbed_u)

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit the copula generator to the data by transforming it to uniform margins.

        Args:
            data (np.ndarray): Input data with mixed types (shape: [n_samples, n_features]).

        Returns:
            np.ndarray: Transformed data in uniform space.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Data must be a 2D NumPy array.")

        # Handle missing values
        data = self._handle_missing_values(data)

        # Reduce dimensionality if specified
        if self.reduce_dims is not None:
            self.pca = PCA(n_components=self.reduce_dims)
            data = self.pca.fit_transform(data)

        self.column_types = self._detect_column_types(data)
        u_data = np.zeros_like(data, dtype=float)

        # Precompute transformations for all columns
        for i, col_type in enumerate(self.column_types):
            u_data[:, i] = self._transform_to_uniform(data[:, i], col_type)
            self.transform_params[i] = {"type": col_type, "original": data[:, i].copy()}

        return u_data

    def sample(self, u_data: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Generate new samples by resampling from the empirical copula.

        Args:
            u_data (np.ndarray): Uniform data from `fit`.
            n_samples (int): Number of samples to generate.

        Returns:
            np.ndarray: Sampled uniform data.
        """
        indices = np.random.choice(u_data.shape[0], size=n_samples, replace=True)
        return u_data[indices]

    def transform_back(self, sampled_u_data: np.ndarray) -> np.ndarray:
        """
        Transform sampled uniform data back to the original space.

        Args:
            sampled_u_data (np.ndarray): Uniform samples from `sample`.

        Returns:
            np.ndarray: Synthetic data in the original space.
        """
        transformed_data = np.zeros_like(sampled_u_data, dtype=object)
        for i, col_info in self.transform_params.items():
            transformed_data[:, i] = self._inverse_transform(
                sampled_u_data[:, i], col_info["original"], col_info["type"]
            )

        # Inverse transform PCA if dimensionality was reduced
        if self.pca is not None:
            transformed_data = self.pca.inverse_transform(transformed_data)

        return transformed_data

    def generate(self, data: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Generate synthetic data mimicking the input data's distribution.

        Args:
            data (np.ndarray): Input data with mixed types.
            n_samples (int): Number of synthetic samples to generate.

        Returns:
            np.ndarray: Synthetic data.
        """
        u_data = self.fit(data)
        sampled_u_data = self.sample(u_data, n_samples)
        return self.transform_back(sampled_u_data)


# Example usage
if __name__ == "__main__":
    # Fetch dataset
    census_income = fetch_ucirepo(id=20)
    data = census_income.data.features.iloc[:1000, :14].to_numpy()

    # Initialize generator
    generator = EmpiricalCopulaGenerator(
        noise_scale_continuous=0.01,
        noise_scale_integer=0.01,
        handle_missing="impute",
        reduce_dims=None,  # Set to an integer (e.g., 10) to reduce dimensions
    )

    # Generate synthetic data
    n_samples_generated = 1900
    generated_data = generator.generate(data, n_samples_generated)

    # Optional: Convert to DataFrame for inspection
    generated_df = pd.DataFrame(generated_data, columns=census_income.data.features.columns[:14])
    print(generated_df.head())
