# Install necessary package
!pip install -q scipy imbalanced-learn

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

# === Augmentation functions from your code ===

noise_scale = 0.01

def jitter(data, scale=0.01):
    return data + np.random.uniform(-scale, scale, size=data.shape)

def empirical_copula_transform_C(data):
    u_data = (rankdata(data, method="average") - 1) / (len(data) - 1)
    return u_data

def empirical_copula_transform_I(data):
    u_data = (rankdata(data, method="average") - 1) / (len(data) - 1)
    return u_data

def empirical_copula_transform_Cat(data):
    unique, counts = np.unique(data, return_counts=True)
    cum_probs = np.cumsum(counts) / len(data)
    mapping = dict(zip(unique, cum_probs))
    u_data = np.vectorize(mapping.get)(data)
    return u_data

def inverse_marginals_C(data, u_data):
    sorted_data = np.sort(data)
    cdf = np.linspace(0, 1, len(sorted_data))
    inv_cdf = interp1d(cdf, sorted_data, bounds_error=False, fill_value="extrapolate")
    transformed_data = inv_cdf(u_data)
    transformed_data = np.nan_to_num(transformed_data, nan=np.nanmin(data))
    return transformed_data

def inverse_marginals_Cat(data, u_data):
    unique, counts = np.unique(data, return_counts=True)
    cum_probs = np.cumsum(counts) / len(data)
    # Find closest cum_prob for each u_data value
    idxs = np.searchsorted(cum_probs, u_data, side="right")
    idxs = np.clip(idxs, 0, len(unique)-1)
    transformed_data = unique[idxs]
    return transformed_data

def inverse_empirical_copula_transform_I(copula_samples, original_data):
    new_data = np.quantile(original_data, copula_samples)
    return new_data

def perturb_generated_data(original_data, sampled_u_data, noise_scale):
    perturbed_u_data = np.copy(sampled_u_data)
    sorted_u_data = np.sort(original_data)
    for i in range(sampled_u_data.shape[0]):
        v = perturbed_u_data[i]
        idx = np.searchsorted(sorted_u_data, v)
        v1 = sorted_u_data[max(0, idx - 1)]
        v2 = sorted_u_data[min(len(sorted_data) - 1, idx)] # Corrected from sorted_u_data to sorted_data
        if v1 == v2:
            if idx > 0:
                v1 = sorted_data[idx - 1] # Corrected from sorted_u_data to sorted_data
            if idx < len(sorted_data) - 1:
                v2 = sorted_data[idx + 1] # Corrected from sorted_u_data to sorted_data
        max_noise = min(abs(v2 - v), abs(v - v1))
        noise = np.random.uniform(-max_noise, max_noise)
        noise = noise_scale * noise
        perturbed_u_data[i] = np.clip(v + noise, 0, 1)
    perturbed_data = inverse_marginals_C(original_data, perturbed_u_data)
    return perturbed_data


def empirical_copula_transform_mixed(data):
    u_data = np.zeros_like(data, dtype=float)
    for i in range(data.shape[1]):
        column = data[:, i]
        try:
            column_float = column.astype(float)
            if np.all(np.isclose(column_float, column_float.astype(int))):
                D_jittered = jitter(column_float.astype(int))
                u_data[:, i] = empirical_copula_transform_I(D_jittered)
            else:
                u_data[:, i] = empirical_copula_transform_C(column_float)
        except ValueError:
            u_data[:, i] = empirical_copula_transform_Cat(column)
    return u_data

def inverse_marginals_mixed(data, u_data):
    transformed_data = np.zeros_like(u_data, dtype=object)
    for i in range(data.shape[1]):
        column = data[:, i]
        try:
            column_float = column.astype(float)
            if np.all(np.isclose(column_float, column_float.astype(int))):
                transformed_data[:, i] = inverse_empirical_copula_transform_I(u_data[:, i], column.astype(int))
                transformed_data[:, i] = np.round(transformed_data[:, i].astype(float)).astype(int)
            else:
                transformed_data[:, i] = perturb_generated_data(column_float, u_data[:, i], noise_scale)
        except ValueError:
            transformed_data[:, i] = inverse_marginals_Cat(column, u_data[:, i])
    return transformed_data

def sample_empirical_copula(u_data, n_samples):
    n = u_data.shape[0]
    indices = np.random.choice(n, size=n_samples, replace=True)
    return u_data[indices]

def generate_new_data_mixed(data, n_samples):
    u_data = empirical_copula_transform_mixed(data)
    sampled_u_data = sample_empirical_copula(u_data, n_samples)
    new_data = inverse_marginals_mixed(data, sampled_u_data)
    return new_data

# === Load and preprocess Adult dataset ===
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation',
           'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']

df = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)
df.dropna(inplace=True)
df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Keep original mixed-type features as numpy array (no encoding yet)
X = df.drop('income', axis=1).values
y = df['income'].values

# Create small train/test split simulating few data
X_small, X_unused, y_small, y_unused = train_test_split(X, y, train_size=0.02, stratify=y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_small, y_small, test_size=0.3, stratify=y_small, random_state=42)

print("Original training data shape:", X_train.shape)
print("Original class counts:", np.bincount(y_train))

# Generate 5000 synthetic samples
X_synth = generate_new_data_mixed(X_train, n_samples=5000)

# Combine original and synthetic data
X_train_combined = np.vstack([X_train, X_synth])
y_train_combined = np.hstack([y_train, np.tile(y_train, int(np.ceil(len(X_synth) / len(y_train))))[:len(X_synth)]]) # Corrected label duplication

print("Combined training data shape:", X_train_combined.shape)
print("Combined class counts:", np.bincount(y_train_combined))


# === Label encode categorical columns for RF ===
cat_cols = [i for i, dtype in enumerate(df.dtypes[:-1]) if dtype == 'object']

def label_encode_np(data, cat_cols):
    data_enc = data.copy()
    for col in cat_cols:
        col_vals = data[:, col].astype(str)
        uniques, encoded = np.unique(col_vals, return_inverse=True)
        data_enc[:, col] = encoded
    return data_enc.astype(float)


X_train_combined_enc = label_encode_np(X_train_combined, cat_cols)
X_test_enc = label_encode_np(X_test, cat_cols)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_combined_enc)
X_test_scaled = scaler.transform(X_test_enc)

# Baseline RF on original small training data only
X_train_enc = label_encode_np(X_train, cat_cols)
X_train_scaled_base = scaler.fit_transform(X_train_enc)

rf_base = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
rf_base.fit(X_train_scaled_base, y_train)
y_pred_base = rf_base.predict(X_test_scaled)

print("\n--- Baseline RF (no augmentation) ---")
print(classification_report(y_test, y_pred_base))
print("Accuracy:", accuracy_score(y_test, y_pred_base))
print("F1 Score:", f1_score(y_test, y_pred_base))

# RF trained on combined original + synthetic data
rf_aug = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
rf_aug.fit(X_train_scaled, y_train_combined)
y_pred_aug = rf_aug.predict(X_test_scaled)

print("\n--- RF with your augmentation ---")
print(classification_report(y_test, y_pred_aug))
print("Accuracy:", accuracy_score(y_test, y_pred_aug))
print("F1 Score:", f1_score(y_test, y_pred_aug))