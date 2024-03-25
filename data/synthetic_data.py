import numpy as np


def generate_regression_dataset(
    num_features,
    num_samples,
    noise_std,
    std=10
):
    random_coefs = np.round(np.random.randn(num_features), 2)
    features = np.random.randn(num_samples, num_features)*std
    target = features @ random_coefs + np.random.randn(num_samples)*noise_std
    
    return random_coefs, features, target


def generate_binary_classification_dataset(
    num_features,
    num_samples,
    noise_std,
    std=10
):
    random_coefs = np.round(np.random.randn(num_features), 2)
    features = np.random.randn(num_samples, num_features)*std
    target = features @ random_coefs + np.random.randn(num_samples)*noise_std
    
    return random_coefs, features, (target>=0).astype(int)