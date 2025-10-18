"""
Sample Data Generator
Creates sample datasets for testing the pipeline
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression


def generate_classification_dataset(
    n_samples=1000,
    n_features=20,
    output_path='sample_classification.csv'
):
    """Generate sample classification dataset"""
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=3,
        n_classes=2,
        weights=[0.6, 0.4],
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some categorical features
    df['category_A'] = np.random.choice(['A1', 'A2', 'A3'], size=n_samples)
    df['category_B'] = np.random.choice(['B1', 'B2'], size=n_samples)
    
    # Add some missing values (5%)
    mask = np.random.random(df.shape) < 0.05
    df = df.mask(mask)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Classification dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    
    return df


def generate_regression_dataset(
    n_samples=1000,
    n_features=15,
    output_path='sample_regression.csv'
):
    """Generate sample regression dataset"""
    
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=12,
        noise=10.0,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some categorical features
    df['category_A'] = np.random.choice(['A1', 'A2', 'A3'], size=n_samples)
    
    # Add some missing values (3%)
    mask = np.random.random(df.shape) < 0.03
    df = df.mask(mask)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Regression dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Target statistics:\n{df['target'].describe()}")
    
    return df


if __name__ == "__main__":
    print("Generating sample datasets...\n")
    
    # Generate classification dataset
    print("1. Classification Dataset:")
    generate_classification_dataset()
    
    print("\n" + "="*80 + "\n")
    
    # Generate regression dataset
    print("2. Regression Dataset:")
    generate_regression_dataset()
    
    print("\n" + "="*80)
    print("Sample datasets generated successfully!")

