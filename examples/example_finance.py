"""
Example: Finance Domain - Credit Default Prediction
Demonstrates the ML pipeline with financial data
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import sys
sys.path.append('..')

from src.orchestrator import Orchestrator
import yaml


def create_credit_dataset(n_samples=1000):
    """Create a synthetic credit default dataset"""
    
    # Generate synthetic features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_classes=2,
        weights=[0.7, 0.3],  # Imbalanced classes
        random_state=42
    )
    
    # Create meaningful feature names
    feature_names = [
        'credit_score',
        'income',
        'debt_to_income_ratio',
        'employment_length',
        'loan_amount',
        'interest_rate',
        'loan_term',
        'num_credit_lines',
        'num_delinquencies',
        'revolving_balance',
        'revolving_utilization',
        'total_accounts',
        'age',
        'home_ownership',
        'verification_status',
    ]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['default'] = y
    
    # Make values more realistic
    df['credit_score'] = (df['credit_score'] * 100 + 650).clip(300, 850)
    df['income'] = np.exp(df['income'] * 0.5 + 10)  # Log-normal distribution
    df['loan_amount'] = np.exp(df['loan_amount'] * 0.3 + 9) * 1000
    
    print(f"Dataset shape: {df.shape}")
    print(f"Default rate: {df['default'].mean():.2%}")
    print(f"\nFeature summary:")
    print(df.describe())
    
    return df


def main():
    """Run finance example"""
    print("="*80)
    print("Finance Example: Credit Default Prediction")
    print("="*80)
    
    # Create dataset
    print("\n1. Creating credit default dataset...")
    df = create_credit_dataset(n_samples=1500)
    
    # Load configuration
    print("\n2. Loading configuration...")
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize orchestrator
    print("\n3. Initializing orchestrator...")
    orchestrator = Orchestrator(config)
    
    # Run pipeline
    print("\n4. Running ML pipeline...")
    results = orchestrator.run_pipeline(
        data=df,
        target_column='default',
        task_type='classification',
        domain='finance'
    )
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    final_results = results['final_results']
    print(f"\nModel Approved: {final_results['model_approved']}")
    print(f"Deployment Ready: {final_results['deployment_ready']}")
    print(f"Best Model: {final_results['best_model']}")
    print(f"Performance Score: {final_results['performance_score']:.4f}")
    
    print("\nKey Metrics:")
    metrics = final_results['metrics']
    for key in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    
    print("\nRecommendations:")
    for rec in results['recommendations']:
        print(f"  - {rec}")
    
    # Save model
    print("\n5. Saving model...")
    orchestrator.save_final_model("../models/credit_default_model.pkl")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

