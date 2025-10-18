"""
Example: Healthcare Domain - Diabetes Prediction
Demonstrates the ML pipeline with healthcare data
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
import sys
sys.path.append('..')

from src.orchestrator import Orchestrator
import yaml


def create_diabetes_dataset():
    """Create a diabetes classification dataset"""
    # Load sklearn diabetes dataset (regression)
    diabetes = load_diabetes()
    
    # Convert to DataFrame
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    
    # Convert target to binary classification (diabetes yes/no)
    # Using median as threshold
    df['diabetes'] = (diabetes.target > np.median(diabetes.target)).astype(int)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['diabetes'].value_counts()}")
    
    return df


def main():
    """Run healthcare example"""
    print("="*80)
    print("Healthcare Example: Diabetes Prediction")
    print("="*80)
    
    # Create dataset
    print("\n1. Creating diabetes dataset...")
    df = create_diabetes_dataset()
    
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
        target_column='diabetes',
        task_type='classification',
        domain='healthcare'
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
    for key in ['accuracy', 'precision', 'recall', 'f1_score']:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    
    print("\nRecommendations:")
    for rec in results['recommendations']:
        print(f"  - {rec}")
    
    # Save model
    print("\n5. Saving model...")
    orchestrator.save_final_model("../models/diabetes_model.pkl")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

