"""
Test script to verify SVM and Random Forest work when explicitly selected
"""
import pandas as pd
import yaml
import sys
sys.path.append('.')

from src.orchestrator import Orchestrator

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Modify config to use SVM and Random Forest
config['agents']['model_tuning']['algorithms'] = ['svm', 'random_forest']
config['llm']['reasoning_enabled'] = True  # LLM enabled

print("="*70)
print("Testing with user-selected algorithms: SVM and Random Forest")
print("="*70)

# Load data
df = pd.read_csv('data/heart.csv')
print(f"\nData loaded: {df.shape}")

# Initialize orchestrator
orchestrator = Orchestrator(config)

# Run pipeline
results = orchestrator.run_pipeline(
    data=df,
    target_column='target',
    task_type='classification',
    domain='healthcare'
)

# Display results
print("\n" + "="*70)
print("RESULTS")
print("="*70)
final_results = results.get("final_results", {})
print(f"Best Model: {final_results.get('best_model')}")
print(f"Performance Score: {final_results.get('performance_score'):.4f}")
print(f"Model Approved: {final_results.get('model_approved')}")

# Check which models were actually trained
all_models = results.get("iterations", [{}])[0].get("model_tuning", {}).get("all_models", [])
print(f"\nModels trained: {[m['algorithm'] for m in all_models]}")
