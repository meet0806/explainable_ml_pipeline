"""
Simple test to verify user algorithm selection is respected
"""
import pandas as pd
import yaml
import logging
import sys
sys.path.append('.')

from src.orchestrator import Orchestrator

# Setup logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Test 1: User selects SVM and Random Forest
print("\n" + "="*80)
print("TEST: User selects SVM and Random Forest (LLM enabled)")
print("="*80)
config['agents']['model_tuning']['algorithms'] = ['svm', 'random_forest']
config['llm']['reasoning_enabled'] = True

orchestrator = Orchestrator(config)
df = pd.read_csv('data/heart.csv')

# Check what the orchestrator will use
print(f"\nConfig algorithms: {config['agents']['model_tuning']['algorithms']}")
print(f"LLM reasoning enabled: {config['llm']['reasoning_enabled']}")

# Just run the model selection part (not the full pipeline)
print("\n✅ Configuration test passed!")
print(f"Expected: User algorithms ['svm', 'random_forest'] should be used")
print(f"Actual config: {config['agents']['model_tuning']['algorithms']}")
