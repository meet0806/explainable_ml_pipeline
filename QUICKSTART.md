# Quick Start Guide

Get up and running with the Explainable ML Pipeline in 5 minutes!

## 1. Installation (2 minutes)

```bash
# Clone repository
git clone <your-repo-url>
cd explainable-ml-pipeline

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data models logs results
```

## 2. Generate Sample Data (30 seconds)

```bash
cd data
python sample_data_generator.py
cd ..
```

This creates:

- `data/sample_classification.csv` - Classification dataset
- `data/sample_regression.csv` - Regression dataset

## 3. Run Your First Pipeline (2 minutes)

### Option A: Command Line

```bash
python main.py \
  --data data/sample_classification.csv \
  --target target \
  --task classification \
  --domain general \
  --save-model
```

### Option B: Python Script

Create `test_pipeline.py`:

```python
import pandas as pd
import yaml
from src.orchestrator import Orchestrator

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load data
df = pd.read_csv('data/sample_classification.csv')

# Run pipeline
orchestrator = Orchestrator(config)
results = orchestrator.run_pipeline(
    data=df,
    target_column='target',
    task_type='classification',
    domain='general'
)

print(f"\nModel Approved: {results['final_results']['model_approved']}")
print(f"Best Model: {results['final_results']['best_model']}")
print(f"Performance: {results['final_results']['performance_score']:.4f}")
```

Run it:

```bash
python test_pipeline.py
```

## 4. View Results (30 seconds)

Check the output:

```bash
# View final report
cat results/final_report_*.json

# View logs
tail -f logs/pipeline.log

# List saved models
ls -lh models/
```

## 5. Try Domain-Specific Examples (1 minute)

### Healthcare Example

```bash
cd examples
python example_healthcare.py
```

### Finance Example

```bash
cd examples
python example_finance.py
```

## What's Happening?

The pipeline automatically:

1. ✅ **EDA Agent** - Analyzes your data quality
2. ✅ **Feature Engineering Agent** - Processes and selects features
3. ✅ **Model Tuning Agent** - Trains and optimizes multiple models
4. ✅ **Evaluator Agent** - Calculates performance metrics
5. ✅ **Judge Agent** - Decides if model is deployment-ready

If performance is insufficient, it automatically retrains with different parameters!

## Next Steps

- **Customize Configuration**: Edit `config.yaml`
- **Use Your Own Data**: Replace sample data with your dataset
- **Add LLM Reasoning**: Enable LLM integration in config
- **Explore Agents**: Check `src/agents/` for agent implementations
- **Read Full Docs**: See `README.md` for detailed information

## Need Help?

- Check `README.md` for detailed documentation
- Review example scripts in `examples/`
- Open an issue on GitHub

---

**You're ready to build explainable ML models! 🚀**
