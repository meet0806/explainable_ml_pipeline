# 📜 Run History Feature

## Overview

The ML Pipeline now includes a comprehensive **Run History** system that tracks all pipeline executions, similar to ChatGPT's conversation history. This allows you to:

- 📊 **Save every pipeline run** automatically with metadata
- 🔍 **View and load previous runs** to revisit results
- 🔄 **Compare multiple runs** side-by-side
- 📥 **Export comparisons** to CSV or JSON
- 🗑️ **Delete old runs** to clean up history

---

## Features

### 1. Automatic Run Tracking

Every time you run the ML pipeline, it automatically saves:

- Dataset name and shape
- Task type (classification/regression)
- Domain (healthcare/finance/general)
- Target column
- Model used
- Performance metrics
- Deployment status
- Timestamp
- Full pipeline results

### 2. Sidebar History Panel

In the Streamlit UI sidebar, you'll find:

```
📜 Run History
├── Total Runs: X
├── Deployed: Y
├── Load Previous Run (dropdown)
│   ├── 📂 Load Run (button)
│   └── 🗑️ Delete (button)
```

**How to use:**

1. Select a run from the dropdown
2. Click **"📂 Load Run"** to view its results
3. Click **"🗑️"** to delete that run

### 3. Compare Runs Tab

A new tab **"🔄 Compare Runs"** allows you to:

- Select 2-5 runs to compare
- View side-by-side metrics table
- See grouped bar chart comparing performance
- Identify the best performer automatically
- Export comparison data

**Metrics compared:**

- **Classification**: Accuracy, F1 Score, Precision, Recall
- **Regression**: R² Score, RMSE, MAE

### 4. Data Persistence

All runs are stored in the `history/` directory:

```
history/
├── run_index.json          # Index of all runs
├── run_20251017_230256.json  # Full results for each run
├── run_20251017_231634.json
└── ...
```

---

## API Reference

### `RunHistory` Class

Located in `src/utils/run_history.py`

#### Methods

**`save_run(results, dataset_name, task_type, domain, target_column, dataset_shape, run_name=None)`**

- Saves a pipeline run with metadata
- Returns: `run_id` (unique identifier)

**`get_all_runs()`**

- Returns list of all run metadata (most recent first)

**`get_run(run_id)`**

- Returns full results for a specific run
- Returns: Dict with complete pipeline results

**`get_run_metadata(run_id)`**

- Returns just the metadata for a run (without full results)

**`delete_run(run_id)`**

- Deletes a run from history
- Returns: `True` if deleted, `False` if not found

**`get_comparison_data(run_ids)`**

- Returns comparison DataFrame for multiple runs
- Input: List of run IDs
- Output: pandas DataFrame with metrics

**`get_stats()`**

- Returns overall statistics:
  - `total_runs`: Total number of runs
  - `successful_deployments`: Runs marked as deployment-ready
  - `most_used_model`: Most frequently used model
  - `best_accuracy`: Best accuracy achieved (classification)

**`export_run(run_id, export_path)`**

- Exports a run to a JSON file
- Returns: `True` if successful

---

## Usage Examples

### Programmatic Usage

```python
from src.utils.run_history import RunHistory

# Initialize history manager
history = RunHistory()

# Get all runs
all_runs = history.get_all_runs()
print(f"Total runs: {len(all_runs)}")

# Load a specific run
run_id = "20251017_230256"
results = history.get_run(run_id)
print(f"Model: {results['final_results']['model_name']}")

# Compare runs
comparison = history.get_comparison_data([
    "20251017_230256",
    "20251017_231634"
])
print(comparison)

# Get stats
stats = history.get_stats()
print(f"Success rate: {stats['successful_deployments'] / stats['total_runs']:.1%}")

# Delete a run
history.delete_run("20251017_230256")
```

### UI Usage

1. **Run Pipeline**: Results automatically saved
2. **Load Previous Run**:
   - Go to sidebar → "📜 Run History"
   - Select run from dropdown
   - Click "📂 Load Run"
3. **Compare Runs**:
   - Go to "🔄 Compare Runs" tab
   - Select 2-5 runs
   - View comparison charts
   - Download CSV/JSON

---

## Data Structure

### Run Metadata (stored in `run_index.json`)

```json
{
  "run_id": "20251017_230256",
  "run_name": "Run 2025-10-17 23:02",
  "timestamp": "2025-10-17T23:02:56.123456",
  "dataset_name": "heart.csv",
  "dataset_shape": [303, 14],
  "task_type": "classification",
  "domain": "healthcare",
  "target_column": "target",
  "model_name": "xgboost",
  "deployment_ready": true,
  "metrics": {
    "accuracy": 0.868,
    "f1_score": 0.871,
    "precision": 0.882,
    "recall": 0.86
  },
  "iterations": 1
}
```

### Full Run Results (stored in `run_YYYYMMDD_HHMMSS.json`)

Contains complete pipeline output including:

- EDA results
- Feature engineering transformations
- All model training results
- Evaluation metrics and explainability
- Judge decisions
- Agent communication logs

---

## Benefits

### 🔄 **Reproducibility**

- Track every experiment
- Revisit past results anytime
- Understand what worked and why

### 📊 **Performance Tracking**

- See improvement over time
- Identify best models
- Compare different approaches

### 🤝 **Team Collaboration**

- Share run history with team members
- Document experiments automatically
- Export results for reports

### 🧹 **Experiment Management**

- Keep workspace organized
- Clean up old runs easily
- Focus on successful approaches

---

## Configuration

No additional configuration needed! The feature works out of the box.

**Default location**: `history/` directory in project root

**Storage format**: JSON (human-readable and easy to parse)

**Automatic cleanup**: Delete runs manually or programmatically

---

## Troubleshooting

### History not showing in sidebar?

- Make sure you've run at least one pipeline
- Refresh the Streamlit page

### Can't load a run?

- Check that the `history/` directory exists
- Verify the `run_*.json` file exists
- Check file permissions

### Comparison not working?

- You need at least 2 runs to compare
- Selected runs must have compatible task types

### Want to reset history?

- Delete all files in `history/` directory (except `.gitkeep`)
- Or delete specific runs via the UI

---

## Future Enhancements

Potential additions:

- 🏷️ **Tags and Categories**: Organize runs by experiment type
- 🔍 **Search and Filter**: Find runs by metrics, model, date
- 📈 **Trend Analysis**: Track performance over time
- 🔔 **Notifications**: Alert when new best model is found
- ☁️ **Cloud Sync**: Save history to cloud storage
- 🤖 **LLM Insights**: Get AI-powered recommendations based on history

---

## License

Part of the Explainable ML Pipeline project. Same license applies.

---

**Happy Experimenting! 🚀**
