# 🧠 LLM-Driven Model Selection Feature

## Overview

The pipeline now uses **LLM intelligence** to suggest different models across iterations, making each training cycle actually improve rather than just repeating the same models.

---

## 🎯 Problem Solved

**Before:** Each iteration trained the same models (e.g., Random Forest + XGBoost), so there was no actual improvement across iterations.

**After:** The LLM analyzes results and suggests **different algorithms** for the next iteration based on performance patterns.

---

## 🔄 How It Works

### Iteration Flow:

```
BEFORE Iteration 1:
└── 🧠 LLM Analyzes: Dataset (1599 rows, 11 features, regression, wine domain)
    └── Recommends: "xgboost,ridge,random_forest"

Iteration 1:
├── Train: XGBoost, Ridge, Random Forest  ← LLM suggestions!
├── Evaluate: R²=0.75, RMSE=0.82
├── Judge: ❌ Below threshold (0.80)
└── 🧠 LLM Suggests: "Try linear_regression and svm for better generalization"

Iteration 2:
├── Train: Linear Regression, SVM  ← Different models!
├── Evaluate: R²=0.82, RMSE=0.71
├── Judge: ✅ Approved!
└── Deploy model
```

---

## 🏗️ Architecture

### 1. **Judge Agent** (Decision Maker)

```python
# Judge asks LLM for model recommendations
llm_model_suggestion = self.llm_reason(
    prompt="""Based on current performance, suggest which ML algorithms to try next.

Current model: random_forest
Performance: F1=0.75, Accuracy=0.78
Task: classification

Available: random_forest, xgboost, logistic_regression, svm

Respond with ONLY: algorithm1,algorithm2,algorithm3""",
    context={...}
)
```

### 2. **Orchestrator** (Coordinator)

```python
# Track models across iterations
all_models_tried = []  # e.g., ['random_forest', 'xgboost']

# Get LLM recommendations
recommended_models = judge_results.get("recommended_models", [])

# Pass to next iteration
pipeline_input["suggested_algorithms"] = recommended_models
```

### 4. **Model Tuning Agent** (Executor)

```python
# Use LLM suggestions if available
suggested_algorithms = input_data.get("suggested_algorithms", [])
if suggested_algorithms:
    algorithms = suggested_algorithms  # Use LLM recommendations
else:
    algorithms = ["random_forest", "xgboost"]  # Default
```

---

## 📊 Key Components

### Judge Agent Enhancements

**New Method:** `_parse_model_recommendations()`

- Parses LLM response to extract valid model names
- Validates against task type (classification/regression)
- Falls back to defaults if parsing fails

**LLM Prompt:**

```
Based on the current model performance, suggest which ML algorithms to try in the next iteration.

Current model: {current_model}
Task type: {task_type}
Performance: {metrics}
Models already tried: {all_models_tried}

Available algorithms:
- random_forest: Good for complex non-linear relationships
- xgboost: Excellent for structured data
- logistic_regression: Fast, interpretable
- svm: Good for small datasets
- linear_regression (regression only)
- ridge (regression only)

Respond with ONLY a comma-separated list of 2-3 algorithm names.
Example: xgboost,random_forest,svm
```

### Orchestrator Enhancements

**Tracking:**

- `all_models_tried`: List of all models across iterations
- `recommended_models`: LLM suggestions for next iteration

**Logging:**

```
🧠 Using LLM-recommended models: ['svm', 'logistic_regression']
🎯 Next iteration will try: svm, logistic_regression
```

### Model Tuning Agent Enhancements

**Input Parameter:**

- `suggested_algorithms`: List from LLM (optional)
- Falls back to config if not provided

---

## 🎮 Usage

### Enable LLM Reasoning

In `config.yaml`:

```yaml
llm:
  provider: "ollama"
  model: "llama3.1:8b"
  reasoning_enabled: true # Must be true
```

### Run Pipeline

The LLM will automatically:

1. Analyze model performance
2. Suggest different algorithms
3. Apply suggestions in next iteration

**No manual intervention needed!**

---

## 📈 Example Workflow

### Healthcare Heart Disease Dataset

**Iteration 1:**

```
Training: Random Forest, XGBoost
Results:
  - Random Forest: F1=0.87
  - XGBoost: F1=0.85

Judge: ❌ Below 0.90 threshold

LLM Analysis:
"Both tree-based models show overfitting.
Try linear models for better generalization:
logistic_regression,svm"
```

**Iteration 2:**

```
Training: Logistic Regression, SVM  ← NEW MODELS
Results:
  - Logistic Regression: F1=0.92  ← BETTER!
  - SVM: F1=0.89

Judge: ✅ Approved (F1=0.92 > 0.90)
```

---

## 🔍 LLM Decision Logic

The LLM considers:

1. **Current Performance**: Are metrics improving or degrading?
2. **Model Type**: Tree-based vs linear vs ensemble
3. **Domain Context**: Healthcare/finance specific needs
4. **Previous Models**: What has already been tried?
5. **Task Type**: Classification vs regression constraints

### Example LLM Responses:

**Scenario 1: Overfitting**

```
Input: Random Forest F1=0.95 (training), F1=0.75 (test)
LLM: "High variance detected. Try regularized models:
      ridge,logistic_regression"
```

**Scenario 2: Underfitting**

```
Input: Logistic Regression F1=0.65
LLM: "Linear model underperforms. Try non-linear:
      xgboost,random_forest,svm"
```

**Scenario 3: Good Performance**

```
Input: XGBoost F1=0.88 (close to threshold)
LLM: "Try ensemble approach for final boost:
      random_forest,xgboost"  (different params)
```

---

## 🎯 Benefits

### 1. **Intelligent Iteration**

- Each iteration tries something different
- Actual improvement over time
- No wasted compute on repeating same models

### 2. **Domain-Aware**

- LLM considers healthcare/finance context
- Suggests interpretable models when needed
- Balances accuracy vs explainability

### 3. **Adaptive Strategy**

- Responds to overfitting/underfitting
- Explores model diversity
- Learns from previous attempts

### 4. **Fully Automated**

- No manual model selection
- Self-improving pipeline
- Reduces ML expertise requirements

---

## 🛠️ Configuration

### Default Models (No LLM)

```yaml
agents:
  model_tuning:
    algorithms:
      - random_forest
      - xgboost
```

### LLM-Driven (Recommended)

```yaml
llm:
  reasoning_enabled: true
  model: "llama3.1:8b"

agents:
  judge:
    max_retrain_cycles: 3 # Allow multiple iterations
  model_tuning:
    algorithms: # Ignored if LLM enabled
      - random_forest
      - xgboost
```

---

## 📊 Monitoring

### Check LLM Recommendations

**In Logs:**

```
INFO: LLM recommends models for next iteration: ['svm', 'logistic_regression']
INFO: 🧠 Using LLM-recommended models: ['svm', 'logistic_regression']
INFO: 🎯 Next iteration will try: svm, logistic_regression
```

**In Results JSON:**

```json
{
  "judgment": {
    "recommended_models": ["svm", "logistic_regression"],
    "llm_decision_support": "Model shows overfitting..."
  }
}
```

### Streamlit UI

The **Agents** tab will show:

- LLM recommendations in agent messages
- Model changes across iterations
- Decision reasoning

---

## 🚀 Advanced Usage

### Custom Model Pool

Modify valid models in `judge_agent.py`:

```python
valid_models = {
    "classification": [
        "random_forest",
        "xgboost",
        "logistic_regression",
        "svm",
        "naive_bayes",  # Add custom
        "mlp"           # Add custom
    ]
}
```

### Custom LLM Prompt

Edit prompt in `judge_agent.py` line ~93:

```python
llm_model_suggestion = self.llm_reason(
    prompt=f"""Your custom prompt here...

    Include: {current_model}, {metrics}, etc.
    """,
    context={...}
)
```

---

## 🧪 Testing

### Test with Wine Quality Dataset

```bash
# Run with regression
python main.py --data wine.csv --target quality --task regression --domain general

# Watch logs for LLM suggestions
# Iteration 1: random_forest, xgboost
# Iteration 2: ridge, linear_regression  ← Different!
```

---

## ⚠️ Troubleshooting

### LLM Not Suggesting Models

**Check:**

1. `llm.reasoning_enabled: true` in config
2. Ollama is running: `ollama run llama3.1:8b`
3. Model needs retraining (threshold not met)
4. Logs show "LLM recommends models..."

### Same Models Every Iteration

**Possible Causes:**

- LLM disabled
- LLM response not parsed correctly
- All models already tried (increase pool)

**Fix:**

```bash
# Check logs
grep "LLM recommends" logs/pipeline.log

# Enable debug logging
export LOGLEVEL=DEBUG
```

### Invalid Model Suggestions

**Safety:**

- Parser validates against `valid_models`
- Falls back to defaults if invalid
- Limits to 3 models max per iteration

---

## 📚 Related Documentation

- `README.md` - Main project documentation
- `HISTORY_FEATURE.md` - Run history and comparison
- `UI_README.md` - Streamlit UI guide
- `config.yaml` - Configuration options

---

## 🎓 Key Takeaways

✅ **LLM analyzes** model performance patterns  
✅ **Suggests different models** for each iteration  
✅ **Pipeline adapts** based on intelligent recommendations  
✅ **Fully automated** - no manual model selection needed  
✅ **Domain-aware** - considers healthcare/finance context  
✅ **Logged and tracked** - full transparency

---

**This is what makes the agentic pipeline truly "intelligent" compared to traditional AutoML!** 🚀🧠
