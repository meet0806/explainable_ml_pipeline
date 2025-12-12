# 🔍 Explainability Guide

Complete guide to using SHAP and LIME for model interpretability in the Explainable ML Pipeline.

## 📋 Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
- [Methods Available](#methods-available)
- [Usage Examples](#usage-examples)
- [Visualization](#visualization)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project includes **full integration** of three powerful explainability methods:

1. **Feature Importance** - Built-in model importance scores
2. **SHAP** - SHapley Additive exPlanations
3. **LIME** - Local Interpretable Model-agnostic Explanations

All methods are automatically computed during the evaluation phase and displayed in the Streamlit UI.

---

## Configuration

### Enable Explainability Methods

Edit `config.yaml`:

```yaml
agents:
  evaluator:
    enabled: true
    explainability_methods: ["shap", "lime"]  # Enable both methods
```

### Install Required Libraries

```bash
pip install shap lime
```

These libraries are already included in `requirements.txt`.

---

## Methods Available

### 1. Feature Importance

**What it is:**
- Built-in importance scores from tree-based models
- Shows global feature relevance
- Fast to compute

**When to use:**
- Quick overview of important features
- Tree-based models (Random Forest, XGBoost)
- Initial feature analysis

**Example output:**
```python
{
  "feature_1": 0.234,
  "feature_2": 0.187,
  "feature_3": 0.145
}
```

---

### 2. SHAP (SHapley Additive exPlanations)

**What it is:**
- Game theory-based approach to explain predictions
- Fairly distributes prediction contribution among features
- Works with any model type

**Key Benefits:**
- **Theoretically sound**: Based on Shapley values from game theory
- **Consistent**: Same contribution for same feature across similar instances
- **Global & Local**: Can explain individual predictions or overall model behavior
- **Model-agnostic**: Works with any ML model

**How it works:**
1. For each prediction, SHAP calculates how much each feature contributed
2. Values are additive: sum of all SHAP values = prediction - baseline
3. Positive values push prediction higher, negative values push it lower

**Example interpretation:**
```
Patient diagnosis prediction = 0.75 (75% probability of disease)

SHAP contributions:
- age (+0.15): Older age increases disease risk
- blood_pressure (+0.10): High BP increases risk  
- exercise_frequency (-0.08): Regular exercise decreases risk
- baseline (0.48): Average baseline risk
```

---

### 3. LIME (Local Interpretable Model-agnostic Explanations)

**What it is:**
- Creates simple, interpretable models locally around each prediction
- Explains individual predictions by perturbing inputs
- Model-agnostic approach

**Key Benefits:**
- **Instance-specific**: Explains individual predictions in detail
- **Intuitive**: Easy to understand local approximations
- **Model-agnostic**: Works with any model (even black boxes)
- **Feature interactions**: Captures local feature relationships

**How it works:**
1. Select an instance to explain
2. Generate similar instances by perturbing features
3. Train a simple model (e.g., linear) on these instances
4. Use the simple model to explain the prediction

**Example interpretation:**
```
Loan application prediction = Approved

LIME explanation for this applicant:
- income > 50K (+0.45): Strong positive factor
- employment_years > 5 (+0.30): Positive factor
- debt_ratio < 0.3 (+0.25): Positive factor
- age = 28 (-0.12): Slight negative factor
```

---

## Usage Examples

### Via Command Line

```bash
# Run pipeline with explainability enabled
python main.py \
  --data data/heart.csv \
  --target target \
  --task classification \
  --domain healthcare
```

Results will include explainability analysis in the output JSON.

### Via Python API

```python
from src.orchestrator import Orchestrator
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize orchestrator
orchestrator = Orchestrator(config)

# Run pipeline
results = orchestrator.run_pipeline(
    data=df,
    target_column='target',
    task_type='classification',
    domain='healthcare'
)

# Access explainability results
explainability = results['final_results']['explainability']

# SHAP feature importance
shap_importance = explainability['shap_feature_importance']
print("Top features by SHAP:", shap_importance)

# LIME explanations
lime_explanations = explainability['lime_sample_explanations']
for exp in lime_explanations:
    print(f"Instance {exp['instance_index']}:", exp['feature_contributions'])
```

### Via Streamlit UI

1. Upload your dataset
2. Run the pipeline
3. Navigate to **📊 Results** tab
4. Scroll to **🔍 Model Explainability** section
5. Explore tabs:
   - 📊 Feature Importance
   - 🎯 SHAP Analysis
   - 🔬 LIME Analysis
   - ⚖️ Method Comparison

---

## Visualization

### Available Visualizations

#### 1. Feature Importance Bar Charts
- Horizontal bar charts showing top 10 features
- Sorted by importance score
- Available for all three methods

#### 2. SHAP Summary
- Feature importance based on mean absolute SHAP values
- Color-coded by impact magnitude
- Shows number of samples analyzed

#### 3. LIME Instance Explanations
- Individual prediction explanations
- Positive (green) and negative (red) contributions
- Multiple sample instances shown

#### 4. Method Comparison
- Side-by-side comparison of all methods
- Normalized importance scores
- Grouped bar chart format

### Interpreting Visualizations

**High importance = high impact on predictions**

- Features at the top have the most influence
- Longer bars = more important features
- Colors help distinguish magnitude

---

## Best Practices

### 1. When to Use Each Method

| Use Case | Recommended Method |
|----------|-------------------|
| Quick overview | Feature Importance |
| Global model understanding | SHAP |
| Individual prediction explanation | LIME |
| Regulatory compliance | SHAP + LIME |
| Tree-based models | Feature Importance + SHAP |
| Black-box models | LIME |
| Feature selection | SHAP |
| Debugging predictions | LIME |

### 2. Performance Considerations

**SHAP:**
- Fast for tree-based models (TreeExplainer)
- Slower for other models (KernelExplainer)
- Sample data if dataset is large (default: 100 samples)

**LIME:**
- Moderate speed
- Explain representative samples (default: 3 instances)
- Increase `num_features` for more detailed explanations

### 3. Interpretation Tips

**Healthcare Domain:**
- Look for medically relevant features (age, vitals, history)
- Validate that relationships make clinical sense
- Use explanations to build trust with clinicians

**Finance Domain:**
- Check for expected economic relationships
- Verify fairness across demographic groups
- Use for regulatory compliance and transparency

### 4. Troubleshooting Common Issues

**"SHAP library not installed"**
```bash
pip install shap
```

**"LIME library not installed"**
```bash
pip install lime
```

**SHAP computation is slow:**
- Already optimized: uses 100 samples max
- For tree models, uses fast TreeExplainer
- For others, uses sampled background data

**Different methods rank features differently:**
- This is normal and expected
- Each method has different strengths
- Use multiple methods for robust understanding

---

## Advanced Configuration

### Customize Sample Sizes

Edit `src/agents/evaluator_agent.py`:

```python
# SHAP sample size (line ~230)
sample_size = min(100, len(X_test))  # Increase for more accuracy

# LIME number of instances (line ~290)
num_samples = min(3, len(X_test))  # Increase for more examples
```

### Add Custom Explainability Methods

Extend `_generate_explainability()` in `evaluator_agent.py`:

```python
# Add your custom method
if "custom_method" in self.eval_config.get("explainability_methods", []):
    try:
        # Your implementation
        custom_results = compute_custom_explainability(model, X_test)
        explainability["custom_results"] = custom_results
        explainability["methods_available"].append("custom_method")
    except Exception as e:
        self.logger.warning(f"Custom method failed: {e}")
```

---

## Technical Details

### SHAP Implementation

**For tree-based models (Random Forest, XGBoost, Decision Tree):**
```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
```

**For other models (Logistic Regression, SVM, Neural Networks):**
```python
background = shap.sample(X_train, 100)
explainer = shap.KernelExplainer(model.predict, background)
shap_values = explainer.shap_values(X_sample)
```

### LIME Implementation

```python
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    mode='classification',  # or 'regression'
    random_state=42
)

explanation = explainer.explain_instance(
    X_test.iloc[idx].values,
    model.predict_proba,  # or model.predict
    num_features=10
)
```

---

## References

### Academic Papers

**SHAP:**
- Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." *Advances in neural information processing systems*, 30.

**LIME:**
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you? Explaining the predictions of any classifier." *Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining*.

### Online Resources

- SHAP Documentation: https://shap.readthedocs.io/
- LIME Documentation: https://lime-ml.readthedocs.io/
- Interpretable ML Book: https://christophm.github.io/interpretable-ml-book/

---

## Examples

### Healthcare: Heart Disease Prediction

```python
# Top features by SHAP
{
  "age": 0.145,              # Age is most important
  "chest_pain_type": 0.132,   # Type of chest pain
  "max_heart_rate": 0.098,    # Maximum heart rate achieved
  "blood_pressure": 0.087,    # Resting blood pressure
  "cholesterol": 0.076        # Serum cholesterol
}

# LIME explanation for a specific patient
Instance 0: High Risk (0.89 probability)
- age > 60 (+0.35): Older age increases risk
- chest_pain_type = 3 (+0.28): Asymptomatic pain concerning
- max_heart_rate < 120 (+0.19): Low max heart rate indicates risk
- exercise_induced_angina = Yes (+0.15): Exercise angina is risk factor
```

### Finance: Credit Default Prediction

```python
# Top features by SHAP
{
  "income": 0.198,           # Annual income most important
  "debt_to_income": 0.167,   # Debt-to-income ratio
  "credit_history": 0.143,   # Years of credit history
  "employment_length": 0.129, # Employment stability
  "loan_amount": 0.115       # Requested loan amount
}

# LIME explanation for a specific application
Instance 0: Approved (0.78 probability)
- income > 75K (+0.42): High income strong positive
- debt_to_income < 0.25 (+0.31): Low debt ratio positive
- credit_history > 10 (+0.24): Long history positive
- employment_length > 5 (+0.19): Stable employment positive
```

---

## Summary

✅ **Full integration** of SHAP and LIME for comprehensive model explainability  
✅ **Automatic computation** during evaluation phase  
✅ **Rich visualizations** in Streamlit UI  
✅ **Multiple methods** for robust interpretation  
✅ **Production-ready** with error handling and logging  

For questions or issues, please refer to the main README.md or open a GitHub issue.

---

**Built with ❤️ for Explainable AI in Healthcare and Finance**
