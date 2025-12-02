# 🤖 Available ML Models Guide

## Overview

The ML Pipeline now supports **9 different algorithms** for both classification and regression tasks. The LLM intelligently selects the best models based on your dataset characteristics.

---

## 📊 Classification Models

### 1. **Random Forest** (`random_forest`)

```python
RandomForestClassifier(random_state=42)
```

**Best For:**

- ✅ Medium to large datasets (500+ samples)
- ✅ Non-linear relationships
- ✅ Feature importance analysis
- ✅ Handling missing values
- ✅ Robust to outliers

**Hyperparameters Tuned:**

- `n_estimators`: 100, 200, 300
- `max_depth`: 10, 20, 30, None
- `min_samples_split`: 2, 5, 10
- `min_samples_leaf`: 1, 2, 4

**When LLM Suggests:** General-purpose, balanced accuracy

---

### 2. **XGBoost** (`xgboost`)

```python
XGBClassifier(random_state=42, eval_metric='logloss')
```

**Best For:**

- ✅ Structured/tabular data
- ✅ Winning Kaggle competitions
- ✅ Imbalanced datasets
- ✅ High accuracy requirements
- ✅ Gradient boosting power

**Hyperparameters Tuned:**

- `n_estimators`: 100, 200
- `max_depth`: 3, 5, 7
- `learning_rate`: 0.01, 0.1, 0.3
- `subsample`: 0.8, 1.0

**When LLM Suggests:** Healthcare/finance domains, structured data

---

### 3. **Logistic Regression** (`logistic_regression`)

```python
LogisticRegression(random_state=42, max_iter=1000)
```

**Best For:**

- ✅ Binary classification
- ✅ Interpretability required
- ✅ Small to medium datasets
- ✅ Linear relationships
- ✅ Fast training/inference

**Hyperparameters Tuned:**

- `C`: 0.01, 0.1, 1, 10
- `penalty`: l2
- `solver`: lbfgs, liblinear

**When LLM Suggests:** Need interpretability, linear patterns detected

---

### 4. **Support Vector Machine** (`svm`)

```python
SVC(random_state=42)
```

**Best For:**

- ✅ Small to medium datasets (<10k samples)
- ✅ High-dimensional data
- ✅ Clear margin of separation
- ✅ Non-linear decision boundaries (RBF kernel)

**Hyperparameters Tuned:**

- `C`: 0.1, 1, 10
- `kernel`: rbf, linear
- `gamma`: scale, auto

**When LLM Suggests:** Small dataset, complex boundaries

---

### 5. **Decision Tree** (`decision_tree`) ⭐ NEW!

```python
DecisionTreeClassifier(random_state=42)
```

**Best For:**

- ✅ Interpretability (visual tree)
- ✅ Fast training
- ✅ Handling categorical features
- ✅ No data preprocessing needed
- ⚠️ Prone to overfitting

**Hyperparameters Tuned:**

- `max_depth`: 5, 10, 20, None
- `min_samples_split`: 2, 5, 10
- `min_samples_leaf`: 1, 2, 4
- `criterion`: gini, entropy

**When LLM Suggests:** Need quick baseline, interpretability critical

---

### 6. **Neural Network** (`neural_network` or `mlp`) ⭐ NEW!

```python
MLPClassifier(random_state=42, max_iter=1000)
```

**Best For:**

- ✅ Large datasets (1000+ samples)
- ✅ Complex non-linear patterns
- ✅ Deep feature learning
- ⚠️ Requires data scaling
- ⚠️ Slower training

**Hyperparameters Tuned:**

- `hidden_layer_sizes`: (50,), (100,), (50, 50), (100, 50)
- `activation`: relu, tanh
- `alpha`: 0.0001, 0.001, 0.01
- `learning_rate`: constant, adaptive

**When LLM Suggests:** Large dataset, complex patterns, high accuracy needed

---

## 📈 Regression Models

### 1. **Random Forest** (`random_forest`)

```python
RandomForestRegressor(random_state=42)
```

**Best For:**

- ✅ Non-linear relationships
- ✅ Feature importance
- ✅ Robust predictions
- ✅ Handles outliers well

**Same benefits as classification version**

---

### 2. **XGBoost** (`xgboost`)

```python
XGBRegressor(random_state=42)
```

**Best For:**

- ✅ Structured data
- ✅ High R² scores
- ✅ Gradient boosting
- ✅ Production deployments

**Industry standard for structured data regression**

---

### 3. **Ridge Regression** (`ridge`)

```python
Ridge(random_state=42)
```

**Best For:**

- ✅ Linear relationships
- ✅ Multicollinearity
- ✅ Regularization needed
- ✅ Fast and interpretable

**Hyperparameters Tuned:**

- `alpha`: 0.01, 0.1, 1.0, 10.0, 100.0

**When LLM Suggests:** Linear patterns, need regularization

---

### 4. **Linear Regression** (`linear_regression`)

```python
Ridge(random_state=42)  # Uses Ridge with tuning
```

**Best For:**

- ✅ Simple baseline
- ✅ Interpretability
- ✅ Fast training
- ✅ Linear relationships

**When LLM Suggests:** Starting point, baseline model

---

### 5. **Support Vector Regression** (`svm`)

```python
SVR()
```

**Best For:**

- ✅ Small datasets
- ✅ Non-linear patterns
- ✅ Kernel methods
- ⚠️ Slower on large data

---

### 6. **Decision Tree** (`decision_tree`) ⭐ NEW!

```python
DecisionTreeRegressor(random_state=42)
```

**Best For:**

- ✅ Interpretable predictions
- ✅ Fast training
- ✅ No scaling needed
- ⚠️ Can overfit

---

### 7. **Neural Network** (`neural_network` or `mlp`) ⭐ NEW!

```python
MLPRegressor(random_state=42, max_iter=1000)
```

**Best For:**

- ✅ Large datasets
- ✅ Complex patterns
- ✅ Deep learning
- ⚠️ Needs scaling

---

## 🧠 LLM Model Selection Logic

### Dataset Size-Based:

```
Small (<500 rows):
  → LLM suggests: logistic_regression, svm, decision_tree

Medium (500-5000 rows):
  → LLM suggests: random_forest, xgboost, neural_network

Large (5000+ rows):
  → LLM suggests: xgboost, neural_network, random_forest
```

### Domain-Based:

```
Healthcare:
  → Interpretability matters: logistic_regression, decision_tree
  → Accuracy critical: xgboost, random_forest

Finance:
  → Regulatory compliance: logistic_regression, decision_tree
  → Fraud detection: xgboost, neural_network, random_forest

General:
  → Balanced approach: xgboost, random_forest, neural_network
```

### Task-Based:

```
Binary Classification:
  → logistic_regression, xgboost, neural_network

Multi-class Classification:
  → random_forest, xgboost, neural_network

Regression:
  → xgboost, ridge, neural_network
```

---

## ⚙️ Configuration

### Enable All Models in `config.yaml`:

```yaml
agents:
  model_tuning:
    enabled: true
    algorithms: # Ignored if LLM enabled
      - random_forest
      - xgboost
      - decision_tree

llm:
  reasoning_enabled: true # Let LLM choose models
  model: "llama3.1:8b"
```

### Manual Model Selection:

```yaml
agents:
  model_tuning:
    algorithms:
      - neural_network # Deep learning
      - decision_tree # Interpretable
      - xgboost # High accuracy
```

---

## 📊 Performance Comparison

### Speed (Training Time):

```
Fastest:  decision_tree < logistic_regression < linear_regression
Medium:   random_forest < svm
Slowest:  xgboost < neural_network
```

### Accuracy (General):

```
Highest:  xgboost ≈ neural_network ≈ random_forest
Medium:   svm ≈ ridge
Lower:    decision_tree ≈ logistic_regression
```

### Interpretability:

```
Most:     decision_tree > logistic_regression > linear_regression
Medium:   random_forest > ridge
Least:    neural_network < xgboost < svm
```

---

## 🎯 Model Selection Flowchart

```
Start
│
├─ Need Interpretability?
│  ├─ Yes → decision_tree, logistic_regression, ridge
│  └─ No  → Continue
│
├─ Large Dataset (>5000)?
│  ├─ Yes → xgboost, neural_network, random_forest
│  └─ No  → Continue
│
├─ Complex Patterns?
│  ├─ Yes → xgboost, neural_network, random_forest
│  └─ No  → logistic_regression, ridge, svm
│
└─ Time Constrained?
   ├─ Yes → decision_tree, logistic_regression
   └─ No  → xgboost, neural_network
```

---

## 🚫 About Clustering

**Note:** Clustering algorithms (K-Means, DBSCAN, etc.) are **unsupervised learning** and don't fit in this **supervised learning** pipeline since we have labeled data (target column).

If you need clustering:

1. Use for **exploratory data analysis** (separate script)
2. Create **features from clusters** (cluster ID as a feature)
3. Use in **anomaly detection** (outlier analysis)

But they cannot be used as primary models in a supervised classification/regression pipeline.

---

## 🧪 Example LLM Suggestions

### Wine Quality (Regression, 1599 rows):

```
🧠 LLM: "xgboost, ridge, random_forest"
Reasoning: Medium dataset, structured data, regression task
```

### Heart Disease (Classification, 303 rows):

```
🧠 LLM: "logistic_regression, decision_tree, svm"
Reasoning: Small dataset, healthcare (interpretability), binary classification
```

### Fraud Detection (Classification, 50k rows):

```
🧠 LLM: "xgboost, neural_network, random_forest"
Reasoning: Large dataset, imbalanced, needs high accuracy
```

---

## 🎓 Model Usage Tips

### 1. **Start Simple**

- Use decision_tree or logistic_regression as baseline
- Establish performance benchmark
- Understand data patterns

### 2. **Scale Up**

- Try xgboost or random_forest for better accuracy
- Use neural_network for complex patterns
- Compare performance gains

### 3. **Optimize**

- Hyperparameter tuning (done automatically)
- Feature engineering
- Ensemble methods

### 4. **Deploy**

- Choose based on accuracy + interpretability + speed
- Document model selection reasoning
- Monitor in production

---

## 📚 When to Use Each Model

| Model          | Small Data | Large Data | Interpretable | Fast Training | High Accuracy |
| -------------- | ---------- | ---------- | ------------- | ------------- | ------------- |
| Decision Tree  | ✅         | ❌         | ✅            | ✅            | ⚠️            |
| Logistic Reg   | ✅         | ⚠️         | ✅            | ✅            | ⚠️            |
| SVM            | ✅         | ❌         | ❌            | ⚠️            | ✅            |
| Random Forest  | ⚠️         | ✅         | ⚠️            | ⚠️            | ✅            |
| XGBoost        | ✅         | ✅         | ❌            | ❌            | ✅            |
| Neural Network | ❌         | ✅         | ❌            | ❌            | ✅            |
| Ridge          | ✅         | ✅         | ✅            | ✅            | ⚠️            |

✅ = Excellent, ⚠️ = Good, ❌ = Not Recommended

---

## 🚀 Getting Started

### Run with LLM Model Selection:

```bash
# LLM chooses best models automatically
streamlit run app.py
```

### Run with Specific Models:

```bash
# Edit config.yaml first
llm:
  reasoning_enabled: false

agents:
  model_tuning:
    algorithms:
      - neural_network
      - xgboost
      - decision_tree
```

---

**The LLM will intelligently select the best models based on your data!** 🧠✨

