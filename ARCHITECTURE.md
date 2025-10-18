# System Architecture

## Overview

The Explainable ML Pipeline is built on an **agentic architecture** where specialized, autonomous agents collaborate to execute machine learning workflows. Each agent is responsible for a specific phase of the ML lifecycle and communicates with other agents through a structured JSON messaging protocol.

## Core Design Principles

1. **Modularity**: Each agent is self-contained and can be developed/tested independently
2. **Extensibility**: New agents and algorithms can be added without modifying existing code
3. **Reproducibility**: All decisions and transformations are logged and versioned
4. **Explainability**: Built-in support for model interpretability at every stage
5. **Autonomy**: Agents make intelligent decisions based on data and configuration
6. **Communication**: Standardized JSON-based messaging between components

## Component Architecture

### 1. Communication Layer

**Location**: `src/core/communication.py`

**Purpose**: Provides structured messaging between agents

**Key Components**:

- `AgentMessage`: Pydantic model for type-safe messages
- `MessageType`: Enum defining message types (REQUEST, RESPONSE, ERROR, etc.)
- `CommunicationProtocol`: Manages message routing and logging

**Message Structure**:

```json
{
  "sender": "agent_name",
  "receiver": "target_agent",
  "message_type": "request",
  "timestamp": "2024-01-01T12:00:00",
  "content": {
    "data": "...",
    "parameters": {}
  },
  "metadata": {}
}
```

### 2. Base Agent

**Location**: `src/core/base_agent.py`

**Purpose**: Abstract base class for all agents

**Key Methods**:

- `execute(input_data)`: Main execution method (must be implemented)
- `receive_message(message)`: Process incoming messages
- `send_message(receiver, content)`: Send messages to other agents
- `llm_reason(prompt, context)`: Placeholder for LLM integration
- `save_state()`: Persist agent state
- `reset()`: Reset agent for new pipeline run

**Agent State**:

- Configuration from config.yaml
- Execution results
- Communication history
- LLM reasoning (if enabled)

### 3. Specialized Agents

#### EDA Agent

**Location**: `src/agents/eda_agent.py`

**Responsibilities**:

- Statistical analysis of dataset
- Missing value detection and analysis
- Correlation analysis between features
- Outlier detection (IQR method)
- Data quality scoring
- Automated recommendations

**Output Format**:

```json
{
  "dataset_info": {...},
  "statistical_summary": {...},
  "missing_values": {...},
  "correlation_analysis": {...},
  "outliers": {...},
  "data_quality_score": 0.85,
  "recommendations": [...]
}
```

#### Feature Engineering Agent

**Location**: `src/agents/feature_engineering_agent.py`

**Responsibilities**:

- Missing value imputation
- Categorical encoding (one-hot, label)
- Feature scaling (standard, minmax, robust)
- Feature creation (polynomial, log transforms, domain-specific)
- Feature selection (importance, mutual info, RFE)

**Pipeline**:

```
Raw Data → Imputation → Encoding → Feature Creation → Scaling → Selection → Processed Data
```

**Artifacts**:

- Scalers (saved for production)
- Encoders (saved for production)
- Selected features list

#### Model Tuning Agent

**Location**: `src/agents/model_tuning_agent.py`

**Responsibilities**:

- Multi-algorithm training (RF, XGBoost, LR, SVM)
- Hyperparameter optimization (Grid/Random Search)
- Cross-validation
- Model selection based on CV scores
- Feature importance extraction

**Supported Algorithms**:

- Classification: Random Forest, XGBoost, Logistic Regression, SVM
- Regression: Random Forest, XGBoost, Linear Regression (Ridge)

**Tuning Strategy**:

- RandomizedSearchCV for large parameter spaces
- GridSearchCV for exhaustive search
- 5-fold cross-validation by default
- Parallel execution (n_jobs=-1)

#### Evaluator Agent

**Location**: `src/agents/evaluator_agent.py`

**Responsibilities**:

- Comprehensive metrics calculation
- Confusion matrix generation
- Classification/regression reports
- Explainability analysis (SHAP, LIME placeholders)
- Fairness checking
- Performance recommendations

**Metrics by Task**:

- **Classification**: accuracy, precision, recall, F1, ROC-AUC
- **Regression**: RMSE, MAE, R², MAPE

**Explainability Methods**:

- Feature importance (built-in)
- SHAP values (integration placeholder)
- LIME explanations (integration placeholder)

#### Judge Agent

**Location**: `src/agents/judge_agent.py`

**Responsibilities**:

- Performance threshold checking
- Deployment readiness assessment
- Retrain decision logic
- Performance trend analysis
- Quality gate enforcement

**Decision Logic**:

```python
if performance_score >= threshold:
    return "APPROVED"
elif retrain_count < max_retrains:
    return "RETRAIN_REQUIRED"
else:
    return "MANUAL_INTERVENTION_REQUIRED"
```

**Tracked Metrics**:

- Performance history across iterations
- Performance trends (improving/declining/stable)
- Retrain count
- Deployment confidence score

### 4. Orchestrator

**Location**: `src/orchestrator.py`

**Purpose**: Coordinates workflow and manages pipeline execution

**Responsibilities**:

- Initialize all agents
- Execute sequential workflow
- Manage retraining loops
- Handle state across iterations
- Save intermediate results
- Generate final reports

**Workflow**:

```
1. EDA Agent → Analyze data
2. Feature Engineering Agent → Process features
3. Model Tuning Agent → Train models
4. Evaluator Agent → Calculate metrics
5. Judge Agent → Make decision
   ├─ If approved → End
   ├─ If retrain required → Go to step 2
   └─ If max retrains → End with warning
```

**State Management**:

- Current iteration number
- Results from each agent per iteration
- Final decision and recommendations
- Execution time and metadata

## Data Flow

```
Input Data (CSV/Excel/Parquet)
    ↓
┌─────────────────────────────────┐
│     EDA Agent                   │
│  • Analyze structure            │
│  • Check quality                │
│  • Generate recommendations     │
└─────────────────────────────────┘
    ↓ (EDA Results)
┌─────────────────────────────────┐
│  Feature Engineering Agent      │
│  • Impute missing values        │
│  • Encode categoricals          │
│  • Create new features          │
│  • Scale numerical features     │
│  • Select important features    │
└─────────────────────────────────┘
    ↓ (Processed Data)
┌─────────────────────────────────┐
│   Model Tuning Agent            │
│  • Train multiple algorithms    │
│  • Hyperparameter tuning        │
│  • Cross-validation             │
│  • Select best model            │
└─────────────────────────────────┘
    ↓ (Trained Model)
┌─────────────────────────────────┐
│    Evaluator Agent              │
│  • Calculate metrics            │
│  • Generate explanations        │
│  • Check fairness               │
│  • Provide recommendations      │
└─────────────────────────────────┘
    ↓ (Evaluation Results)
┌─────────────────────────────────┐
│      Judge Agent                │
│  • Check thresholds             │
│  • Assess deployment readiness  │
│  • Decide: Deploy or Retrain?   │
└─────────────────────────────────┘
    ↓
Decision: Approved / Retrain / Manual Intervention
```

## Configuration System

**Location**: `config.yaml`

**Structure**:

```yaml
project:
  name: "Explainable ML Pipelines"
  version: "1.0.0"

paths:
  data_dir: "./data"
  models_dir: "./models"
  logs_dir: "./logs"
  results_dir: "./results"

agents:
  eda:
    enabled: true
    correlation_threshold: 0.7
    # ...

  feature_engineering:
    enabled: true
    scaling_method: "standard"
    # ...

  # ... other agents

llm:
  provider: "ollama"
  reasoning_enabled: false

orchestrator:
  execution_mode: "sequential"
  save_intermediate_results: true
```

## Extensibility Points

### Adding New Agents

1. Create new agent class:

```python
from src.core.base_agent import BaseAgent

class MyNewAgent(BaseAgent):
    def execute(self, input_data):
        # Implementation
        return results
```

2. Register in orchestrator:

```python
self.agents["my_new_agent"] = MyNewAgent(...)
```

3. Add to workflow sequence

### Adding New Algorithms

Modify `ModelTuningAgent._get_model_and_params()`:

```python
if algorithm == "my_algorithm":
    model = MyModel()
    params = {...}
    return model, params
```

### Adding LLM Reasoning

Implement in `BaseAgent.llm_reason()`:

```python
from langchain.llms import Ollama

llm = Ollama(model="llama2")
response = llm(prompt)
return response
```

## Error Handling

- All agents wrapped in try-except blocks
- Errors communicated via ERROR message type
- Failed agents don't crash pipeline
- Errors logged with full traceback

## Performance Considerations

- **Scalability**: Use sampling for very large datasets in EDA
- **Parallelization**: Model tuning uses n_jobs=-1
- **Memory**: Feature selection reduces dimensionality
- **Caching**: Intermediate results saved to disk

## Security Considerations

- No credentials in configuration files (use environment variables)
- Input validation at each agent
- Secure model serialization (joblib)
- Audit trail of all decisions

## Future Enhancements

1. **Distributed Execution**: Multi-machine orchestration
2. **Real-time Monitoring**: Live dashboard for pipeline execution
3. **Auto-ML**: Automated algorithm and architecture search
4. **Model Registry**: Centralized model versioning
5. **A/B Testing**: Built-in experiment framework
6. **Drift Detection**: Continuous monitoring in production

---

This architecture provides a solid foundation for building production-grade, explainable ML systems with built-in quality assurance and autonomous decision-making.
