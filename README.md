# Explainable ML Pipelines with Agentic AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, agentic machine learning pipeline system designed for **explainable AI** in **Healthcare** and **Finance** domains. The system uses autonomous agents that communicate through structured JSON messages to perform end-to-end ML workflows including EDA, feature engineering, model tuning, evaluation, and deployment decisions.

## 🎯 Project Overview

This project implements an **agentic AI architecture** where specialized agents work together to build, evaluate, and deploy machine learning models with built-in explainability and quality assurance.

### Key Features

- **🤖 Autonomous Agents**: 5 specialized agents (EDA, Feature Engineering, Model Tuning, Evaluator, Judge)
- **📊 Explainability First**: Built-in SHAP and LIME integration placeholders
- **🔄 Iterative Improvement**: Automatic retraining loop based on performance
- **🏥💰 Domain-Aware**: Specialized processing for Healthcare and Finance
- **📝 Structured Communication**: JSON-based agent messaging protocol
- **🎓 Production-Ready**: Modular, testable, and extensible design
- **🔍 Comprehensive Logging**: Full audit trail of agent decisions

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Orchestrator                         │
│         (Coordinates workflow and manages state)         │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  EDA Agent   │  │   Feature    │  │    Model     │
│              │──│  Engineering │──│   Tuning     │
│  • Stats     │  │   Agent      │  │   Agent      │
│  • Missing   │  │              │  │              │
│  • Outliers  │  │  • Scaling   │  │  • RF, XGB   │
│  • Quality   │  │  • Encoding  │  │  • CV tuning │
└──────────────┘  │  • Selection │  │  • Best model│
                  └──────────────┘  └──────────────┘
                          │                 │
                          ▼                 ▼
                  ┌──────────────┐  ┌──────────────┐
                  │  Evaluator   │  │    Judge     │
                  │    Agent     │──│    Agent     │
                  │              │  │              │
                  │  • Metrics   │  │  • Approval  │
                  │  • SHAP/LIME │  │  • Retrain?  │
                  │  • Fairness  │  │  • Deploy?   │
                  └──────────────┘  └──────────────┘
```

## 📁 Project Structure

```
.
├── config.yaml                 # Main configuration file
├── requirements.txt            # Python dependencies
├── main.py                     # Main entry point
│
├── src/
│   ├── __init__.py
│   ├── core/                   # Core framework
│   │   ├── base_agent.py       # Base agent class
│   │   └── communication.py    # Agent messaging protocol
│   │
│   ├── agents/                 # Specialized agents
│   │   ├── eda_agent.py
│   │   ├── feature_engineering_agent.py
│   │   ├── model_tuning_agent.py
│   │   ├── evaluator_agent.py
│   │   └── judge_agent.py
│   │
│   └── orchestrator.py         # Workflow coordinator
│
├── examples/                   # Domain-specific examples
│   ├── example_healthcare.py
│   └── example_finance.py
│
├── data/                       # Data directory
│   └── sample_data_generator.py
│
├── models/                     # Saved models
├── logs/                       # Execution logs
└── results/                    # Pipeline results
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**:

```bash
git clone <repository-url>
cd explainable-ml-pipeline
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Create necessary directories**:

```bash
mkdir -p data models logs results
```

### Basic Usage

#### Command Line Interface

```bash
python main.py \
  --data data/sample_classification.csv \
  --target target \
  --task classification \
  --domain healthcare \
  --save-model
```

#### Python API

```python
import pandas as pd
import yaml
from src.orchestrator import Orchestrator

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize orchestrator
orchestrator = Orchestrator(config)

# Run pipeline
results = orchestrator.run_pipeline(
    data=df,
    target_column='target',
    task_type='classification',
    domain='healthcare'
)

# Save model
orchestrator.save_final_model('models/my_model.pkl')
```

## 📚 Agent Descriptions

### 1. **EDA Agent**

Performs exploratory data analysis and quality assessment:

- Dataset statistics and distributions
- Missing value analysis
- Correlation analysis
- Outlier detection
- Data quality scoring
- Automated recommendations

### 2. **Feature Engineering Agent**

Handles data preprocessing and feature creation:

- Missing value imputation
- Categorical encoding (one-hot, label)
- Feature scaling (standard, minmax, robust)
- Domain-specific feature creation
- Feature selection (importance, mutual info, RFE)

### 3. **Model Tuning Agent**

Performs model selection and hyperparameter optimization:

- Multiple algorithms (Random Forest, XGBoost, Logistic Regression, etc.)
- Cross-validation with multiple folds
- Hyperparameter tuning (Grid/Random Search)
- Best model selection
- Feature importance extraction

### 4. **Evaluator Agent**

Comprehensive model evaluation and explainability:

- Performance metrics (accuracy, precision, recall, F1, ROC-AUC, RMSE, R², etc.)
- Confusion matrices and classification reports
- SHAP and LIME integration (placeholders)
- Fairness checks
- Performance recommendations

### 5. **Judge Agent**

Makes critical decisions on model deployment:

- Performance threshold checking
- Retraining decision logic
- Performance trend analysis
- Deployment readiness assessment
- Quality gate enforcement

## ⚙️ Configuration

Edit `config.yaml` to customize the pipeline:

```yaml
agents:
  eda:
    enabled: true
    correlation_threshold: 0.7

  model_tuning:
    enabled: true
    algorithms:
      - "random_forest"
      - "xgboost"
    cv_folds: 5
    max_trials: 50

  judge:
    enabled: true
    min_performance_threshold: 0.75
    max_retrain_cycles: 3
```

## 🔌 LLM Integration

The system includes placeholders for LLM-based reasoning at each agent level:

```python
# Enable LLM reasoning in config.yaml
llm:
  provider: "ollama"  # or "openai", "anthropic"
  model: "llama2"
  temperature: 0.7
  reasoning_enabled: true
```

### Integration Options:

1. **Ollama (Local)**:

```python
from langchain.llms import Ollama
llm = Ollama(model="llama2")
response = llm("Analyze this dataset...")
```

2. **OpenAI**:

```python
from langchain.llms import OpenAI
llm = OpenAI(api_key="your-key")
```

3. **Anthropic Claude**:

```python
from langchain.llms import Anthropic
llm = Anthropic(api_key="your-key")
```

## 📊 Examples

### Healthcare Example

```bash
cd examples
python example_healthcare.py
```

Demonstrates diabetes prediction with healthcare-specific features.

### Finance Example

```bash
cd examples
python example_finance.py
```

Demonstrates credit default prediction with financial data.

### Generate Sample Data

```bash
cd data
python sample_data_generator.py
```

Creates sample classification and regression datasets.

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=src
```

## 📈 Output and Results

The pipeline generates comprehensive outputs:

1. **Intermediate Results**: Saved after each agent execution
2. **Final Report**: JSON file with complete pipeline results
3. **Trained Models**: Serialized model files (`.pkl`)
4. **Communication Logs**: Agent message history
5. **Evaluation Reports**: Detailed performance metrics

Example output structure:

```
results/
├── final_report_20240101_120000.json
├── eda_agent_iter1_20240101_120005.json
├── feature_engineering_agent_iter1_20240101_120010.json
└── ...
```

## 🔧 Extending the System

### Adding a New Agent

1. Create agent class inheriting from `BaseAgent`:

```python
from src.core.base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    def execute(self, input_data):
        # Your logic here
        return results
```

2. Register in orchestrator
3. Update configuration file

### Adding New Algorithms

Edit `src/agents/model_tuning_agent.py`:

```python
def _get_model_and_params(self, algorithm, task_type):
    if algorithm == "my_algorithm":
        model = MyModel()
        params = {...}
        return model, params
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH includes project root
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Data Format**: Verify target column name matches dataset
4. **Memory Issues**: Reduce `max_trials` in config for large datasets

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Contact

For questions or issues, please open a GitHub issue.

## 🙏 Acknowledgments

- Built with scikit-learn, XGBoost, and PyDantic
- Inspired by autonomous agent architectures
- Designed for healthcare and finance ML applications

## 🗺️ Roadmap

- [ ] Full SHAP/LIME integration
- [ ] Real-time model monitoring
- [ ] Distributed training support
- [ ] Web UI for pipeline management
- [ ] Auto-ML capabilities
- [ ] Model versioning and registry
- [ ] A/B testing framework
- [ ] Drift detection implementation

---

**Built with ❤️ for Explainable AI in Healthcare and Finance**
