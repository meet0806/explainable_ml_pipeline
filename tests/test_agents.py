"""
Unit tests for agents
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

import sys
sys.path.append('..')

from src.core.communication import CommunicationProtocol, MessageType, AgentMessage
from src.agents.eda_agent import EDAAgent
from src.agents.feature_engineering_agent import FeatureEngineeringAgent
from src.agents.model_tuning_agent import ModelTuningAgent
from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.judge_agent import JudgeAgent


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "agents": {
            "eda": {
                "enabled": True,
                "correlation_threshold": 0.7,
                "missing_value_threshold": 0.3,
            },
            "feature_engineering": {
                "enabled": True,
                "scaling_method": "standard",
                "encoding_method": "onehot",
                "feature_selection_method": "importance",
            },
            "model_tuning": {
                "enabled": True,
                "algorithms": ["random_forest"],
                "cv_folds": 3,
                "max_trials": 10,
            },
            "evaluator": {
                "enabled": True,
                "metrics": {
                    "classification": ["accuracy", "f1"],
                },
            },
            "judge": {
                "enabled": True,
                "min_performance_threshold": 0.6,
                "max_retrain_cycles": 2,
            },
        },
        "llm": {
            "reasoning_enabled": False,
        },
    }


@pytest.fixture
def sample_data():
    """Generate sample dataset"""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    
    return df


@pytest.fixture
def communication_protocol():
    """Communication protocol for testing"""
    return CommunicationProtocol(enable_logging=False)


def test_communication_protocol():
    """Test communication protocol"""
    protocol = CommunicationProtocol(enable_logging=True)
    
    message = protocol.send_message(
        sender="agent1",
        receiver="agent2",
        message_type=MessageType.REQUEST,
        content={"test": "data"}
    )
    
    assert isinstance(message, AgentMessage)
    assert message.sender == "agent1"
    assert message.receiver == "agent2"
    assert message.message_type == MessageType.REQUEST
    assert len(protocol.get_message_history()) == 1


def test_eda_agent(sample_config, sample_data, communication_protocol):
    """Test EDA Agent"""
    agent = EDAAgent("EDA_Test", sample_config, communication_protocol)
    
    input_data = {
        "data": sample_data,
        "target_column": "target",
    }
    
    results = agent.execute(input_data)
    
    assert "dataset_info" in results
    assert "statistical_summary" in results
    assert "missing_values" in results
    assert "correlation_analysis" in results
    assert "data_quality_score" in results
    assert results["dataset_info"]["num_rows"] == 200


def test_feature_engineering_agent(sample_config, sample_data, communication_protocol):
    """Test Feature Engineering Agent"""
    agent = FeatureEngineeringAgent("FE_Test", sample_config, communication_protocol)
    
    input_data = {
        "data": sample_data,
        "target_column": "target",
        "task_type": "classification",
    }
    
    results = agent.execute(input_data)
    
    assert "processed_data" in results
    assert "selected_features" in results
    assert "num_features_original" in results
    assert "num_features_final" in results
    assert isinstance(results["processed_data"], pd.DataFrame)


def test_model_tuning_agent(sample_config, sample_data, communication_protocol):
    """Test Model Tuning Agent"""
    agent = ModelTuningAgent("MT_Test", sample_config, communication_protocol)
    
    # Prepare data
    input_data = {
        "processed_data": sample_data,
        "target_column": "target",
        "task_type": "classification",
    }
    
    results = agent.execute(input_data)
    
    assert "all_models" in results
    assert "best_model_name" in results
    assert "best_cv_score" in results
    assert agent.best_model is not None


def test_evaluator_agent(sample_config, sample_data, communication_protocol):
    """Test Evaluator Agent"""
    # First train a model
    tuning_agent = ModelTuningAgent("MT_Test", sample_config, communication_protocol)
    tuning_results = tuning_agent.execute({
        "processed_data": sample_data,
        "target_column": "target",
        "task_type": "classification",
    })
    
    # Now evaluate
    eval_agent = EvaluatorAgent("Eval_Test", sample_config, communication_protocol)
    
    input_data = {
        "processed_data": sample_data,
        "trained_model": tuning_agent.best_model,
        "target_column": "target",
        "task_type": "classification",
    }
    
    results = eval_agent.execute(input_data)
    
    assert "metrics" in results
    assert "explainability" in results
    assert "recommendations" in results
    assert "accuracy" in results["metrics"]


def test_judge_agent(sample_config, communication_protocol):
    """Test Judge Agent"""
    agent = JudgeAgent("Judge_Test", sample_config, communication_protocol)
    
    # Mock evaluation results
    eval_results = {
        "metrics": {
            "accuracy": 0.85,
            "f1_score": 0.83,
        },
    }
    
    input_data = {
        "evaluation_results": eval_results,
        "task_type": "classification",
        "iteration": 1,
    }
    
    results = agent.execute(input_data)
    
    assert "judgment" in results
    assert "deployment_ready" in results
    assert "recommendations" in results
    assert results["judgment"]["approved"] == True  # Should pass with 0.85 accuracy


def test_end_to_end_pipeline(sample_config, sample_data, communication_protocol):
    """Test end-to-end pipeline with all agents"""
    
    # Initialize all agents
    eda = EDAAgent("EDA", sample_config, communication_protocol)
    fe = FeatureEngineeringAgent("FE", sample_config, communication_protocol)
    mt = ModelTuningAgent("MT", sample_config, communication_protocol)
    ev = EvaluatorAgent("EV", sample_config, communication_protocol)
    jd = JudgeAgent("JD", sample_config, communication_protocol)
    
    # Run pipeline
    eda_results = eda.execute({
        "data": sample_data,
        "target_column": "target",
    })
    
    fe_results = fe.execute({
        "data": sample_data,
        "target_column": "target",
        "task_type": "classification",
        "eda_results": eda_results,
    })
    
    mt_results = mt.execute({
        "processed_data": fe_results["processed_data"],
        "target_column": "target",
        "task_type": "classification",
    })
    
    ev_results = ev.execute({
        "processed_data": fe_results["processed_data"],
        "trained_model": mt.best_model,
        "target_column": "target",
        "task_type": "classification",
    })
    
    jd_results = jd.execute({
        "evaluation_results": ev_results,
        "task_type": "classification",
        "iteration": 1,
    })
    
    # Verify pipeline completed
    assert jd_results["judgment"]["approved"] in [True, False]
    assert len(communication_protocol.get_message_history()) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

