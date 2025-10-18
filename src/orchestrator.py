"""
Orchestrator
Coordinates the workflow between all agents
"""

import pandas as pd
from typing import Any, Dict, List, Optional
import logging
from datetime import datetime
import json
import os

from src.core.communication import CommunicationProtocol, MessageType
from src.agents.eda_agent import EDAAgent
from src.agents.feature_engineering_agent import FeatureEngineeringAgent
from src.agents.model_tuning_agent import ModelTuningAgent
from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.judge_agent import JudgeAgent


class Orchestrator:
    """
    Orchestrator coordinates the entire ML pipeline workflow
    - Sequential execution of agents
    - Message passing between agents
    - State management
    - Results aggregation
    - Retraining loop management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize orchestrator with configuration
        
        Args:
            config: Configuration dictionary loaded from config.yaml
        """
        self.config = config
        self.orchestrator_config = config.get("orchestrator", {})
        self.verbose = self.orchestrator_config.get("verbose", True)
        
        # Setup logging
        self.logger = logging.getLogger("Orchestrator")
        self.logger.setLevel(logging.INFO)
        
        # Initialize communication protocol
        self.communication = CommunicationProtocol(enable_logging=self.verbose)
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Workflow state
        self.workflow_state = {
            "status": "initialized",
            "current_iteration": 0,
            "results": {},
        }
        
        # Setup output directories
        self._setup_directories()
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents"""
        agents = {}
        
        if self.config.get("agents", {}).get("eda", {}).get("enabled", True):
            agents["eda"] = EDAAgent("EDA_Agent", self.config, self.communication)
        
        if self.config.get("agents", {}).get("feature_engineering", {}).get("enabled", True):
            agents["feature_engineering"] = FeatureEngineeringAgent(
                "FeatureEngineering_Agent", self.config, self.communication
            )
        
        if self.config.get("agents", {}).get("model_tuning", {}).get("enabled", True):
            agents["model_tuning"] = ModelTuningAgent(
                "ModelTuning_Agent", self.config, self.communication
            )
        
        if self.config.get("agents", {}).get("evaluator", {}).get("enabled", True):
            agents["evaluator"] = EvaluatorAgent(
                "Evaluator_Agent", self.config, self.communication
            )
        
        if self.config.get("agents", {}).get("judge", {}).get("enabled", True):
            agents["judge"] = JudgeAgent(
                "Judge_Agent", self.config, self.communication
            )
        
        self.logger.info(f"Initialized {len(agents)} agents: {list(agents.keys())}")
        return agents
    
    def _setup_directories(self):
        """Create necessary output directories"""
        dirs = [
            self.config.get("paths", {}).get("data_dir", "./data"),
            self.config.get("paths", {}).get("models_dir", "./models"),
            self.config.get("paths", {}).get("logs_dir", "./logs"),
            self.config.get("paths", {}).get("results_dir", "./results"),
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def run_pipeline(
        self,
        data: pd.DataFrame,
        target_column: str,
        task_type: str = "classification",
        domain: str = "general"
    ) -> Dict[str, Any]:
        """
        Run the complete ML pipeline
        
        Args:
            data: Input dataset (pandas DataFrame)
            target_column: Name of target variable
            task_type: 'classification' or 'regression'
            domain: Domain context ('healthcare', 'finance', 'general')
            
        Returns:
            Dict containing all pipeline results
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting ML Pipeline Execution")
        self.logger.info("=" * 80)
        
        start_time = datetime.now()
        self.workflow_state["status"] = "running"
        
        # Pipeline input
        pipeline_input = {
            "data": data,
            "target_column": target_column,
            "task_type": task_type,
            "domain": domain,
        }
        
        # Execute pipeline stages
        iteration = 1
        max_iterations = self.agents["judge"].max_retrain_cycles + 1
        
        while iteration <= max_iterations:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"ITERATION {iteration}/{max_iterations}")
            self.logger.info(f"{'='*80}\n")
            
            self.workflow_state["current_iteration"] = iteration
            pipeline_input["iteration"] = iteration
            
            # Stage 1: EDA
            eda_results = self._run_stage("eda", pipeline_input)
            
            # Stage 2: Feature Engineering
            fe_input = {
                **pipeline_input,
                "eda_results": eda_results,
            }
            fe_results = self._run_stage("feature_engineering", fe_input)
            
            # Stage 3: Model Tuning
            tuning_input = {
                **pipeline_input,
                "processed_data": fe_results.get("processed_data"),
                "selected_features": fe_results.get("selected_features", []),
            }
            tuning_results = self._run_stage("model_tuning", tuning_input)
            
            # Stage 4: Evaluation
            eval_input = {
                **pipeline_input,
                "processed_data": fe_results.get("processed_data"),
                "trained_model": self.agents["model_tuning"].best_model,
                "selected_features": fe_results.get("selected_features", []),
            }
            eval_results = self._run_stage("evaluator", eval_input)
            
            # Stage 5: Judge
            judge_input = {
                **pipeline_input,
                "evaluation_results": eval_results,
                "model_metadata": tuning_results,
            }
            judge_results = self._run_stage("judge", judge_input)
            
            # Store iteration results
            self.workflow_state["results"][f"iteration_{iteration}"] = {
                "eda": eda_results,
                "feature_engineering": fe_results,
                "model_tuning": tuning_results,
                "evaluation": eval_results,
                "judgment": judge_results,
            }
            
            # Check if approved or needs retrain
            if judge_results["judgment"]["approved"]:
                self.logger.info("\n✓ Model approved by Judge Agent")
                break
            elif judge_results["judgment"]["requires_retrain"]:
                self.logger.info(f"\n→ Model requires retraining (iteration {iteration})")
                iteration += 1
                # In a real scenario, you might adjust hyperparameters here
            else:
                self.logger.info("\n⚠ Model not approved and max iterations reached")
                break
        
        # Pipeline completion
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.workflow_state["status"] = "completed"
        self.workflow_state["duration_seconds"] = duration
        self.workflow_state["final_iteration"] = iteration
        
        # Generate final report
        final_report = self._generate_final_report()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Pipeline Execution Completed")
        self.logger.info(f"Duration: {duration:.2f} seconds")
        self.logger.info(f"Iterations: {iteration}")
        self.logger.info("=" * 80)
        
        return final_report
    
    def _run_stage(self, agent_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single pipeline stage (agent execution)
        
        Args:
            agent_name: Name of agent to execute
            input_data: Input data for agent
            
        Returns:
            Agent results
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found or not enabled")
        
        agent = self.agents[agent_name]
        
        self.logger.info(f"\n{'─'*80}")
        self.logger.info(f"Running: {agent.agent_name}")
        self.logger.info(f"{'─'*80}")
        
        # Send message to agent
        message = self.communication.send_message(
            sender="orchestrator",
            receiver=agent_name,
            message_type=MessageType.REQUEST,
            content=input_data
        )
        
        # Execute agent
        results = agent.execute(input_data)
        
        # Save intermediate results if configured
        if self.orchestrator_config.get("save_intermediate_results", True):
            self._save_intermediate_results(agent_name, results)
        
        return results
    
    def _save_intermediate_results(self, agent_name: str, results: Dict[str, Any]):
        """Save intermediate results to disk"""
        results_dir = self.config.get("paths", {}).get("results_dir", "./results")
        iteration = self.workflow_state["current_iteration"]
        
        # Create filename
        filename = f"{agent_name}_iter{iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(results_dir, filename)
        
        # Save (excluding non-serializable objects)
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                serializable_results[key] = value
            elif isinstance(value, pd.DataFrame):
                serializable_results[key] = f"<DataFrame: {value.shape}>"
            else:
                serializable_results[key] = f"<{type(value).__name__}>"
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        final_iteration = self.workflow_state["final_iteration"]
        final_results = self.workflow_state["results"].get(f"iteration_{final_iteration}", {})
        
        report = {
            "pipeline_info": {
                "project": self.config.get("project", {}).get("name", "Unknown"),
                "version": self.config.get("project", {}).get("version", "1.0.0"),
                "execution_time": self.workflow_state.get("duration_seconds", 0),
                "total_iterations": final_iteration,
                "status": self.workflow_state.get("status"),
            },
            "final_results": {
                "model_approved": final_results.get("judgment", {}).get("judgment", {}).get("approved", False),
                "deployment_ready": final_results.get("judgment", {}).get("deployment_ready", {}).get("ready", False),
                "best_model": final_results.get("model_tuning", {}).get("best_model_name"),
                "performance_score": final_results.get("judgment", {}).get("judgment", {}).get("performance_score"),
                "metrics": final_results.get("evaluation", {}).get("metrics", {}),
            },
            "recommendations": final_results.get("judgment", {}).get("recommendations", []),
            "all_iterations": self.workflow_state["results"],
        }
        
        # Save final report
        results_dir = self.config.get("paths", {}).get("results_dir", "./results")
        report_path = os.path.join(results_dir, f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(report_path, 'w') as f:
            # Create serializable version
            serializable_report = self._make_serializable(report)
            json.dump(serializable_report, f, indent=2)
        
        self.logger.info(f"\nFinal report saved to: {report_path}")
        
        return report
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, pd.DataFrame):
            return f"<DataFrame: {obj.shape}>"
        else:
            return str(obj)
    
    def save_final_model(self, output_path: Optional[str] = None):
        """Save the final trained model"""
        if "model_tuning" not in self.agents:
            raise ValueError("Model tuning agent not available")
        
        if output_path is None:
            models_dir = self.config.get("paths", {}).get("models_dir", "./models")
            output_path = os.path.join(
                models_dir,
                f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            )
        
        self.agents["model_tuning"].save_model(output_path)
        self.logger.info(f"Final model saved to: {output_path}")
    
    def get_message_history(self) -> List[Dict[str, Any]]:
        """Get communication history between agents"""
        return [msg.to_dict() for msg in self.communication.get_message_history()]

