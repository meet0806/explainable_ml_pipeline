"""
Judge Agent
Makes decisions on model retraining and deployment readiness
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.core.base_agent import BaseAgent
from src.core.communication import MessageType


class JudgeAgent(BaseAgent):
    """
    Judge Agent evaluates model performance and makes critical decisions
    - Determines if model meets performance thresholds
    - Decides if retraining is needed
    - Tracks model performance over time
    - Provides deployment recommendations
    - Monitors for model drift (placeholder)
    """
    
    def __init__(self, agent_name: str, config: Dict[str, Any], communication_protocol):
        super().__init__(agent_name, config, communication_protocol)
        self.judge_config = config.get("agents", {}).get("judge", {})
        self.performance_history: List[Dict[str, Any]] = []
        self.retrain_count = 0
        self.max_retrain_cycles = self.judge_config.get("max_retrain_cycles", 3)
        
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute judgment on model performance
        
        Args:
            input_data: Dict containing:
                - evaluation_results: Results from EvaluatorAgent
                - model_metadata: Information about trained model
                - task_type: Classification or regression
                - iteration: Current training iteration
                
        Returns:
            Dict with judgment decision and recommendations
        """
        self.logger.info("Judge Agent evaluating model performance...")
        
        # Extract evaluation metrics
        evaluation_results = input_data.get("evaluation_results", {})
        metrics = evaluation_results.get("metrics", {})
        task_type = input_data.get("task_type", "classification")
        current_iteration = input_data.get("iteration", 1)
        
        # Store performance history
        self.performance_history.append({
            "iteration": current_iteration,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        })
        
        # Make judgment
        judgment = self._make_judgment(metrics, task_type, current_iteration)
        
        # Check for performance degradation
        performance_trend = self._analyze_performance_trend()
        
        # Deployment readiness assessment
        deployment_ready = self._assess_deployment_readiness(
            judgment, evaluation_results
        )
        
        # LLM-based decision support (if enabled)
        llm_model_recommendations = []
        if self.llm_enabled:
            # Get deployment decision
            llm_decision = self.llm_reason(
                prompt="Should this model be deployed or retrained? Provide reasoning.",
                context={
                    "metrics": metrics,
                    "judgment": judgment,
                    "deployment_ready": deployment_ready,
                    "domain": input_data.get("domain", "general"),
                }
            )
            judgment["llm_decision_support"] = llm_decision
            
            # Get model recommendations for next iteration
            if judgment["requires_retrain"]:
                current_model = input_data.get("model_metadata", {}).get("model_name", "unknown")
                all_models_tried = input_data.get("all_models_tried", [])
                
                llm_model_suggestion = self.llm_reason(
                    prompt=f"""Based on the current model performance, suggest which ML algorithms to try in the next iteration.
                    
Current model: {current_model}
Task type: {task_type}
Performance: {metrics}
Models already tried: {all_models_tried}

Available algorithms:
- random_forest: Good for complex non-linear relationships, handles missing data well
- xgboost: Excellent for structured data, handles imbalanced datasets, gradient boosting
- logistic_regression: Fast, interpretable, works well for linearly separable data
- svm: Good for small datasets with clear margins, kernel methods
- decision_tree: Simple, interpretable, fast training, prone to overfitting
- neural_network/mlp: Powerful for complex patterns, requires more data, slower training
- linear_regression (regression only): Simple, interpretable baseline
- ridge (regression only): Linear with regularization

Respond with ONLY a comma-separated list of 2-3 algorithm names from the list above. Example: xgboost,random_forest,neural_network""",
                    context={
                        "current_model": current_model,
                        "metrics": metrics,
                        "task_type": task_type,
                        "all_models_tried": all_models_tried
                    }
                )
                
                # Parse LLM response to extract model names
                llm_model_recommendations = self._parse_model_recommendations(
                    llm_model_suggestion, task_type
                )
                self.logger.info(f"LLM recommends models for next iteration: {llm_model_recommendations}")
        
        results = {
            "judgment": judgment,
            "deployment_ready": deployment_ready,
            "performance_trend": performance_trend,
            "retrain_count": self.retrain_count,
            "max_retrain_cycles": self.max_retrain_cycles,
            "recommended_models": llm_model_recommendations,  # NEW: LLM model suggestions
            "recommendations": self._generate_recommendations(
                judgment, deployment_ready, task_type
            ),
        }
        
        self.results = results
        self.save_state()
        
        # Send decision message to orchestrator
        self.send_message(
            receiver="orchestrator",
            content=results,
            message_type=MessageType.DECISION
        )
        
        return results
    
    def _make_judgment(
        self,
        metrics: Dict[str, Any],
        task_type: str,
        iteration: int
    ) -> Dict[str, Any]:
        """
        Make judgment on model performance
        
        Returns judgment with decision and reasoning
        """
        min_threshold = self.judge_config.get("min_performance_threshold", 0.75)
        
        judgment = {
            "approved": False,
            "performance_score": 0.0,
            "meets_threshold": False,
            "reason": "",
            "requires_retrain": False,
        }
        
        # Calculate performance score based on task type
        if task_type == "classification":
            # Use F1 score as primary metric
            f1_score = metrics.get("f1_score", 0)
            accuracy = metrics.get("accuracy", 0)
            
            # Weighted performance score
            performance_score = 0.6 * f1_score + 0.4 * accuracy
            judgment["performance_score"] = float(performance_score)
            judgment["primary_metric"] = "f1_score"
            judgment["primary_metric_value"] = float(f1_score)
            
        else:  # regression
            # Use R² score as primary metric
            r2_score = metrics.get("r2_score", 0)
            
            # For regression, R² can be negative, so normalize
            performance_score = max(0, r2_score)
            judgment["performance_score"] = float(performance_score)
            judgment["primary_metric"] = "r2_score"
            judgment["primary_metric_value"] = float(r2_score)
        
        # Check if meets threshold
        meets_threshold = performance_score >= min_threshold
        judgment["meets_threshold"] = meets_threshold
        
        # Make decision
        if meets_threshold:
            judgment["approved"] = True
            judgment["reason"] = f"Model performance ({performance_score:.3f}) exceeds minimum threshold ({min_threshold})."
        else:
            judgment["approved"] = False
            judgment["reason"] = f"Model performance ({performance_score:.3f}) below minimum threshold ({min_threshold})."
            
            # Determine if retrain is possible
            if self.retrain_count < self.max_retrain_cycles:
                judgment["requires_retrain"] = True
                judgment["reason"] += f" Retraining recommended (cycle {self.retrain_count + 1}/{self.max_retrain_cycles})."
                self.retrain_count += 1
            else:
                judgment["requires_retrain"] = False
                judgment["reason"] += f" Max retrain cycles ({self.max_retrain_cycles}) reached. Manual intervention required."
        
        return judgment
    
    def _analyze_performance_trend(self) -> Dict[str, Any]:
        """Analyze performance trend across iterations"""
        
        if len(self.performance_history) < 2:
            return {
                "trend": "insufficient_data",
                "message": "Need at least 2 iterations to analyze trend",
            }
        
        # Get performance scores from history
        scores = [
            entry["metrics"].get("f1_score") or entry["metrics"].get("r2_score", 0)
            for entry in self.performance_history
        ]
        
        # Calculate trend
        if len(scores) >= 2:
            recent_change = scores[-1] - scores[-2]
            decline_threshold = self.judge_config.get("performance_decline_threshold", 0.05)
            
            if recent_change > decline_threshold:
                trend = "improving"
                message = f"Performance is improving (+{recent_change:.3f})"
            elif recent_change < -decline_threshold:
                trend = "declining"
                message = f"Performance is declining ({recent_change:.3f}). Monitor for drift."
            else:
                trend = "stable"
                message = "Performance is stable"
            
            return {
                "trend": trend,
                "message": message,
                "recent_change": float(recent_change),
                "history_length": len(scores),
            }
        
        return {
            "trend": "unknown",
            "message": "Unable to determine trend",
        }
    
    def _assess_deployment_readiness(
        self,
        judgment: Dict[str, Any],
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess if model is ready for deployment"""
        
        readiness = {
            "ready": False,
            "confidence": 0.0,
            "blockers": [],
            "warnings": [],
        }
        
        # Check judgment approval
        if not judgment.get("approved", False):
            readiness["blockers"].append("Model performance below threshold")
        
        # Check for explainability
        explainability = evaluation_results.get("explainability", {})
        if not explainability.get("methods_available"):
            readiness["warnings"].append("Limited explainability available")
        
        # Check fairness
        fairness = evaluation_results.get("fairness", {})
        if not fairness.get("checked"):
            readiness["warnings"].append("Fairness not thoroughly checked")
        
        # Check recommendations from evaluator
        eval_recommendations = evaluation_results.get("recommendations", [])
        for rec in eval_recommendations:
            if "leakage" in rec.lower() or "overfitting" in rec.lower():
                readiness["blockers"].append(f"Critical issue: {rec}")
        
        # Calculate confidence
        confidence_factors = []
        
        if judgment.get("approved"):
            confidence_factors.append(0.4)  # Base confidence if approved
        
        performance_score = judgment.get("performance_score", 0)
        confidence_factors.append(min(performance_score, 0.4))  # Up to 0.4 from performance
        
        if not readiness["blockers"]:
            confidence_factors.append(0.2)  # Bonus if no blockers
        
        readiness["confidence"] = sum(confidence_factors)
        
        # Final decision
        readiness["ready"] = (
            judgment.get("approved", False) and
            len(readiness["blockers"]) == 0 and
            readiness["confidence"] >= 0.7
        )
        
        return readiness
    
    def _generate_recommendations(
        self,
        judgment: Dict[str, Any],
        deployment_ready: Dict[str, Any],
        task_type: str
    ) -> List[str]:
        """Generate recommendations based on judgment"""
        recommendations = []
        
        if deployment_ready.get("ready"):
            recommendations.append("✓ Model is ready for deployment")
            recommendations.append("- Conduct final UAT (User Acceptance Testing)")
            recommendations.append("- Set up monitoring for production")
            recommendations.append("- Prepare rollback plan")
        else:
            if judgment.get("requires_retrain"):
                recommendations.append("→ Retrain model with adjusted parameters")
                recommendations.append("- Consider collecting more training data")
                recommendations.append("- Try different algorithms or ensembles")
                recommendations.append("- Review feature engineering pipeline")
            else:
                recommendations.append("⚠ Model performance insufficient")
                recommendations.append("- Review data quality and collection process")
                recommendations.append("- Consult domain experts for feature suggestions")
                recommendations.append("- Consider if problem is solvable with current data")
        
        # Add blocker-specific recommendations
        for blocker in deployment_ready.get("blockers", []):
            if "threshold" in blocker.lower():
                recommendations.append(f"- Address: {blocker}")
        
        return recommendations
    
    def reset_retrain_counter(self):
        """Reset retrain counter for new model development cycle"""
        self.retrain_count = 0
        self.logger.info("Retrain counter reset")
    
    def get_performance_history(self) -> List[Dict[str, Any]]:
        """Get full performance history"""
        return self.performance_history
    
    def _parse_model_recommendations(self, llm_response: str, task_type: str) -> List[str]:
        """
        Parse LLM response to extract model recommendations
        
        Args:
            llm_response: Raw LLM response text
            task_type: 'classification' or 'regression'
            
        Returns:
            List of valid model names
        """
        # Valid models by task type
        valid_models = {
            "classification": [
                "random_forest", "xgboost", "logistic_regression", "svm", 
                "decision_tree", "neural_network", "mlp"
            ],
            "regression": [
                "random_forest", "xgboost", "linear_regression", "ridge", 
                "svm", "decision_tree", "neural_network", "mlp"
            ]
        }
        
        valid_for_task = valid_models.get(task_type, [])
        
        # Parse response - look for model names
        recommendations = []
        response_lower = llm_response.lower().strip()
        
        # Split by common delimiters
        parts = response_lower.replace('\n', ',').replace(';', ',').split(',')
        
        for part in parts:
            part = part.strip()
            # Check if part contains any valid model name
            for model in valid_for_task:
                if model in part:
                    if model not in recommendations:
                        recommendations.append(model)
        
        # If no valid models found, return default
        if not recommendations:
            self.logger.warning(f"Could not parse model recommendations from LLM. Using defaults.")
            recommendations = valid_for_task[:2]  # Use first 2 models as default
        
        return recommendations[:3]  # Limit to 3 models

