"""
Evaluator Agent
Performs comprehensive model evaluation and explainability analysis
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from src.core.base_agent import BaseAgent


class EvaluatorAgent(BaseAgent):
    """
    Evaluator Agent performs model evaluation and generates explanations
    - Performance metrics (accuracy, precision, recall, F1, ROC-AUC, RMSE, R², etc.)
    - Explainability (SHAP, LIME) - placeholders for integration
    - Model fairness checks
    - Performance visualization suggestions
    """
    
    def __init__(self, agent_name: str, config: Dict[str, Any], communication_protocol):
        super().__init__(agent_name, config, communication_protocol)
        self.eval_config = config.get("agents", {}).get("evaluator", {})
        self.evaluation_results = {}
        
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute model evaluation
        
        Args:
            input_data: Dict containing:
                - processed_data: Full dataset
                - trained_model: Trained model instance
                - target_column: Target variable name
                - task_type: 'classification' or 'regression'
                
        Returns:
            Dict with evaluation metrics and explanations
        """
        self.logger.info("Starting model evaluation...")
        
        # Extract inputs
        df = input_data.get("processed_data")
        model = input_data.get("trained_model")
        target_column = input_data.get("target_column")
        task_type = input_data.get("task_type", "classification")
        
        # Split data for evaluation
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if task_type == "classification" else None
        )
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if task_type == "classification":
            metrics = self._evaluate_classification(y_test, y_pred, model, X_test)
        else:
            metrics = self._evaluate_regression(y_test, y_pred)
        
        # Explainability analysis
        explainability = self._generate_explainability(
            model, X_train, X_test, input_data.get("selected_features", [])
        )
        
        # Model fairness check (basic)
        fairness = self._check_fairness(y_test, y_pred)
        
        # LLM insights on model performance
        if self.llm_enabled:
            llm_insights = self.llm_reason(
                prompt=f"Interpret these model evaluation results for {input_data.get('domain', 'general')} domain",
                context={"metrics": metrics, "task_type": task_type}
            )
            metrics["llm_insights"] = llm_insights
        
        results = {
            "metrics": metrics,
            "explainability": explainability,
            "fairness": fairness,
            "task_type": task_type,
            "test_set_size": len(X_test),
            "recommendations": self._generate_recommendations(metrics, task_type),
        }
        
        self.evaluation_results = results
        self.results = results
        self.save_state()
        
        return results
    
    def _evaluate_classification(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model: Any,
        X_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate classification metrics"""
        
        # Basic metrics
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        }
        
        # ROC-AUC (for binary or probability predictions)
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            else:  # Multi-class
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)
                    metrics["roc_auc"] = float(roc_auc_score(
                        y_true, y_proba, multi_class='ovr', average='weighted'
                    ))
        except Exception as e:
            self.logger.warning(f"Could not calculate ROC-AUC: {e}")
            metrics["roc_auc"] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics["classification_report"] = report
        
        return metrics
    
    def _evaluate_regression(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate regression metrics"""
        
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2_score": float(r2_score(y_true, y_pred)),
        }
        
        # MAPE (Mean Absolute Percentage Error)
        try:
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics["mape"] = float(mape)
        except:
            metrics["mape"] = None
        
        # Residual statistics
        residuals = y_true - y_pred
        metrics["residual_stats"] = {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "min": float(residuals.min()),
            "max": float(residuals.max()),
        }
        
        return metrics
    
    def _generate_explainability(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Generate model explainability analysis
        Placeholder for SHAP and LIME integration
        """
        explainability = {
            "methods_available": [],
            "feature_importance": {},
            "shap_values": None,
            "lime_explanations": None,
        }
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            if feature_names:
                feature_importance = dict(zip(feature_names, importances))
            else:
                feature_importance = dict(zip(
                    [f"feature_{i}" for i in range(len(importances))],
                    importances
                ))
            
            # Sort by importance
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            )
            explainability["feature_importance"] = {
                k: float(v) for k, v in feature_importance.items()
            }
            explainability["methods_available"].append("feature_importance")
        
        # SHAP integration
        if "shap" in self.eval_config.get("explainability_methods", []):
            if not SHAP_AVAILABLE:
                self.logger.warning("SHAP library not installed. Install with: pip install shap")
                explainability["shap_values"] = "SHAP not available - install shap library"
            else:
                try:
                    self.logger.info("Computing SHAP values...")
                    
                    # Sample data if too large (SHAP can be slow)
                    sample_size = min(100, len(X_test))
                    X_sample = X_test.iloc[:sample_size]
                    
                    # Choose appropriate explainer based on model type
                    if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
                        # Tree-based models (RF, XGBoost, Decision Tree)
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_sample)
                    else:
                        # Other models (use KernelExplainer with background data)
                        background = shap.sample(X_train, min(100, len(X_train)))
                        explainer = shap.KernelExplainer(model.predict, background)
                        shap_values = explainer.shap_values(X_sample)
                    
                    # Handle multi-class output
                    if isinstance(shap_values, list) and len(shap_values) > 0:
                        # Multi-class: use class 1 (positive class for binary)
                        shap_values_array = np.array(shap_values[1] if len(shap_values) > 1 else shap_values[0])
                    else:
                        shap_values_array = np.array(shap_values)
                    
                    # Calculate mean absolute SHAP values for feature importance
                    if hasattr(shap_values_array, 'shape'):
                        if len(shap_values_array.shape) == 3:
                            # (samples, features, classes) -> average over samples and classes
                            mean_shap = np.abs(shap_values_array).mean(axis=(0, 2))
                        elif len(shap_values_array.shape) == 2:
                            # (samples, features) -> average over samples
                            mean_shap = np.abs(shap_values_array).mean(axis=0)
                        elif len(shap_values_array.shape) == 1:
                            # (features,) -> use directly
                            mean_shap = np.abs(shap_values_array)
                        else:
                            raise ValueError(f"Unexpected SHAP values shape: {shap_values_array.shape}")
                        
                        if feature_names:
                            shap_importance = dict(zip(feature_names, mean_shap))
                        else:
                            shap_importance = dict(zip(
                                [f"feature_{i}" for i in range(len(mean_shap))],
                                mean_shap
                            ))
                        
                        # Sort and get top features
                        shap_importance = dict(
                            sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                        )
                        
                        explainability["shap_feature_importance"] = {
                            k: float(v) for k, v in shap_importance.items()
                        }
                        explainability["shap_summary"] = {
                            "samples_analyzed": sample_size,
                            "top_feature": max(shap_importance, key=shap_importance.get),
                            "analysis_complete": True
                        }
                        explainability["methods_available"].append("shap")
                        self.logger.info(f"SHAP analysis completed for {sample_size} samples")
                    
                except Exception as e:
                    self.logger.warning(f"SHAP analysis failed: {e}")
                    explainability["shap_values"] = f"SHAP analysis failed: {str(e)}"
        
        # LIME integration
        if "lime" in self.eval_config.get("explainability_methods", []):
            if not LIME_AVAILABLE:
                self.logger.warning("LIME library not installed. Install with: pip install lime")
                explainability["lime_explanations"] = "LIME not available - install lime library"
            else:
                try:
                    self.logger.info("Computing LIME explanations...")
                    
                    # Determine mode based on model type
                    if hasattr(model, 'predict_proba'):
                        mode = 'classification'
                        predict_fn = model.predict_proba
                    else:
                        mode = 'regression'
                        predict_fn = model.predict
                    
                    # Create LIME explainer
                    explainer = lime.lime_tabular.LimeTabularExplainer(
                        X_train.values,
                        feature_names=feature_names if feature_names else [f"feature_{i}" for i in range(X_train.shape[1])],
                        mode=mode,
                        random_state=42
                    )
                    
                    # Explain multiple instances (first 3 as examples)
                    num_samples = min(3, len(X_test))
                    lime_explanations_list = []
                    
                    for idx in range(num_samples):
                        explanation = explainer.explain_instance(
                            X_test.iloc[idx].values,
                            predict_fn,
                            num_features=10
                        )
                        
                        # Extract feature contributions
                        exp_list = explanation.as_list()
                        lime_explanations_list.append({
                            "instance_index": int(idx),
                            "feature_contributions": [
                                {"feature": feat, "contribution": float(contrib)}
                                for feat, contrib in exp_list
                            ]
                        })
                    
                    # Aggregate LIME feature importance across samples
                    feature_importance_lime = {}
                    for exp in lime_explanations_list:
                        for feat_contrib in exp["feature_contributions"]:
                            feat_name = feat_contrib["feature"].split()[0]  # Extract feature name
                            contrib = abs(feat_contrib["contribution"])
                            if feat_name in feature_importance_lime:
                                feature_importance_lime[feat_name] += contrib
                            else:
                                feature_importance_lime[feat_name] = contrib
                    
                    # Average and sort
                    for key in feature_importance_lime:
                        feature_importance_lime[key] /= num_samples
                    
                    feature_importance_lime = dict(
                        sorted(feature_importance_lime.items(), key=lambda x: x[1], reverse=True)[:10]
                    )
                    
                    explainability["lime_feature_importance"] = feature_importance_lime
                    explainability["lime_sample_explanations"] = lime_explanations_list
                    explainability["lime_summary"] = {
                        "samples_explained": num_samples,
                        "top_feature": max(feature_importance_lime, key=feature_importance_lime.get) if feature_importance_lime else None,
                        "analysis_complete": True
                    }
                    explainability["methods_available"].append("lime")
                    self.logger.info(f"LIME analysis completed for {num_samples} samples")
                    
                except Exception as e:
                    self.logger.warning(f"LIME analysis failed: {e}")
                    explainability["lime_explanations"] = f"LIME analysis failed: {str(e)}"
        
        return explainability
    
    def _check_fairness(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Basic fairness check
        In production, this should check for bias across sensitive attributes
        """
        fairness = {
            "checked": True,
            "metrics": {
                "prediction_distribution": {},
            },
            "notes": "Basic fairness check. Implement demographic parity and equalized odds for production."
        }
        
        # Check prediction distribution
        unique, counts = np.unique(y_pred, return_counts=True)
        total = len(y_pred)
        fairness["metrics"]["prediction_distribution"] = {
            str(label): float(count / total) for label, count in zip(unique, counts)
        }
        
        return fairness
    
    def _generate_recommendations(
        self,
        metrics: Dict[str, Any],
        task_type: str
    ) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        if task_type == "classification":
            accuracy = metrics.get("accuracy", 0)
            f1 = metrics.get("f1_score", 0)
            
            if accuracy < 0.7:
                recommendations.append(
                    "Low accuracy detected. Consider: more data, feature engineering, or different algorithms."
                )
            
            if f1 < 0.6:
                recommendations.append(
                    "Low F1 score suggests class imbalance. Consider resampling techniques (SMOTE, undersampling)."
                )
            
            if accuracy > 0.95:
                recommendations.append(
                    "Very high accuracy - verify for data leakage or overfitting."
                )
        
        else:  # regression
            r2 = metrics.get("r2_score", 0)
            
            if r2 < 0.5:
                recommendations.append(
                    "Low R² score. Consider: polynomial features, different algorithms, or more relevant features."
                )
            
            if r2 > 0.95:
                recommendations.append(
                    "Very high R² - verify for data leakage or overfitting."
                )
        
        if not recommendations:
            recommendations.append("Model performance is acceptable. Proceed to deployment considerations.")
        
        return recommendations
    
    def generate_report(self, output_path: str):
        """Generate evaluation report"""
        import json
        
        with open(output_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        self.logger.info(f"Evaluation report saved to {output_path}")

