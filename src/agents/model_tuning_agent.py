"""
Model Tuning Agent
Performs model selection, hyperparameter tuning, and training
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import make_scorer, accuracy_score, f1_score, mean_squared_error, r2_score
import joblib
import json

from src.core.base_agent import BaseAgent


class ModelTuningAgent(BaseAgent):
    """
    Model Tuning Agent handles model selection and hyperparameter optimization
    - Multiple algorithm support (RF, XGBoost, Logistic Regression, etc.)
    - Hyperparameter tuning with cross-validation
    - Model comparison and selection
    - Training with best parameters
    """
    
    def __init__(self, agent_name: str, config: Dict[str, Any], communication_protocol):
        super().__init__(agent_name, config, communication_protocol)
        self.tuning_config = config.get("agents", {}).get("model_tuning", {})
        self.trained_models = {}
        self.best_model = None
        self.best_params = {}
        
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute model tuning pipeline
        
        Args:
            input_data: Dict containing processed data, target_column, and task_type
            
        Returns:
            Dict with trained models and tuning results
        """
        self.logger.info("Starting model tuning...")
        
        # Load data
        df = input_data.get("processed_data")
        target_column = input_data.get("target_column")
        task_type = input_data.get("task_type", "classification")
        
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Get algorithms to try - use LLM suggestions if available
        suggested_algorithms = input_data.get("suggested_algorithms", [])
        if suggested_algorithms:
            self.logger.info(f"🧠 Using LLM-suggested algorithms: {suggested_algorithms}")
            algorithms = suggested_algorithms
        else:
            algorithms = self.tuning_config.get("algorithms", ["random_forest", "xgboost"])
            self.logger.info(f"Using default algorithms: {algorithms}")
        
        # Train and tune models
        model_results = []
        for algo in algorithms:
            self.logger.info(f"Training {algo}...")
            
            model, params, cv_score = self._train_model(
                X, y, algo, task_type
            )
            
            self.trained_models[algo] = model
            
            model_results.append({
                "algorithm": algo,
                "cv_score": float(cv_score),
                "best_params": params,
            })
        
        # Select best model
        best_result = max(model_results, key=lambda x: x["cv_score"])
        self.best_model = self.trained_models[best_result["algorithm"]]
        self.best_params = best_result["best_params"]
        
        # LLM reasoning for model selection (if enabled)
        if self.llm_enabled:
            llm_recommendation = self.llm_reason(
                prompt=f"Which model is best for {input_data.get('domain', 'general')} domain and why?",
                context={"model_results": model_results, "task_type": task_type}
            )
            best_result["llm_recommendation"] = llm_recommendation
        
        results = {
            "all_models": model_results,
            "best_model_name": best_result["algorithm"],
            "best_cv_score": best_result["cv_score"],
            "best_params": self.best_params,
            "num_models_trained": len(algorithms),
            "task_type": task_type,
        }
        
        self.results = results
        self.save_state()
        
        return results
    
    def _train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        algorithm: str,
        task_type: str
    ) -> Tuple[Any, Dict, float]:
        """
        Train and tune a single model
        
        Returns:
            Tuple of (trained_model, best_params, cv_score)
        """
        # Get model and parameter grid
        model, param_grid = self._get_model_and_params(algorithm, task_type)
        
        # Setup cross-validation
        cv_folds = self.tuning_config.get("cv_folds", 5)
        max_trials = self.tuning_config.get("max_trials", 50)
        
        # Choose search strategy
        if len(param_grid) > 0 and max_trials < 100:
            # Randomized search for efficiency
            search = RandomizedSearchCV(
                model,
                param_distributions=param_grid,
                n_iter=min(max_trials, 20),
                cv=cv_folds,
                scoring=self._get_scoring_metric(task_type),
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
        else:
            # Grid search for exhaustive search
            search = GridSearchCV(
                model,
                param_grid=param_grid if param_grid else {},
                cv=cv_folds,
                scoring=self._get_scoring_metric(task_type),
                n_jobs=-1,
                verbose=0
            )
        
        # Fit model
        search.fit(X, y)
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def _get_model_and_params(
        self,
        algorithm: str,
        task_type: str
    ) -> Tuple[Any, Dict]:
        """Get model instance and parameter grid for algorithm"""
        
        if task_type == "classification":
            if algorithm == "random_forest":
                model = RandomForestClassifier(random_state=42)
                params = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                }
            elif algorithm == "xgboost":
                model = XGBClassifier(random_state=42, eval_metric='logloss')
                params = {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 1.0],
                }
            elif algorithm == "logistic_regression":
                model = LogisticRegression(random_state=42, max_iter=1000)
                params = {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear'],
                }
            elif algorithm == "svm":
                model = SVC(random_state=42)
                params = {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto'],
                }
            elif algorithm == "decision_tree":
                model = DecisionTreeClassifier(random_state=42)
                params = {
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy'],
                }
            elif algorithm == "neural_network" or algorithm == "mlp":
                model = MLPClassifier(random_state=42, max_iter=1000)
                params = {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive'],
                }
            else:
                raise ValueError(f"Unknown classification algorithm: {algorithm}")
                
        else:  # regression
            if algorithm == "random_forest":
                model = RandomForestRegressor(random_state=42)
                params = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                }
            elif algorithm == "xgboost":
                model = XGBRegressor(random_state=42)
                params = {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                }
            elif algorithm == "linear_regression":
                model = Ridge(random_state=42)
                params = {
                    'alpha': [0.1, 1.0, 10.0],
                }
            elif algorithm == "ridge":
                model = Ridge(random_state=42)
                params = {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                }
            elif algorithm == "svm":
                model = SVR()
                params = {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto'],
                }
            elif algorithm == "decision_tree":
                model = DecisionTreeRegressor(random_state=42)
                params = {
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                }
            elif algorithm == "neural_network" or algorithm == "mlp":
                model = MLPRegressor(random_state=42, max_iter=1000)
                params = {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive'],
                }
            else:
                raise ValueError(f"Unknown regression algorithm: {algorithm}")
        
        return model, params
    
    def _get_scoring_metric(self, task_type: str) -> str:
        """Get appropriate scoring metric for task type"""
        if task_type == "classification":
            return "f1_weighted"  # Use weighted F1 for multi-class
        else:
            return "r2"  # Use R² for regression
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using best model"""
        if self.best_model is None:
            raise ValueError("No model trained yet. Call execute() first.")
        
        return self.best_model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (classification only)"""
        if self.best_model is None:
            raise ValueError("No model trained yet. Call execute() first.")
        
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)
        else:
            raise ValueError("Model does not support probability predictions")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from best model"""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        if hasattr(self.best_model, 'feature_importances_'):
            # Tree-based models
            importances = self.best_model.feature_importances_
            return dict(zip(
                range(len(importances)),
                importances.tolist()
            ))
        elif hasattr(self.best_model, 'coef_'):
            # Linear models
            coefs = np.abs(self.best_model.coef_)
            if len(coefs.shape) > 1:
                coefs = coefs.mean(axis=0)
            return dict(zip(
                range(len(coefs)),
                coefs.tolist()
            ))
        else:
            return {}
    
    def save_model(self, output_path: str):
        """Save best model to disk"""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        joblib.dump(self.best_model, output_path)
        self.logger.info(f"Model saved to {output_path}")
        
        # Save metadata
        metadata = {
            "best_params": self.best_params,
            "results": self.results,
        }
        with open(output_path.replace('.pkl', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

