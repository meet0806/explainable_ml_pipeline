"""
Feature Engineering Agent
Handles feature transformation, scaling, encoding, and selection
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

from src.core.base_agent import BaseAgent


class FeatureEngineeringAgent(BaseAgent):
    """
    Feature Engineering Agent performs data preprocessing and feature creation
    - Feature scaling and normalization
    - Categorical encoding
    - Feature selection
    - Feature creation (domain-specific)
    - Missing value imputation
    """
    
    def __init__(self, agent_name: str, config: Dict[str, Any], communication_protocol):
        super().__init__(agent_name, config, communication_protocol)
        self.fe_config = config.get("agents", {}).get("feature_engineering", {})
        self.scalers = {}
        self.encoders = {}
        self.selected_features = []
        
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute feature engineering pipeline
        
        Args:
            input_data: Dict containing 'data', 'target_column', and optional 'eda_results'
            
        Returns:
            Dict with processed data and feature engineering metadata
        """
        self.logger.info("Starting feature engineering...")
        
        # Load data
        df = self._load_data(input_data)
        target_column = input_data.get("target_column")
        task_type = input_data.get("task_type", "classification")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Check for and handle missing values in target
        if y.isnull().any():
            self.logger.warning(f"Target column '{target_column}' has {y.isnull().sum()} missing values. Dropping rows with missing targets.")
            # Drop rows where target is missing
            valid_idx = ~y.isnull()
            X = X[valid_idx]
            y = y[valid_idx]
        
        # Encode target if it's categorical (for classification tasks)
        target_encoder = None
        if task_type == "classification" and y.dtype == 'object':
            self.logger.info(f"Encoding categorical target column '{target_column}' with values: {y.unique()}")
            target_encoder = LabelEncoder()
            y = pd.Series(target_encoder.fit_transform(y), index=y.index, name=target_column)
            self.encoders[f"target_{target_column}"] = target_encoder
            self.logger.info(f"Target encoded to: {y.unique()}")
        
        # Feature engineering pipeline
        X_processed = X.copy()
        
        # 1. Handle missing values
        X_processed = self._handle_missing_values(X_processed)
        
        # 2. Encode categorical features
        X_processed = self._encode_categorical(X_processed)
        
        # 3. Create domain-specific features (if applicable)
        if self.llm_enabled:
            feature_suggestions = self.llm_reason(
                prompt=f"Suggest domain-specific features for {input_data.get('domain', 'general')} domain",
                context={"columns": X.columns.tolist(), "task": task_type}
            )
            self.logger.info(f"LLM feature suggestions: {feature_suggestions}")
        
        X_processed = self._create_features(X_processed, input_data.get("domain"))
        
        # 4. Scale features
        X_processed = self._scale_features(X_processed)
        
        # 5. Feature selection
        X_selected, selected_features = self._select_features(
            X_processed, y, task_type
        )
        self.selected_features = selected_features
        
        # Final safety check: ensure no NaN values remain
        if X_selected.isnull().any().any():
            self.logger.warning("NaN values detected after feature engineering. Filling remaining NaN with 0.")
            X_selected = X_selected.fillna(0)
        
        # Combine with target
        processed_df = X_selected.copy()
        processed_df[target_column] = y.values
        
        results = {
            "processed_data": processed_df,
            "original_features": X.columns.tolist(),
            "engineered_features": X_processed.columns.tolist(),
            "selected_features": selected_features,
            "num_features_original": len(X.columns),
            "num_features_final": len(selected_features),
            "scaling_method": self.fe_config.get("scaling_method", "standard"),
            "encoding_method": self.fe_config.get("encoding_method", "onehot"),
            "target_encoder": target_encoder,
            "target_encoded": target_encoder is not None,
            "feature_engineering_summary": self._generate_summary(X, X_selected),
        }
        
        self.results = results
        self.save_state()
        
        return results
    
    def _load_data(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Load data from input"""
        if isinstance(input_data.get("data"), pd.DataFrame):
            return input_data["data"]
        elif isinstance(input_data.get("data"), str):
            return pd.read_csv(input_data["data"])
        else:
            raise ValueError("Invalid data format")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using appropriate strategies"""
        df_filled = df.copy()
        
        for col in df_filled.columns:
            if df_filled[col].isnull().any():
                if df_filled[col].dtype in ['int64', 'float64']:
                    # Numerical: fill with median
                    df_filled[col].fillna(df_filled[col].median(), inplace=True)
                else:
                    # Categorical: fill with mode
                    df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
        
        return df_filled
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        df_encoded = df.copy()
        encoding_method = self.fe_config.get("encoding_method", "onehot")
        
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        # Determine which columns are high-cardinality (too many unique values)
        HIGH_CARDINALITY_THRESHOLD = 50  # Max unique values for one-hot encoding
        
        low_cardinality_cols = []
        high_cardinality_cols = []
        
        for col in categorical_cols:
            n_unique = df_encoded[col].nunique()
            if n_unique > HIGH_CARDINALITY_THRESHOLD:
                high_cardinality_cols.append(col)
                self.logger.warning(f"Column '{col}' has {n_unique} unique values (high cardinality). Using label encoding instead of one-hot.")
            else:
                low_cardinality_cols.append(col)
        
        if encoding_method == "onehot":
            # One-hot encoding only for low cardinality columns
            if low_cardinality_cols:
                df_encoded = pd.get_dummies(
                    df_encoded, 
                    columns=low_cardinality_cols, 
                    drop_first=True,
                    prefix=low_cardinality_cols
                )
            
            # Label encoding for high cardinality columns
            for col in high_cardinality_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le
                
        elif encoding_method == "label":
            # Label encoding for all categorical columns
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le
        
        return df_encoded
    
    def _create_features(self, df: pd.DataFrame, domain: Optional[str] = None) -> pd.DataFrame:
        """Create domain-specific features"""
        df_new = df.copy()
        
        # Generic feature engineering
        numeric_cols = df_new.select_dtypes(include=[np.number]).columns
        
        # Polynomial features (selected pairs)
        if len(numeric_cols) >= 2:
            # Create interaction features for top pairs
            for i in range(min(3, len(numeric_cols))):
                for j in range(i+1, min(3, len(numeric_cols))):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    df_new[f"{col1}_x_{col2}"] = df_new[col1] * df_new[col2]
        
        # Log transformations for skewed features
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            if (df_new[col] > 0).all():  # Only if all values are positive
                df_new[f"{col}_log"] = np.log1p(df_new[col])
        
        # Domain-specific features
        if domain == "healthcare":
            # Healthcare-specific feature engineering
            self.logger.info("Applying healthcare-specific feature engineering")
            # Example: BMI calculation if height and weight exist
            # if 'height' in df_new.columns and 'weight' in df_new.columns:
            #     df_new['bmi'] = df_new['weight'] / (df_new['height'] ** 2)
            
        elif domain == "finance":
            # Finance-specific feature engineering
            self.logger.info("Applying finance-specific feature engineering")
            # Example: Financial ratios, moving averages, etc.
            # if 'income' in df_new.columns and 'expenses' in df_new.columns:
            #     df_new['savings_rate'] = (df_new['income'] - df_new['expenses']) / df_new['income']
        
        return df_new
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        df_scaled = df.copy()
        scaling_method = self.fe_config.get("scaling_method", "standard")
        
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        
        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()
        elif scaling_method == "robust":
            scaler = RobustScaler()
        else:
            self.logger.warning(f"Unknown scaling method: {scaling_method}, using standard")
            scaler = StandardScaler()
        
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
        self.scalers['main_scaler'] = scaler
        
        return df_scaled
    
    def _select_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        task_type: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select most important features"""
        selection_method = self.fe_config.get("feature_selection_method", "importance")
        
        # Limit number of features
        k = min(20, len(X.columns))  # Select top 20 or all if less
        
        if selection_method == "mutual_info":
            # Mutual information
            if task_type == "classification":
                mi = mutual_info_classif(X, y)
            else:
                mi = mutual_info_regression(X, y)
            
            # Get top k features
            top_k_idx = np.argsort(mi)[-k:]
            selected_features = X.columns[top_k_idx].tolist()
            
        elif selection_method == "rfe":
            # Recursive Feature Elimination
            if task_type == "classification":
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            
            rfe = RFE(estimator, n_features_to_select=k)
            rfe.fit(X, y)
            selected_features = X.columns[rfe.support_].tolist()
            
        else:  # "importance"
            # Feature importance from Random Forest
            if task_type == "classification":
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
            
            rf.fit(X, y)
            importances = rf.feature_importances_
            
            # Get top k features
            top_k_idx = np.argsort(importances)[-k:]
            selected_features = X.columns[top_k_idx].tolist()
        
        return X[selected_features], selected_features
    
    def _generate_summary(self, X_original: pd.DataFrame, X_final: pd.DataFrame) -> Dict[str, Any]:
        """Generate feature engineering summary"""
        return {
            "original_feature_count": len(X_original.columns),
            "final_feature_count": len(X_final.columns),
            "feature_reduction_pct": (1 - len(X_final.columns) / len(X_original.columns)) * 100,
            "numerical_features": len(X_final.select_dtypes(include=[np.number]).columns),
        }
    
    def save_artifacts(self, output_dir: str):
        """Save scalers and encoders for production use"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(self.scalers, f"{output_dir}/scalers.pkl")
        joblib.dump(self.encoders, f"{output_dir}/encoders.pkl")
        joblib.dump(self.selected_features, f"{output_dir}/selected_features.pkl")
        
        self.logger.info(f"Feature engineering artifacts saved to {output_dir}")

