"""
Exploratory Data Analysis (EDA) Agent
Performs comprehensive data analysis and quality checks
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
import json

from src.core.base_agent import BaseAgent


class EDAAgent(BaseAgent):
    """
    EDA Agent performs initial data exploration and quality assessment
    - Data statistics and distributions
    - Missing value analysis
    - Correlation analysis
    - Outlier detection
    - Data quality recommendations
    """
    
    def __init__(self, agent_name: str, config: Dict[str, Any], communication_protocol):
        super().__init__(agent_name, config, communication_protocol)
        self.eda_config = config.get("agents", {}).get("eda", {})
        
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute EDA analysis on provided dataset
        
        Args:
            input_data: Dict containing 'data' (DataFrame or path) and 'target' column
            
        Returns:
            Dict with EDA results and recommendations
        """
        self.logger.info("Starting EDA analysis...")
        
        # Load data
        df = self._load_data(input_data)
        target_column = input_data.get("target_column")
        
        # Perform analysis
        results = {
            "dataset_info": self._get_dataset_info(df),
            "statistical_summary": self._get_statistical_summary(df),
            "missing_values": self._analyze_missing_values(df),
            "correlation_analysis": self._analyze_correlations(df, target_column),
            "outliers": self._detect_outliers(df),
            "data_quality_score": self._calculate_data_quality(df),
            "recommendations": self._generate_recommendations(df),
        }
        
        # Use LLM for insights (if enabled)
        if self.llm_enabled:
            llm_insights = self.llm_reason(
                prompt=f"Analyze this dataset and provide insights for {input_data.get('domain', 'general')} domain",
                context=results
            )
            results["llm_insights"] = llm_insights
        
        self.results = results
        self.save_state()
        
        return results
    
    def _load_data(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Load data from input"""
        if isinstance(input_data.get("data"), pd.DataFrame):
            return input_data["data"]
        elif isinstance(input_data.get("data"), str):
            # Load from file path
            return pd.read_csv(input_data["data"])
        else:
            raise ValueError("Invalid data format. Provide DataFrame or file path.")
    
    def _get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information"""
        return {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "column_types": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
        }
    
    def _get_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistical summary of numerical columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        summary = {}
        for col in numeric_cols:
            summary[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "skewness": float(df[col].skew()),
                "kurtosis": float(df[col].kurtosis()),
            }
        
        return summary
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values in dataset"""
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df) * 100)
        
        missing_info = {}
        for col in df.columns:
            if missing_counts[col] > 0:
                missing_info[col] = {
                    "count": int(missing_counts[col]),
                    "percentage": float(missing_pct[col]),
                }
        
        threshold = self.eda_config.get("missing_value_threshold", 0.3)
        high_missing_cols = [
            col for col, info in missing_info.items() 
            if info["percentage"] / 100 > threshold
        ]
        
        return {
            "columns_with_missing": missing_info,
            "high_missing_columns": high_missing_cols,
            "total_missing_values": int(missing_counts.sum()),
        }
    
    def _analyze_correlations(
        self, 
        df: pd.DataFrame, 
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze correlations between features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {"message": "Insufficient numeric columns for correlation analysis"}
        
        corr_matrix = df[numeric_cols].corr()
        
        # Find highly correlated pairs
        threshold = self.eda_config.get("correlation_threshold", 0.7)
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append({
                        "feature_1": corr_matrix.columns[i],
                        "feature_2": corr_matrix.columns[j],
                        "correlation": float(corr_matrix.iloc[i, j]),
                    })
        
        result = {
            "high_correlation_pairs": high_corr_pairs,
            "correlation_matrix_shape": corr_matrix.shape,
        }
        
        # Target correlations if target is specified
        if target_column and target_column in numeric_cols:
            target_corr = corr_matrix[target_column].drop(target_column).to_dict()
            result["target_correlations"] = {
                k: float(v) for k, v in 
                sorted(target_corr.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            }
        
        return result
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if len(outliers) > 0:
                outlier_info[col] = {
                    "count": len(outliers),
                    "percentage": float(len(outliers) / len(df) * 100),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                }
        
        return outlier_info
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-1)"""
        scores = []
        
        # Missing values score
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        scores.append(1 - missing_pct)
        
        # Duplicate rows score
        duplicate_pct = df.duplicated().sum() / len(df)
        scores.append(1 - duplicate_pct)
        
        # Data type consistency (all columns have consistent types)
        scores.append(0.9)  # Placeholder - could be more sophisticated
        
        return float(np.mean(scores))
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate data quality recommendations"""
        recommendations = []
        
        # Check missing values
        missing_pct = df.isnull().sum() / len(df)
        if missing_pct.max() > 0.3:
            recommendations.append(
                "High missing values detected. Consider imputation or feature removal."
            )
        
        # Check for duplicates
        if df.duplicated().any():
            recommendations.append(
                f"Found {df.duplicated().sum()} duplicate rows. Consider removing them."
            )
        
        # Check for class imbalance (if target is categorical)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            if len(value_counts) > 1:
                imbalance_ratio = value_counts.max() / value_counts.min()
                if imbalance_ratio > 3:
                    recommendations.append(
                        f"Class imbalance detected in '{col}'. Consider resampling techniques."
                    )
        
        # Check feature count
        if len(df.columns) > 50:
            recommendations.append(
                "High number of features. Consider feature selection or dimensionality reduction."
            )
        
        if not recommendations:
            recommendations.append("Data quality is good. Proceed with feature engineering.")
        
        return recommendations

