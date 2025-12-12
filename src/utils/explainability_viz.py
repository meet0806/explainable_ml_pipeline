"""
Explainability Visualization Utilities
Helper functions for SHAP and LIME visualizations in Streamlit
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any
import pandas as pd


def plot_shap_feature_importance(shap_importance: Dict[str, float], title: str = "SHAP Feature Importance") -> go.Figure:
    """
    Create a bar plot of SHAP feature importance
    
    Args:
        shap_importance: Dict mapping feature names to SHAP importance values
        title: Plot title
        
    Returns:
        Plotly figure
    """
    if not shap_importance:
        return None
    
    features = list(shap_importance.keys())
    values = list(shap_importance.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker=dict(
                color=values,
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Impact")
            ),
            text=[f'{v:.4f}' for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Mean |SHAP value|",
        yaxis_title="Features",
        height=max(400, len(features) * 40),
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def plot_lime_feature_importance(lime_importance: Dict[str, float], title: str = "LIME Feature Importance") -> go.Figure:
    """
    Create a bar plot of LIME feature importance
    
    Args:
        lime_importance: Dict mapping feature names to LIME importance values
        title: Plot title
        
    Returns:
        Plotly figure
    """
    if not lime_importance:
        return None
    
    features = list(lime_importance.keys())
    values = list(lime_importance.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker=dict(
                color=values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Contribution")
            ),
            text=[f'{v:.4f}' for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Average |LIME contribution|",
        yaxis_title="Features",
        height=max(400, len(features) * 40),
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def plot_lime_instance_explanation(explanation: Dict[str, Any], instance_idx: int) -> go.Figure:
    """
    Plot LIME explanation for a single instance
    
    Args:
        explanation: Single instance explanation from LIME
        instance_idx: Index of the instance
        
    Returns:
        Plotly figure
    """
    feature_contributions = explanation.get("feature_contributions", [])
    
    if not feature_contributions:
        return None
    
    features = [fc["feature"] for fc in feature_contributions]
    contributions = [fc["contribution"] for fc in feature_contributions]
    
    # Color based on positive/negative contribution
    colors = ['green' if c > 0 else 'red' for c in contributions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=contributions,
            y=features,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{c:.4f}' for c in contributions],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f"LIME Explanation - Instance {instance_idx}",
        xaxis_title="Feature Contribution",
        yaxis_title="Features",
        height=max(400, len(features) * 40),
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_explainability_summary_table(explainability: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a summary table of explainability methods and results
    
    Args:
        explainability: Explainability dict from evaluator agent
        
    Returns:
        DataFrame with summary information
    """
    methods_available = explainability.get("methods_available", [])
    
    summary_data = {
        "Method": [],
        "Status": [],
        "Top Feature": [],
        "Details": []
    }
    
    # Feature importance
    if "feature_importance" in methods_available:
        feature_imp = explainability.get("feature_importance", {})
        if feature_imp:
            top_feature = max(feature_imp, key=feature_imp.get)
            summary_data["Method"].append("Feature Importance")
            summary_data["Status"].append("✅ Available")
            summary_data["Top Feature"].append(top_feature)
            summary_data["Details"].append(f"{len(feature_imp)} features analyzed")
    
    # SHAP
    if "shap" in methods_available:
        shap_summary = explainability.get("shap_summary", {})
        shap_importance = explainability.get("shap_feature_importance", {})
        if shap_importance:
            top_feature = max(shap_importance, key=shap_importance.get)
            samples = shap_summary.get("samples_analyzed", 0)
            summary_data["Method"].append("SHAP")
            summary_data["Status"].append("✅ Completed")
            summary_data["Top Feature"].append(top_feature)
            summary_data["Details"].append(f"{samples} samples analyzed")
    
    # LIME
    if "lime" in methods_available:
        lime_summary = explainability.get("lime_summary", {})
        lime_importance = explainability.get("lime_feature_importance", {})
        if lime_importance:
            top_feature = max(lime_importance, key=lime_importance.get)
            samples = lime_summary.get("samples_explained", 0)
            summary_data["Method"].append("LIME")
            summary_data["Status"].append("✅ Completed")
            summary_data["Top Feature"].append(top_feature)
            summary_data["Details"].append(f"{samples} instances explained")
    
    return pd.DataFrame(summary_data)


def compare_explainability_methods(explainability: Dict[str, Any]) -> go.Figure:
    """
    Compare feature rankings across different explainability methods
    
    Args:
        explainability: Explainability dict from evaluator agent
        
    Returns:
        Plotly figure comparing methods
    """
    feature_imp = explainability.get("feature_importance", {})
    shap_imp = explainability.get("shap_feature_importance", {})
    lime_imp = explainability.get("lime_feature_importance", {})
    
    # Get all unique features
    all_features = set()
    if feature_imp:
        all_features.update(feature_imp.keys())
    if shap_imp:
        all_features.update(shap_imp.keys())
    if lime_imp:
        all_features.update(lime_imp.keys())
    
    if not all_features:
        return None
    
    # Normalize importance scores to 0-1 for comparison
    def normalize_dict(d):
        if not d:
            return {}
        max_val = max(d.values()) if d else 1
        return {k: v / max_val for k, v in d.items()} if max_val > 0 else d
    
    feature_imp_norm = normalize_dict(feature_imp)
    shap_imp_norm = normalize_dict(shap_imp)
    lime_imp_norm = normalize_dict(lime_imp)
    
    # Create data for grouped bar chart
    features_list = list(all_features)[:10]  # Top 10 features
    
    fig = go.Figure()
    
    if feature_imp_norm:
        fig.add_trace(go.Bar(
            name='Feature Importance',
            x=features_list,
            y=[feature_imp_norm.get(f, 0) for f in features_list],
        ))
    
    if shap_imp_norm:
        fig.add_trace(go.Bar(
            name='SHAP',
            x=features_list,
            y=[shap_imp_norm.get(f, 0) for f in features_list],
        ))
    
    if lime_imp_norm:
        fig.add_trace(go.Bar(
            name='LIME',
            x=features_list,
            y=[lime_imp_norm.get(f, 0) for f in features_list],
        ))
    
    fig.update_layout(
        title="Feature Importance Comparison Across Methods",
        xaxis_title="Features",
        yaxis_title="Normalized Importance",
        barmode='group',
        height=500,
        xaxis={'tickangle': -45}
    )
    
    return fig
