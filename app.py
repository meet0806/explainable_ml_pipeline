"""
Streamlit Web UI for Explainable ML Agentic Pipeline
Healthcare and Finance Domain Application
"""

import streamlit as st
import pandas as pd
import yaml
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Add src to path
sys.path.append('.')

from src.orchestrator import Orchestrator
from src.utils.run_history import RunHistory

# Page config
st.set_page_config(
    page_title="Explainable ML Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .agent-card {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def load_config():
    """Load configuration"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return None


def save_uploaded_file(uploaded_file):
    """Save uploaded file temporarily"""
    try:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


def display_dataframe_info(df):
    """Display dataset information"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Rows", f"{len(df):,}")
    with col2:
        st.metric("📋 Columns", len(df.columns))
    with col3:
        st.metric("💾 Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        st.metric("⚠️ Missing %", f"{missing_pct:.1f}%")


def plot_metrics(metrics, task_type):
    """Plot performance metrics"""
    if task_type == "classification":
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_values = [metrics.get(m, 0) for m in metric_names]
        
        fig = go.Figure(data=[
            go.Bar(
                x=metric_names,
                y=metric_values,
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                text=[f"{v:.3f}" for v in metric_values],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Classification Metrics",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            height=400
        )
        
    else:  # regression
        metric_names = ['r2_score', 'rmse', 'mae']
        metric_values = [metrics.get(m, 0) for m in metric_names]
        
        fig = go.Figure(data=[
            go.Bar(
                x=metric_names,
                y=metric_values,
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                text=[f"{v:.3f}" for v in metric_values],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Regression Metrics",
            yaxis_title="Score",
            height=400
        )
    
    return fig


def display_confusion_matrix(metrics):
    """Display confusion matrix"""
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted 0', 'Predicted 1'],
            y=['Actual 0', 'Actual 1'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20},
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            height=400
        )
        
        return fig
    return None


def main():
    """Main Streamlit app"""
    
    # Initialize run history
    if 'run_history' not in st.session_state:
        st.session_state['run_history'] = RunHistory()
    
    run_history = st.session_state['run_history']
    
    # Header
    st.markdown('<div class="main-header">🤖 Explainable ML Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Agentic AI for Healthcare and Finance</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/000000/artificial-intelligence.png", width=150)
        st.title("⚙️ Configuration")
        
        # Load config
        config = load_config()
        
        if config:
            # Task type
            task_type = st.selectbox(
                "📋 Task Type",
                ["classification", "regression"],
                index=0
            )
            
            # Domain
            domain = st.selectbox(
                "🏥 Domain",
                ["healthcare", "finance", "general"],
                index=0
            )
            
            # LLM Settings
            st.subheader("🧠 LLM Settings")
            llm_enabled = st.checkbox(
                "Enable LLM Reasoning",
                value=config.get("llm", {}).get("reasoning_enabled", False)
            )
            
            if llm_enabled:
                llm_model = st.selectbox(
                    "Model",
                    ["llama3.1:8b", "llama3.1:70b", "mistral:7b"],
                    index=0
                )
            
            # Agent Settings
            st.subheader("🤖 Agent Settings")
            
            with st.expander("Model Algorithms"):
                algorithms = st.multiselect(
                    "Select Algorithms",
                    ["random_forest", "xgboost", "logistic_regression", "svm"],
                    default=["random_forest", "xgboost"]
                )
            
            with st.expander("Advanced Settings"):
                cv_folds = st.slider("CV Folds", 3, 10, 5)
                perf_threshold = st.slider("Performance Threshold", 0.5, 1.0, 0.75, 0.05)
                max_retrain = st.slider("Max Retrain Cycles", 1, 5, 3)
            
            st.divider()
            
            # Run History Section
            st.subheader("📜 Run History")
            
            all_runs = run_history.get_all_runs()
            
            if all_runs:
                stats = run_history.get_stats()
                
                # Quick stats
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Runs", stats['total_runs'])
                with col2:
                    st.metric("Deployed", stats['successful_deployments'])
                
                # Run selector
                run_options = {
                    f"{run['run_name']} ({run['task_type']})": run['run_id'] 
                    for run in all_runs
                }
                
                selected_run_name = st.selectbox(
                    "Load Previous Run",
                    options=list(run_options.keys()),
                    index=None,
                    placeholder="Select a run to load..."
                )
                
                if selected_run_name:
                    selected_run_id = run_options[selected_run_name]
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if st.button("📂 Load Run", use_container_width=True):
                            # Load the selected run
                            loaded_results = run_history.get_run(selected_run_id)
                            if loaded_results:
                                st.session_state['results'] = loaded_results
                                st.session_state['task_type'] = run_history.get_run_metadata(selected_run_id)['task_type']
                                st.session_state['current_run_id'] = selected_run_id
                                st.success(f"✅ Loaded: {selected_run_name}")
                                st.rerun()
                    
                    with col2:
                        if st.button("🗑️", use_container_width=True):
                            if run_history.delete_run(selected_run_id):
                                st.success("Deleted!")
                                st.rerun()
            else:
                st.info("No runs yet. Start your first pipeline!")
            
            st.divider()
            st.caption("v1.0.0 | Built with ❤️")
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Upload & Run", "📊 Results", "🤖 Agents", "📈 Visualizations", "🔄 Compare Runs"])
    
    with tab1:
        st.header("📤 Upload Dataset")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your dataset in CSV format"
        )
        
        if uploaded_file is not None:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
            
            # Display data info
            st.subheader("📊 Dataset Preview")
            display_dataframe_info(df)
            
            # Show first rows
            st.dataframe(df.head(10), use_container_width=True)
            
            # Target column selection
            st.subheader("🎯 Select Target Column")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                target_column = st.selectbox(
                    "Target Variable",
                    options=df.columns.tolist(),
                    index=len(df.columns)-1
                )
            
            with col2:
                st.metric("Unique Values", df[target_column].nunique())
            
            # Target distribution
            st.subheader("📈 Target Distribution")
            target_counts = df[target_column].value_counts()
            
            fig = px.bar(
                x=target_counts.index,
                y=target_counts.values,
                labels={'x': target_column, 'y': 'Count'},
                title=f"Distribution of {target_column}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Run pipeline button
            st.divider()
            
            run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
            with run_col2:
                run_button = st.button(
                    "🚀 Run ML Pipeline",
                    type="primary",
                    use_container_width=True
                )
            
            if run_button:
                # Save file
                file_path = save_uploaded_file(uploaded_file)
                
                # Update config
                if llm_enabled:
                    config['llm']['reasoning_enabled'] = True
                    config['llm']['model'] = llm_model
                
                config['agents']['model_tuning']['algorithms'] = algorithms
                config['agents']['model_tuning']['cv_folds'] = cv_folds
                config['agents']['judge']['min_performance_threshold'] = perf_threshold
                config['agents']['judge']['max_retrain_cycles'] = max_retrain
                
                # Run pipeline
                with st.spinner("🔄 Pipeline running... This may take a few minutes."):
                    try:
                        # Initialize orchestrator
                        orchestrator = Orchestrator(config)
                        
                        # Create progress placeholder
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Run pipeline
                        status_text.text("Initializing agents...")
                        progress_bar.progress(10)
                        
                        results = orchestrator.run_pipeline(
                            data=df,
                            target_column=target_column,
                            task_type=task_type,
                            domain=domain
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Pipeline completed!")
                        
                        # Store results in session state
                        st.session_state['results'] = results
                        st.session_state['task_type'] = task_type
                        st.session_state['orchestrator'] = orchestrator
                        
                        # Save run to history
                        dataset_name = uploaded_file.name if uploaded_file else "demo_dataset"
                        run_id = run_history.save_run(
                            results=results,
                            dataset_name=dataset_name,
                            task_type=task_type,
                            domain=domain,
                            target_column=target_column,
                            dataset_shape=df.shape
                        )
                        st.session_state['current_run_id'] = run_id
                        
                        st.success("🎉 Pipeline completed successfully!")
                        st.balloons()
                        
                        # Display quick summary
                        final_results = results['final_results']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(
                                "Model Approved",
                                "✅ Yes" if final_results['model_approved'] else "❌ No"
                            )
                        with col2:
                            st.metric(
                                "Best Model",
                                final_results['best_model']
                            )
                        with col3:
                            st.metric(
                                "Performance",
                                f"{final_results['performance_score']:.3f}"
                            )
                        with col4:
                            st.metric(
                                "Iterations",
                                results['pipeline_info']['total_iterations']
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Error running pipeline: {str(e)}")
                        st.exception(e)
    
    with tab2:
        st.header("📊 Pipeline Results")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            final_results = results['final_results']
            
            # Status cards
            col1, col2 = st.columns(2)
            
            with col1:
                if final_results['model_approved']:
                    st.markdown(
                        '<div class="success-box"><h3>✅ Model Approved</h3><p>The model meets performance thresholds.</p></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="warning-box"><h3>⚠️ Model Not Approved</h3><p>Performance below threshold. Retraining recommended.</p></div>',
                        unsafe_allow_html=True
                    )
            
            with col2:
                if final_results['deployment_ready']:
                    st.markdown(
                        '<div class="success-box"><h3>🚀 Deployment Ready</h3><p>Model is ready for production deployment.</p></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="warning-box"><h3>⏸️ Not Ready</h3><p>Additional validation required.</p></div>',
                        unsafe_allow_html=True
                    )
            
            # All Models Comparison
            st.subheader("🔬 All Models Trained & Compared")
            
            # Extract all models from iterations
            all_models_data = []
            for iter_key, iter_data in results.get('all_iterations', {}).items():
                if 'model_tuning' in iter_data:
                    model_tuning = iter_data['model_tuning']
                    for model in model_tuning.get('all_models', []):
                        all_models_data.append({
                            'Model': model['algorithm'].replace('_', ' ').title(),
                            'CV Score': f"{model['cv_score']:.4f}",
                            'CV Score (numeric)': model['cv_score'],
                            'Parameters': ', '.join([f"{k}={v}" for k, v in model.get('best_params', {}).items()][:3]),
                            'Selected': '✅ Best' if model['algorithm'] == final_results['best_model'] else ''
                        })
            
            if all_models_data:
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Models Trained", len(all_models_data))
                with col2:
                    best_model_name = final_results['best_model'].replace('_', ' ').title()
                    st.metric("Best Model", best_model_name)
                with col3:
                    best_cv_score = max([m['CV Score (numeric)'] for m in all_models_data])
                    st.metric("Best CV Score", f"{best_cv_score:.4f}")
                
                # Create DataFrame
                models_df = pd.DataFrame(all_models_data)
                models_df = models_df.drop('CV Score (numeric)', axis=1)  # Remove numeric column used for calculations
                
                # Display as styled table
                st.dataframe(
                    models_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Visual comparison chart
                if len(all_models_data) > 1:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig = go.Figure()
                        
                        colors = ['#2ecc71' if m['Selected'] == '✅ Best' else '#3498db' for m in all_models_data]
                        
                        fig.add_trace(go.Bar(
                            x=[m['Model'] for m in all_models_data],
                            y=[float(m['CV Score']) for m in all_models_data],
                            marker_color=colors,
                            text=[m['CV Score'] for m in all_models_data],
                            textposition='auto',
                            hovertemplate='<b>%{x}</b><br>CV Score: %{y:.4f}<extra></extra>'
                        ))
                        
                        fig.update_layout(
                            title="Cross-Validation Scores Comparison",
                            xaxis_title="Model",
                            yaxis_title="CV Score",
                            height=400,
                            showlegend=False,
                            yaxis_range=[0, 1.05]
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 📊 Model Ranking")
                        
                        # Sort by CV score
                        sorted_models = sorted(all_models_data, key=lambda x: x['CV Score (numeric)'], reverse=True)
                        
                        for i, model in enumerate(sorted_models, 1):
                            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                            st.markdown(f"{medal} **{model['Model']}**")
                            st.caption(f"CV Score: {model['CV Score']}")
                
                # Show detailed parameters in expander
                with st.expander("🔧 View All Model Parameters"):
                    for iter_key, iter_data in results.get('all_iterations', {}).items():
                        if 'model_tuning' in iter_data:
                            model_tuning = iter_data['model_tuning']
                            for model in model_tuning.get('all_models', []):
                                is_best = model['algorithm'] == final_results['best_model']
                                status = "✅ **SELECTED**" if is_best else ""
                                
                                st.markdown(f"**{model['algorithm'].replace('_', ' ').title()}** {status}")
                                st.write(f"CV Score: `{model['cv_score']:.6f}`")
                                st.json(model.get('best_params', {}))
                                st.divider()
            else:
                st.info("Model comparison data not available")
            
            # Metrics
            st.subheader("📈 Best Model Performance Metrics")
            
            metrics = final_results['metrics']
            task_type = st.session_state['task_type']
            
            # Plot metrics
            fig = plot_metrics(metrics, task_type)
            st.plotly_chart(fig, use_container_width=True)
            
            # Confusion matrix for classification
            if task_type == "classification":
                cm_fig = display_confusion_matrix(metrics)
                if cm_fig:
                    st.plotly_chart(cm_fig, use_container_width=True)
            
            # Detailed metrics
            with st.expander("📋 Detailed Metrics"):
                st.json(metrics)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            for i, rec in enumerate(results['recommendations'], 1):
                st.write(f"{i}. {rec}")
            
            # Download results
            st.subheader("💾 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                results_json = json.dumps(results, indent=2, default=str)
                st.download_button(
                    "📥 Download Results (JSON)",
                    data=results_json,
                    file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                # Save model button
                if st.button("💾 Save Model", use_container_width=True):
                    try:
                        orchestrator = st.session_state['orchestrator']
                        model_path = f"models/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                        orchestrator.save_final_model(model_path)
                        st.success(f"✅ Model saved to {model_path}")
                    except Exception as e:
                        st.error(f"Error saving model: {e}")
        
        else:
            st.info("👆 Upload a dataset and run the pipeline to see results here.")
    
    with tab3:
        st.header("🤖 Agent Communication")
        
        if 'orchestrator' in st.session_state:
            orchestrator = st.session_state['orchestrator']
            messages = orchestrator.get_message_history()
            
            st.metric("Total Messages", len(messages))
            
            # Message timeline
            for i, msg in enumerate(messages, 1):
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    
                    with col1:
                        st.caption(f"Message {i}")
                        st.caption(msg['timestamp'][:19])
                    
                    with col2:
                        msg_type = msg['message_type']
                        
                        # Color schemes for different agents and message types
                        agent_colors = {
                            'orchestrator': '#9c27b0',  # Purple
                            'eda': '#2196f3',            # Blue
                            'feature_engineering': '#009688',  # Teal
                            'model_tuning': '#ff9800',   # Orange
                            'evaluator': '#4caf50',      # Green
                            'judge': '#f44336'           # Red
                        }
                        
                        msg_type_colors = {
                            'request': '#2196f3',
                            'response': '#4caf50',
                            'decision': '#ff9800',
                            'error': '#f44336'
                        }
                        
                        sender_color = agent_colors.get(msg["sender"].lower(), '#757575')
                        receiver_color = agent_colors.get(msg["receiver"].lower(), '#757575')
                        type_color = msg_type_colors.get(msg_type, '#757575')
                        
                        st.markdown(
                            f'<div style="padding: 12px; border-radius: 8px; '
                            f'background: linear-gradient(135deg, {sender_color}15 0%, {receiver_color}15 100%); '
                            f'border-left: 4px solid {sender_color};">'
                            f'<span style="background: {sender_color}; color: white; padding: 4px 12px; '
                            f'border-radius: 12px; font-weight: bold; font-size: 14px;">'
                            f'{msg["sender"]}</span> '
                            f'<span style="color: #666; margin: 0 8px;">→</span> '
                            f'<span style="background: {receiver_color}; color: white; padding: 4px 12px; '
                            f'border-radius: 12px; font-weight: bold; font-size: 14px;">'
                            f'{msg["receiver"]}</span> '
                            f'<span style="float: right; background: {type_color}; color: white; '
                            f'padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold;">'
                            f'{msg_type.upper()}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    st.divider()
        else:
            st.info("👆 Run the pipeline to see agent communication here.")
    
    with tab4:
        st.header("📈 Visualizations")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            
            # Performance over iterations
            iterations = results.get('all_iterations', {})
            
            if len(iterations) > 1:
                st.subheader("📊 Performance Across Iterations")
                
                iter_nums = []
                scores = []
                
                for key, value in iterations.items():
                    if 'judgment' in value:
                        iter_num = int(key.split('_')[1])
                        score = value['judgment']['judgment']['performance_score']
                        iter_nums.append(iter_num)
                        scores.append(score)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=iter_nums,
                    y=scores,
                    mode='lines+markers',
                    name='Performance Score',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=10)
                ))
                
                fig.update_layout(
                    title="Performance Score Across Iterations",
                    xaxis_title="Iteration",
                    yaxis_title="Score",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Pipeline execution time
            st.subheader("⏱️ Execution Time")
            
            pipeline_info = results['pipeline_info']
            exec_time = pipeline_info['execution_time']
            
            st.metric("Total Execution Time", f"{exec_time:.2f} seconds")
            
            # Time breakdown (placeholder - would need actual timing data)
            stages = ['EDA', 'Feature Eng.', 'Model Tuning', 'Evaluation', 'Judge']
            times = [exec_time * 0.1, exec_time * 0.15, exec_time * 0.5, exec_time * 0.15, exec_time * 0.1]
            
            fig = px.pie(
                values=times,
                names=stages,
                title="Time Distribution by Stage (Estimated)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("👆 Run the pipeline to see visualizations here.")
    
    with tab5:
        st.header("🔄 Compare Multiple Runs")
        
        all_runs = run_history.get_all_runs()
        
        if len(all_runs) >= 2:
            # Multi-select for runs to compare
            run_options = {
                f"{run['run_name']} - {run['model_name']} ({run['timestamp'][:19]})": run['run_id']
                for run in all_runs
            }
            
            selected_runs = st.multiselect(
                "Select runs to compare (2-5 runs)",
                options=list(run_options.keys()),
                max_selections=5
            )
            
            if len(selected_runs) >= 2:
                selected_ids = [run_options[run] for run in selected_runs]
                
                # Get comparison data
                comparison_df = run_history.get_comparison_data(selected_ids)
                
                st.subheader("📊 Side-by-Side Comparison")
                st.dataframe(comparison_df, use_container_width=True)
                
                # Visualize metrics comparison
                st.subheader("📈 Metrics Comparison")
                
                # Determine which metrics to plot based on task type
                first_run = run_history.get_run_metadata(selected_ids[0])
                task_type = first_run['task_type']
                
                if task_type == "classification":
                    metric_cols = ['accuracy', 'f1_score', 'precision', 'recall']
                    metric_labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
                else:
                    metric_cols = ['r2_score', 'rmse', 'mae']
                    metric_labels = ['R² Score', 'RMSE', 'MAE']
                
                # Create grouped bar chart
                fig = go.Figure()
                
                for i, metric in enumerate(metric_cols):
                    if metric in comparison_df.columns:
                        fig.add_trace(go.Bar(
                            name=metric_labels[i],
                            x=comparison_df['Run Name'],
                            y=comparison_df[metric],
                            text=comparison_df[metric].round(3),
                            textposition='auto',
                        ))
                
                fig.update_layout(
                    title=f"{task_type.capitalize()} Metrics Comparison",
                    xaxis_title="Run",
                    yaxis_title="Score",
                    barmode='group',
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Best performer
                st.subheader("🏆 Best Performer")
                
                if task_type == "classification":
                    best_idx = comparison_df['accuracy'].idxmax()
                    best_metric = 'Accuracy'
                    best_value = comparison_df.loc[best_idx, 'accuracy']
                else:
                    best_idx = comparison_df['r2_score'].idxmax()
                    best_metric = 'R² Score'
                    best_value = comparison_df.loc[best_idx, 'r2_score']
                
                best_run = comparison_df.loc[best_idx, 'Run Name']
                best_model = comparison_df.loc[best_idx, 'Model']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🥇 Best Run", best_run)
                with col2:
                    st.metric("🎯 Best Model", best_model)
                with col3:
                    st.metric(f"📊 Best {best_metric}", f"{best_value:.3f}")
                
                # Export comparison
                st.subheader("💾 Export Comparison")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = comparison_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=f"run_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    json_data = comparison_df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📥 Download as JSON",
                        data=json_data,
                        file_name=f"run_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            elif len(selected_runs) == 1:
                st.info("Please select at least 2 runs to compare.")
            else:
                st.info("Select runs from the dropdown above to start comparing.")
        
        elif len(all_runs) == 1:
            st.warning("You need at least 2 runs to compare. Run the pipeline again to create more runs!")
        else:
            st.info("No runs available yet. Start by running your first pipeline!")
    
    # Footer
    st.divider()
    st.caption("🤖 Explainable ML Pipeline | Built with Streamlit & LangChain | Powered by Llama 3.1")


if __name__ == "__main__":
    main()

