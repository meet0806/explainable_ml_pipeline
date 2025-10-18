#!/usr/bin/env python3
"""
Demo Script - Quick Pipeline Demonstration
Run this to see the complete pipeline in action with sample data
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import yaml
import os

from src.orchestrator import Orchestrator


def create_demo_dataset():
    """Create a demo dataset for testing"""
    print("📊 Creating demo dataset...")
    
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=2,
        weights=[0.6, 0.4],
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(20)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some categorical features
    df['category_A'] = np.random.choice(['Type_A', 'Type_B', 'Type_C'], size=500)
    df['category_B'] = np.random.choice(['Group_1', 'Group_2'], size=500)
    
    # Add some missing values (5%) - only to numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mask = np.random.random(500) < 0.05
        df.loc[mask, col] = np.nan
    
    print(f"   ✓ Dataset created: {df.shape}")
    print(f"   ✓ Target distribution: {dict(df['target'].value_counts())}")
    
    return df


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*80)
    print("  🤖 EXPLAINABLE ML PIPELINE - DEMO")
    print("  Agentic AI for Healthcare and Finance")
    print("="*80 + "\n")


def print_section(title):
    """Print section header"""
    print("\n" + "─"*80)
    print(f"  {title}")
    print("─"*80)


def main():
    """Run demo"""
    print_banner()
    
    try:
        # Create demo dataset
        print_section("STEP 1: Data Preparation")
        df = create_demo_dataset()
        
        # Load configuration
        print_section("STEP 2: Loading Configuration")
        print("📋 Loading config.yaml...")
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("   ✓ Configuration loaded")
        
        # Initialize orchestrator
        print_section("STEP 3: Initializing Pipeline")
        print("🔧 Initializing orchestrator and agents...")
        orchestrator = Orchestrator(config)
        print(f"   ✓ {len(orchestrator.agents)} agents initialized")
        for agent_name in orchestrator.agents.keys():
            print(f"      • {agent_name.replace('_', ' ').title()}")
        
        # Run pipeline
        print_section("STEP 4: Running ML Pipeline")
        print("🚀 Starting pipeline execution...\n")
        
        results = orchestrator.run_pipeline(
            data=df,
            target_column='target',
            task_type='classification',
            domain='general'
        )
        
        # Display results
        print_section("STEP 5: Results Summary")
        
        final_results = results['final_results']
        
        print("\n📊 PIPELINE RESULTS:")
        print(f"   • Model Approved: {'✅ YES' if final_results['model_approved'] else '❌ NO'}")
        print(f"   • Deployment Ready: {'✅ YES' if final_results['deployment_ready'] else '❌ NO'}")
        print(f"   • Best Model: {final_results['best_model']}")
        print(f"   • Performance Score: {final_results['performance_score']:.4f}")
        
        print("\n📈 KEY METRICS:")
        metrics = final_results['metrics']
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            if metric_name in metrics:
                print(f"   • {metric_name.upper().replace('_', ' ')}: {metrics[metric_name]:.4f}")
        
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(results['recommendations'][:5], 1):
            print(f"   {i}. {rec}")
        
        # Pipeline info
        print("\n⏱️  EXECUTION INFO:")
        pipeline_info = results['pipeline_info']
        print(f"   • Total Iterations: {pipeline_info['total_iterations']}")
        print(f"   • Execution Time: {pipeline_info['execution_time']:.2f} seconds")
        print(f"   • Status: {pipeline_info['status'].upper()}")
        
        # Save model
        print_section("STEP 6: Saving Model")
        print("💾 Saving trained model...")
        orchestrator.save_final_model("models/demo_model.pkl")
        print("   ✓ Model saved to: models/demo_model.pkl")
        
        # Final message
        print("\n" + "="*80)
        print("  ✅ DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\n📚 Next Steps:")
        print("   1. Check results/ directory for detailed reports")
        print("   2. Check logs/ directory for execution logs")
        print("   3. Check models/ directory for saved model")
        print("   4. Try: python examples/example_healthcare.py")
        print("   5. Try: python examples/example_finance.py")
        print("   6. Read: README.md for full documentation")
        
        print("\n🎉 You're ready to build explainable ML models!\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

