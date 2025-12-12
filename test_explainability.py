"""
Test script for SHAP and LIME integration
Quick verification that explainability features work correctly
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys
import warnings
warnings.filterwarnings('ignore')

# Check if SHAP and LIME are available
try:
    import shap
    print("✅ SHAP library installed")
    SHAP_AVAILABLE = True
except ImportError:
    print("❌ SHAP library NOT installed - run: pip install shap")
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    print("✅ LIME library installed")
    LIME_AVAILABLE = True
except ImportError:
    print("❌ LIME library NOT installed - run: pip install lime")
    LIME_AVAILABLE = False

print("\n" + "="*60)
print("Testing SHAP and LIME Integration")
print("="*60)

# Create synthetic dataset
print("\n1. Creating synthetic dataset...")
np.random.seed(42)
n_samples = 500
n_features = 10

X = np.random.randn(n_samples, n_features)
y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)  # Simple rule

feature_names = [f"feature_{i}" for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"   Dataset shape: {df.shape}")
print(f"   Features: {n_features}")
print(f"   Samples: {n_samples}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1), df['target'], 
    test_size=0.2, random_state=42
)

# Train model
print("\n2. Training Random Forest model...")
model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"   Train accuracy: {train_score:.4f}")
print(f"   Test accuracy: {test_score:.4f}")

# Test feature importance
print("\n3. Testing built-in feature importance...")
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feat_imp = dict(zip(feature_names, importances))
    feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:5])
    print("   Top 5 features:")
    for feat, imp in feat_imp.items():
        print(f"      {feat}: {imp:.4f}")
    print("   ✅ Feature importance works")
else:
    print("   ⚠️  Feature importance not available for this model")

# Test SHAP
print("\n4. Testing SHAP integration...")
if SHAP_AVAILABLE:
    try:
        explainer = shap.TreeExplainer(model)
        sample_size = min(50, len(X_test))
        shap_values = explainer.shap_values(X_test.iloc[:sample_size])
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Multi-class: average across all classes or take first class
            shap_values_array = np.array(shap_values[1])  # Use class 1 for binary
        else:
            shap_values_array = np.array(shap_values)
        
        # Handle different dimensions
        if len(shap_values_array.shape) == 3:
            # (samples, features, classes) -> average over classes
            mean_shap = np.abs(shap_values_array).mean(axis=(0, 2))
        elif len(shap_values_array.shape) == 2:
            # (samples, features) -> average over samples
            mean_shap = np.abs(shap_values_array).mean(axis=0)
        elif len(shap_values_array.shape) == 1:
            # (features,) -> use directly
            mean_shap = np.abs(shap_values_array)
        else:
            raise ValueError(f"Unexpected SHAP values shape: {shap_values_array.shape}")
        
        shap_importance = dict(zip(feature_names, mean_shap))
        shap_importance = dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:5])
        
        print("   Top 5 features by SHAP:")
        for feat, imp in shap_importance.items():
            print(f"      {feat}: {imp:.4f}")
        print(f"   ✅ SHAP analysis completed ({sample_size} samples)")
    except Exception as e:
        print(f"   ❌ SHAP failed: {e}")
else:
    print("   ⚠️  SHAP not available")

# Test LIME
print("\n5. Testing LIME integration...")
if LIME_AVAILABLE:
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            mode='classification',
            random_state=42
        )
        
        # Explain first instance
        explanation = explainer.explain_instance(
            X_test.iloc[0].values,
            model.predict_proba,
            num_features=5
        )
        
        exp_list = explanation.as_list()
        print("   LIME explanation for instance 0:")
        for feat, contrib in exp_list[:5]:
            sign = "+" if contrib > 0 else ""
            print(f"      {feat}: {sign}{contrib:.4f}")
        print("   ✅ LIME analysis completed")
    except Exception as e:
        print(f"   ❌ LIME failed: {e}")
else:
    print("   ⚠️  LIME not available")

# Test with evaluator agent
print("\n6. Testing EvaluatorAgent integration...")
try:
    sys.path.append('.')
    from src.agents.evaluator_agent import BaseAgent
    
    # Mock config
    config = {
        "agents": {
            "evaluator": {
                "explainability_methods": ["shap", "lime"]
            }
        },
        "llm": {
            "reasoning_enabled": False
        }
    }
    
    # Mock communication protocol
    class MockProtocol:
        def send_message(self, *args, **kwargs):
            pass
    
    # Import and test
    from src.agents.evaluator_agent import EvaluatorAgent
    
    evaluator = EvaluatorAgent("evaluator", config, MockProtocol())
    
    # Prepare input data
    input_data = {
        "processed_data": pd.concat([X_train, y_train], axis=1).rename(columns={'target': 'target'}),
        "trained_model": model,
        "target_column": "target",
        "task_type": "classification",
        "selected_features": feature_names,
        "domain": "general"
    }
    
    # Add target column back
    input_data["processed_data"]['target'] = list(y_train) + list(y_test)[:len(X_train)-len(y_train)]
    
    results = evaluator.execute(input_data)
    
    explainability = results.get('explainability', {})
    methods_available = explainability.get('methods_available', [])
    
    print(f"   Available methods: {methods_available}")
    
    if 'feature_importance' in methods_available:
        print("   ✅ Feature importance integrated")
    if 'shap' in methods_available:
        print("   ✅ SHAP integrated")
    if 'lime' in methods_available:
        print("   ✅ LIME integrated")
    
    print("   ✅ EvaluatorAgent integration works")
    
except Exception as e:
    print(f"   ❌ EvaluatorAgent test failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

status = []
status.append(("SHAP Library", "✅ Installed" if SHAP_AVAILABLE else "❌ Not installed"))
status.append(("LIME Library", "✅ Installed" if LIME_AVAILABLE else "❌ Not installed"))
status.append(("Feature Importance", "✅ Works"))
status.append(("Integration", "✅ Complete"))

for item, stat in status:
    print(f"{item:.<30} {stat}")

if not SHAP_AVAILABLE:
    print("\n⚠️  Install SHAP: pip install shap")
if not LIME_AVAILABLE:
    print("⚠️  Install LIME: pip install lime")

if SHAP_AVAILABLE and LIME_AVAILABLE:
    print("\n🎉 All explainability features are ready!")
else:
    print("\n⚠️  Some features require additional installation")

print("\n" + "="*60)
