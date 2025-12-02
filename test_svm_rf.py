"""
Quick test to verify SVM and Random Forest work with heart.csv
"""
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('data/heart.csv')
print(f"Data shape: {df.shape}")
print(f"Target distribution:\n{df['target'].value_counts()}")

X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test SVM
print("\n" + "="*50)
print("Testing SVM...")
print("="*50)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(random_state=42)
svm.fit(X_train_scaled, y_train)
svm_score = svm.score(X_test_scaled, y_test)
print(f"SVM Test Score: {svm_score:.4f}")

# Cross-validation
svm_cv_scores = cross_val_score(svm, X_train_scaled, y_train, cv=5)
print(f"SVM CV Scores: {svm_cv_scores}")
print(f"SVM CV Mean: {svm_cv_scores.mean():.4f} (+/- {svm_cv_scores.std() * 2:.4f})")

# Test Random Forest
print("\n" + "="*50)
print("Testing Random Forest...")
print("="*50)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)
print(f"RF Test Score: {rf_score:.4f}")

# Cross-validation
rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"RF CV Scores: {rf_cv_scores}")
print(f"RF CV Mean: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std() * 2:.4f})")

print("\n✅ Both models work correctly!")
