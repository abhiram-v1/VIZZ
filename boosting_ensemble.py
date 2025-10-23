import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

print("=" * 70)
print("BOOSTING ENSEMBLE METHODS COMPARISON")
print("=" * 70)

# 1. Load Data and Models
print("\n1. LOADING DATA AND PREPROCESSING")
print("-" * 40)

# Load the balanced dataset
df = pd.read_csv('stroke_data_balanced.csv')
print(f"Dataset shape: {df.shape}")

# Load training data
with open('training_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
X_train_scaled = data['X_train_scaled']
X_test_scaled = data['X_test_scaled']
y_train = data['y_train']
y_test = data['y_test']

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 2. Define Boosting Algorithms
print("\n2. DEFINING BOOSTING ALGORITHMS")
print("-" * 40)

boosting_models = {}

# AdaBoost with Decision Tree as base estimator
boosting_models['AdaBoost (DT)'] = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # Stump for weak learner
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

# AdaBoost with Logistic Regression as base estimator
boosting_models['AdaBoost (LR)'] = AdaBoostClassifier(
    estimator=LogisticRegression(max_iter=1000),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

# Gradient Boosting
boosting_models['Gradient Boosting'] = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# XGBoost (if available)
if XGBOOST_AVAILABLE:
    boosting_models['XGBoost'] = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        eval_metric='logloss'
    )

# LightGBM (if available)
if LIGHTGBM_AVAILABLE:
    boosting_models['LightGBM'] = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        verbose=-1
    )

# Voting Classifier with our weak learners
boosting_models['Voting Classifier'] = VotingClassifier(
    estimators=[
        ('dt', DecisionTreeClassifier(max_depth=3, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('nb', GaussianNB()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ],
    voting='soft'  # Use predicted probabilities
)

print(f"Defined {len(boosting_models)} boosting algorithms:")
for name in boosting_models.keys():
    print(f"  - {name}")

# 3. Train and Evaluate Boosting Models
print("\n3. TRAINING AND EVALUATING BOOSTING MODELS")
print("-" * 40)

results = {}
training_times = {}

for name, model in boosting_models.items():
    print(f"\nTraining {name}...")
    
    # Measure training time
    import time
    start_time = time.time()
    
    # Use scaled data for models that need it
    if name in ['AdaBoost (LR)', 'Voting Classifier']:
        X_train_use = X_train_scaled
        X_test_use = X_test_scaled
    else:
        X_train_use = X_train
        X_test_use = X_test
    
    # Train the model
    model.fit(X_train_use, y_train)
    
    # Calculate training time
    training_time = time.time() - start_time
    training_times[name] = training_time
    
    # Make predictions
    y_pred = model.predict(X_test_use)
    y_pred_proba = model.predict_proba(X_test_use)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_use, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Store results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'training_time': training_time
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  CV Score: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
    print(f"  Training Time: {training_time:.2f}s")

# 4. Load Individual Weak Classifier Results for Comparison
print("\n4. LOADING INDIVIDUAL CLASSIFIER RESULTS")
print("-" * 40)

# Load previous results
individual_results = pd.read_csv('weak_classifiers_results.csv')
print("Individual classifier results:")
print(individual_results)

# Add individual results to comparison
individual_models = {
    'Decision Tree': individual_results.iloc[0],
    'Logistic Regression': individual_results.iloc[1],
    'Naive Bayes': individual_results.iloc[2],
    'K-Nearest Neighbors': individual_results.iloc[3]
}

# 5. Create Comprehensive Comparison
print("\n5. CREATING COMPREHENSIVE COMPARISON")
print("-" * 40)

# Prepare comparison data
comparison_data = []

# Add boosting results
for name, result in results.items():
    comparison_data.append({
        'Model': name,
        'Type': 'Boosting',
        'Accuracy': result['accuracy'],
        'Precision': result['precision'],
        'Recall': result['recall'],
        'F1-Score': result['f1_score'],
        'CV_Mean': result['cv_mean'],
        'CV_Std': result['cv_std'],
        'Training_Time': result['training_time']
    })

# Add individual results
for name, result in individual_models.items():
    comparison_data.append({
        'Model': name,
        'Type': 'Individual',
        'Accuracy': result['Accuracy'],
        'Precision': result['Precision'],
        'Recall': result['Recall'],
        'F1-Score': result['F1-Score'],
        'CV_Mean': 0,  # Not available in CSV
        'CV_Std': 0,   # Not available in CSV
        'Training_Time': 0  # Not measured for individual models
    })

comparison_df = pd.DataFrame(comparison_data)

# Sort by F1-Score
comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

print("\nCOMPREHENSIVE COMPARISON (sorted by F1-Score):")
print(comparison_df.round(4))

# 6. Create Visualization
print("\n6. CREATING COMPARISON VISUALIZATIONS")
print("-" * 40)

# Set up the plotting style
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Boosting vs Individual Classifiers Comparison', fontsize=16, fontweight='bold')

# 1. Accuracy Comparison
ax1 = axes[0, 0]
colors = ['#2E8B57' if t == 'Boosting' else '#4682B4' for t in comparison_df['Type']]
bars1 = ax1.bar(range(len(comparison_df)), comparison_df['Accuracy'], color=colors, alpha=0.7)
ax1.set_title('Accuracy Comparison', fontweight='bold')
ax1.set_ylabel('Accuracy')
ax1.set_xticks(range(len(comparison_df)))
ax1.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 2. F1-Score Comparison
ax2 = axes[0, 1]
bars2 = ax2.bar(range(len(comparison_df)), comparison_df['F1-Score'], color=colors, alpha=0.7)
ax2.set_title('F1-Score Comparison', fontweight='bold')
ax2.set_ylabel('F1-Score')
ax2.set_xticks(range(len(comparison_df)))
ax2.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 3. Precision vs Recall Scatter
ax3 = axes[1, 0]
boosting_data = comparison_df[comparison_df['Type'] == 'Boosting']
individual_data = comparison_df[comparison_df['Type'] == 'Individual']

ax3.scatter(boosting_data['Recall'], boosting_data['Precision'], 
           c='#2E8B57', s=100, alpha=0.7, label='Boosting', marker='o')
ax3.scatter(individual_data['Recall'], individual_data['Precision'], 
           c='#4682B4', s=100, alpha=0.7, label='Individual', marker='s')

ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title('Precision vs Recall', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Add model labels
for i, row in comparison_df.iterrows():
    ax3.annotate(row['Model'], (row['Recall'], row['Precision']), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

# 4. Training Time Comparison (only for boosting models)
ax4 = axes[1, 1]
boosting_only = comparison_df[comparison_df['Type'] == 'Boosting']
if len(boosting_only) > 0:
    bars4 = ax4.bar(range(len(boosting_only)), boosting_only['Training_Time'], 
                   color='#2E8B57', alpha=0.7)
    ax4.set_title('Training Time Comparison (Boosting Only)', fontweight='bold')
    ax4.set_ylabel('Training Time (seconds)')
    ax4.set_xticks(range(len(boosting_only)))
    ax4.set_xticklabels(boosting_only['Model'], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}s', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('boosting_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Best Model Analysis
print("\n7. BEST MODEL ANALYSIS")
print("-" * 40)

best_model_name = comparison_df.iloc[0]['Model']
best_model_type = comparison_df.iloc[0]['Type']
best_f1 = comparison_df.iloc[0]['F1-Score']

print(f"Best performing model: {best_model_name} ({best_model_type})")
print(f"Best F1-Score: {best_f1:.4f}")

# Show improvement over individual models
best_individual_f1 = comparison_df[comparison_df['Type'] == 'Individual']['F1-Score'].max()
improvement = ((best_f1 - best_individual_f1) / best_individual_f1) * 100

print(f"Improvement over best individual model: {improvement:.2f}%")

# 8. Save Results
print("\n8. SAVING RESULTS")
print("-" * 40)

# Save comprehensive comparison
comparison_df.to_csv('boosting_comparison_results.csv', index=False)
print("Comprehensive comparison saved to 'boosting_comparison_results.csv'")

# Save best model
if best_model_type == 'Boosting':
    best_model = results[best_model_name]['model']
    with open('best_boosting_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Best boosting model ({best_model_name}) saved to 'best_boosting_model.pkl'")

# 9. Detailed Classification Report for Best Model
print("\n9. DETAILED CLASSIFICATION REPORT FOR BEST MODEL")
print("-" * 40)

if best_model_type == 'Boosting':
    best_predictions = results[best_model_name]['predictions']
    print(f"\nDetailed report for {best_model_name}:")
    print(classification_report(y_test, best_predictions))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, best_predictions)
    print(f"\nConfusion Matrix for {best_model_name}:")
    print(cm)

print(f"\n" + "=" * 70)
print("BOOSTING ENSEMBLE COMPARISON COMPLETED!")
print("=" * 70)
print("Summary:")
print(f"  - Best model: {best_model_name}")
print(f"  - Best F1-Score: {best_f1:.4f}")
print(f"  - Improvement: {improvement:.2f}%")
print(f"  - Visualization saved: boosting_comparison.png")
print(f"  - Results saved: boosting_comparison_results.csv")
