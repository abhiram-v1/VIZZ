#!/usr/bin/env python3
"""
Standalone Decision Boundary Visualization Experiment
Trains AdaBoost and XGBoost on stroke data and visualizes decision boundaries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_stroke_dataset(n_samples=1000):
    """Create a realistic stroke dataset for visualization"""
    np.random.seed(42)
    
    # Generate realistic stroke data
    data = []
    
    for i in range(n_samples):
        # Age: 18-80, higher stroke risk with age
        age = np.random.normal(50, 15)
        age = np.clip(age, 18, 80)
        
        # Average glucose level: 70-300 mg/dL
        avg_glucose_level = np.random.normal(120, 40)
        avg_glucose_level = np.clip(avg_glucose_level, 70, 300)
        
        # BMI: 15-50 kg/m²
        bmi = np.random.normal(25, 5)
        bmi = np.clip(bmi, 15, 50)
        
        # Hypertension (0 or 1)
        hypertension = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Heart disease (0 or 1)
        heart_disease = np.random.choice([0, 1], p=[0.8, 0.2])
        
        # Calculate stroke probability based on realistic factors
        stroke_prob = 0.1  # Base probability
        
        # Age factor
        if age > 60:
            stroke_prob += 0.3
        elif age > 40:
            stroke_prob += 0.1
            
        # Glucose factor
        if avg_glucose_level > 200:
            stroke_prob += 0.2
        elif avg_glucose_level > 140:
            stroke_prob += 0.1
            
        # BMI factor
        if bmi > 30:
            stroke_prob += 0.1
            
        # Hypertension factor
        if hypertension:
            stroke_prob += 0.2
            
        # Heart disease factor
        if heart_disease:
            stroke_prob += 0.3
            
        # Add some randomness
        stroke_prob += np.random.normal(0, 0.05)
        stroke_prob = np.clip(stroke_prob, 0, 1)
        
        # Generate stroke label
        stroke = 1 if np.random.random() < stroke_prob else 0
        
        data.append({
            'age': age,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'stroke': stroke
        })
    
    return pd.DataFrame(data)

def plot_decision_boundary_2d(model, X, y, feature_names, title, ax):
    """Plot 2D decision boundary with filled regions"""
    
    # Create a mesh
    h = 0.5
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Get predictions for the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create custom colormap
    colors = ['#ff6b6b', '#4ecdc4']  # Red for no stroke, Teal for stroke
    cmap = ListedColormap(colors)
    
    # Plot decision boundary with filled regions
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
    ax.contour(xx, yy, Z, colors='black', linewidths=2, alpha=0.8)
    
    # Plot data points
    stroke_points = X[y == 1]
    no_stroke_points = X[y == 0]
    
    ax.scatter(no_stroke_points[:, 0], no_stroke_points[:, 1], 
              c='red', label='No Stroke', alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    ax.scatter(stroke_points[:, 0], stroke_points[:, 1], 
              c='green', label='Stroke', alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    # Styling
    ax.set_xlabel(feature_names[0], fontsize=12, fontweight='bold')
    ax.set_ylabel(feature_names[1], fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add accuracy text
    accuracy = accuracy_score(y, model.predict(X))
    ax.text(0.02, 0.98, f'Accuracy: {accuracy:.3f}', transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

def plot_decision_boundary_3d(model, X, y, feature_names, title, ax):
    """Plot 3D decision boundary"""
    
    # Create 3D scatter plot
    stroke_points = X[y == 1]
    no_stroke_points = X[y == 0]
    
    ax.scatter(no_stroke_points[:, 0], no_stroke_points[:, 1], no_stroke_points[:, 2],
              c='red', label='No Stroke', alpha=0.7, s=50)
    ax.scatter(stroke_points[:, 0], stroke_points[:, 1], stroke_points[:, 2],
              c='green', label='Stroke', alpha=0.7, s=50)
    
    # Styling
    ax.set_xlabel(feature_names[0], fontsize=10)
    ax.set_ylabel(feature_names[1], fontsize=10)
    ax.set_zlabel(feature_names[2], fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    
    # Add accuracy text
    accuracy = accuracy_score(y, model.predict(X))
    ax.text2D(0.02, 0.98, f'Accuracy: {accuracy:.3f}', transform=ax.transAxes, 
              fontsize=9, verticalalignment='top',
              bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

def train_and_visualize_models():
    """Main function to train models and create visualizations"""
    
    print("🚀 Creating Stroke Dataset...")
    df = create_stroke_dataset(1000)
    print(f"✅ Dataset created with {len(df)} samples")
    print(f"📊 Stroke rate: {df['stroke'].mean():.1%}")
    
    # Prepare features
    feature_cols = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease']
    X = df[feature_cols].values
    y = df['stroke'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n🔧 Training Models...")
    
    # Train AdaBoost
    print("📈 Training AdaBoost...")
    adaboost = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
    adaboost.fit(X_train_scaled, y_train)
    adaboost_acc = accuracy_score(y_test, adaboost.predict(X_test_scaled))
    print(f"✅ AdaBoost Accuracy: {adaboost_acc:.3f}")
    
    # Train XGBoost
    print("📈 Training XGBoost...")
    xgboost_model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    xgboost_model.fit(X_train_scaled, y_train)
    xgboost_acc = accuracy_score(y_test, xgboost_model.predict(X_test_scaled))
    print(f"✅ XGBoost Accuracy: {xgboost_acc:.3f}")
    
    print("\n🎨 Creating Visualizations...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 2D Visualizations (Age vs Glucose Level)
    print("📊 Creating 2D Decision Boundaries...")
    
    # Select features for 2D visualization
    X_2d = X_train_scaled[:, [0, 1]]  # Age and Glucose Level
    feature_names_2d = ['Age (scaled)', 'Glucose Level (scaled)']
    
    # AdaBoost 2D
    ax1 = plt.subplot(2, 3, 1)
    plot_decision_boundary_2d(adaboost, X_2d, y_train, feature_names_2d, 
                             'AdaBoost Decision Boundary\n(Age vs Glucose Level)', ax1)
    
    # XGBoost 2D
    ax2 = plt.subplot(2, 3, 2)
    plot_decision_boundary_2d(xgboost_model, X_2d, y_train, feature_names_2d, 
                             'XGBoost Decision Boundary\n(Age vs Glucose Level)', ax2)
    
    # 2D Visualizations (Age vs BMI)
    X_2d_bmi = X_train_scaled[:, [0, 2]]  # Age and BMI
    feature_names_2d_bmi = ['Age (scaled)', 'BMI (scaled)']
    
    # AdaBoost 2D BMI
    ax3 = plt.subplot(2, 3, 3)
    plot_decision_boundary_2d(adaboost, X_2d_bmi, y_train, feature_names_2d_bmi, 
                             'AdaBoost Decision Boundary\n(Age vs BMI)', ax3)
    
    # XGBoost 2D BMI
    ax4 = plt.subplot(2, 3, 4)
    plot_decision_boundary_2d(xgboost_model, X_2d_bmi, y_train, feature_names_2d_bmi, 
                             'XGBoost Decision Boundary\n(Age vs BMI)', ax4)
    
    # 3D Visualizations
    print("📊 Creating 3D Scatter Plots...")
    from mpl_toolkits.mplot3d import Axes3D
    
    X_3d = X_train_scaled[:, [0, 1, 2]]  # Age, Glucose, BMI
    feature_names_3d = ['Age (scaled)', 'Glucose (scaled)', 'BMI (scaled)']
    
    # AdaBoost 3D
    ax5 = plt.subplot(2, 3, 5, projection='3d')
    plot_decision_boundary_3d(adaboost, X_3d, y_train, feature_names_3d, 
                             'AdaBoost 3D View\n(Age vs Glucose vs BMI)', ax5)
    
    # XGBoost 3D
    ax6 = plt.subplot(2, 3, 6, projection='3d')
    plot_decision_boundary_3d(xgboost_model, X_3d, y_train, feature_names_3d, 
                             'XGBoost 3D View\n(Age vs Glucose vs BMI)', ax6)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'stroke_decision_boundaries.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Visualization saved as: {output_file}")
    
    # Show the plot
    plt.show()
    
    # Print detailed results
    print("\n📈 MODEL PERFORMANCE SUMMARY")
    print("="*50)
    print(f"AdaBoost Test Accuracy: {adaboost_acc:.3f}")
    print(f"XGBoost Test Accuracy: {xgboost_acc:.3f}")
    
    print("\n📊 AdaBoost Classification Report:")
    print(classification_report(y_test, adaboost.predict(X_test_scaled), 
                              target_names=['No Stroke', 'Stroke']))
    
    print("\n📊 XGBoost Classification Report:")
    print(classification_report(y_test, xgboost_model.predict(X_test_scaled), 
                              target_names=['No Stroke', 'Stroke']))
    
    # Feature importance
    print("\n🔍 FEATURE IMPORTANCE")
    print("="*30)
    feature_names_full = ['Age', 'Glucose Level', 'BMI', 'Hypertension', 'Heart Disease']
    
    print("AdaBoost Feature Importance:")
    for name, importance in zip(feature_names_full, adaboost.feature_importances_):
        print(f"  {name}: {importance:.3f}")
    
    print("\nXGBoost Feature Importance:")
    for name, importance in zip(feature_names_full, xgboost_model.feature_importances_):
        print(f"  {name}: {importance:.3f}")
    
    print("\n✅ Experiment completed successfully!")
    print(f"📁 Results saved in current directory")

if __name__ == "__main__":
    print("🧠 STROKE PREDICTION - DECISION BOUNDARY EXPERIMENT")
    print("="*60)
    print("This script will:")
    print("1. Create a realistic stroke dataset")
    print("2. Train AdaBoost and XGBoost models")
    print("3. Visualize decision boundaries in 2D and 3D")
    print("4. Show model performance and feature importance")
    print("="*60)
    
    try:
        train_and_visualize_models()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have all required packages installed:")
        print("   pip install numpy pandas matplotlib seaborn scikit-learn xgboost")
