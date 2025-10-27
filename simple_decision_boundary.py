#!/usr/bin/env python3
"""
Simple Decision Boundary Visualization
Quick experiment with AdaBoost and XGBoost on stroke data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

def create_simple_stroke_data(n_samples=500):
    """Create simple stroke dataset for visualization"""
    np.random.seed(42)
    
    # Generate two clusters with some overlap
    # Cluster 1: No Stroke (lower risk)
    n_no_stroke = n_samples // 2
    age_no_stroke = np.random.normal(35, 10, n_no_stroke)
    glucose_no_stroke = np.random.normal(100, 20, n_no_stroke)
    
    # Cluster 2: Stroke (higher risk)
    n_stroke = n_samples - n_no_stroke
    age_stroke = np.random.normal(60, 15, n_stroke)
    glucose_stroke = np.random.normal(150, 30, n_stroke)
    
    # Combine data
    X = np.vstack([
        np.column_stack([age_no_stroke, glucose_no_stroke]),
        np.column_stack([age_stroke, glucose_stroke])
    ])
    
    y = np.hstack([np.zeros(n_no_stroke), np.ones(n_stroke)])
    
    return X, y

def plot_decision_boundary(model, X, y, title, ax):
    """Plot decision boundary with filled regions and smooth boundary"""
    
    # Create finer mesh for smoother boundaries
    h = 0.1  # Much finer resolution
    x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Get probability predictions for smoother boundaries
    if hasattr(model, 'predict_proba'):
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot filled regions with smooth colors
    colors = ['#ffcccc', '#ccffcc']  # Light red, light green
    cmap = ListedColormap(colors)
    ax.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap=cmap)
    
    # Plot ONLY the main decision boundary (0.5 probability)
    if hasattr(model, 'predict_proba'):
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3, linestyles='-')
    else:
        # For models without predict_proba, use prediction boundary
        Z_pred = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contour(xx, yy, Z_pred, levels=[0.5], colors='black', linewidths=3, linestyles='-')
    
    # Plot data points
    stroke_points = X[y == 1]
    no_stroke_points = X[y == 0]
    
    ax.scatter(no_stroke_points[:, 0], no_stroke_points[:, 1], 
              c='red', label='No Stroke', alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
    ax.scatter(stroke_points[:, 0], stroke_points[:, 1], 
              c='green', label='Stroke', alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
    
    # Styling
    ax.set_xlabel('Age (scaled)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Glucose Level (scaled)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add accuracy
    accuracy = accuracy_score(y, model.predict(X))
    ax.text(0.02, 0.98, f'Accuracy: {accuracy:.3f}', transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))

def main():
    """Main experiment function"""
    print("STROKE PREDICTION - DECISION BOUNDARY EXPERIMENT")
    print("="*60)
    
    # Create dataset
    print("Creating stroke dataset...")
    X, y = create_simple_stroke_data(500)
    print(f"Dataset created: {len(X)} samples, {y.mean():.1%} stroke rate")
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\nTraining models...")
    
    # AdaBoost
    print("Training AdaBoost...")
    adaboost = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=20,
        learning_rate=1.0,
        random_state=42
    )
    adaboost.fit(X_train_scaled, y_train)
    adaboost_acc = accuracy_score(y_test, adaboost.predict(X_test_scaled))
    print(f"AdaBoost Accuracy: {adaboost_acc:.3f}")
    
    # XGBoost
    print("Training XGBoost...")
    xgboost_model = xgb.XGBClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    xgboost_model.fit(X_train_scaled, y_train)
    xgboost_acc = accuracy_score(y_test, xgboost_model.predict(X_test_scaled))
    print(f"XGBoost Accuracy: {xgboost_acc:.3f}")
    
    # Create visualization
    print("\nCreating decision boundary plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # AdaBoost plot
    plot_decision_boundary(adaboost, X_train_scaled, y_train, 
                          'AdaBoost Decision Boundary\n(Stroke vs No Stroke)', ax1)
    
    # XGBoost plot
    plot_decision_boundary(xgboost_model, X_train_scaled, y_train, 
                          'XGBoost Decision Boundary\n(Stroke vs No Stroke)', ax2)
    
    plt.tight_layout()
    
    # Save and show
    plt.savefig('stroke_decision_boundaries.png', dpi=300, bbox_inches='tight')
    print("Plot saved as: stroke_decision_boundaries.png")
    
    plt.show()
    
    # Print results
    print("\nRESULTS SUMMARY")
    print("="*40)
    print(f"AdaBoost Test Accuracy: {adaboost_acc:.3f}")
    print(f"XGBoost Test Accuracy: {xgboost_acc:.3f}")
    print(f"Dataset: {len(X)} samples")
    print(f"Stroke Rate: {y.mean():.1%}")
    
    print("\nExperiment completed!")
    print("The colored regions show where each model predicts stroke (green) vs no stroke (red)")
    print("The black lines show the decision boundaries")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Install required packages:")
        print("   pip install numpy pandas matplotlib scikit-learn xgboost")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure all required packages are installed")
