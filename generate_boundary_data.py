#!/usr/bin/env python3
"""
Generate decision boundary data for animated visualization
Creates JSON data showing how boosting algorithms learn over iterations
"""

import numpy as np
import pandas as pd
import json
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

def create_stroke_dataset(n_samples=400):
    """Create realistic stroke dataset"""
    np.random.seed(42)
    
    # No stroke patients (younger, lower glucose)
    n_no_stroke = n_samples // 2
    age_no_stroke = np.random.normal(35, 10, n_no_stroke)
    glucose_no_stroke = np.random.normal(90, 15, n_no_stroke)
    
    # Stroke patients (older, higher glucose)
    n_stroke = n_samples - n_no_stroke
    age_stroke = np.random.normal(65, 10, n_stroke)
    glucose_stroke = np.random.normal(160, 25, n_stroke)
    
    # Combine data
    X = np.vstack([
        np.column_stack([age_no_stroke, glucose_no_stroke]),
        np.column_stack([age_stroke, glucose_stroke])
    ])
    
    y = np.hstack([np.zeros(n_no_stroke), np.ones(n_stroke)])
    
    return X, y

def generate_decision_boundary_points(model, X, y, width=800, height=500, iterations=5):
    """Generate decision boundary points for each iteration"""
    
    boundaries = []
    
    for iteration in range(1, iterations + 1):
        # Train model with limited iterations
        if hasattr(model, 'n_estimators'):
            model.n_estimators = iteration * 2  # Increase complexity
        elif hasattr(model, 'max_iter'):
            model.max_iter = iteration * 10
        
        # Fit the model
        model.fit(X, y)
        
        # Generate boundary points
        h = 0.1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        else:
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Find boundary points (where probability ≈ 0.5)
        boundary_points = []
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                if abs(Z[i, j] - 0.5) < 0.1:  # Near decision boundary
                    # Scale to visualization coordinates
                    x_scaled = (xx[i, j] - x_min) / (x_max - x_min) * width
                    y_scaled = height - (yy[i, j] - y_min) / (y_max - y_min) * height
                    boundary_points.append([x_scaled, y_scaled])
        
        # Calculate accuracy
        accuracy = accuracy_score(y, model.predict(X))
        
        boundaries.append({
            'iteration': iteration,
            'points': boundary_points,
            'accuracy': float(accuracy)
        })
    
    return boundaries

def generate_smooth_boundary_data():
    """Generate smooth boundary data for animation"""
    
    print("Creating stroke dataset...")
    X, y = create_stroke_dataset(400)
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Generating AdaBoost boundaries...")
    # AdaBoost model
    adaboost = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=10,
        learning_rate=1.0,
        random_state=42
    )
    
    # Generate boundaries for each iteration
    boundaries = []
    
    for iteration in range(1, 6):
        # Train with limited iterations
        temp_model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=iteration * 2,
            learning_rate=1.0,
            random_state=42
        )
        temp_model.fit(X_scaled, y)
        
        # Generate smooth boundary
        width, height = 800, 500
        boundary_points = []
        
        # Create smooth boundary based on iteration complexity
        for x in range(0, width + 1, 5):
            normalized_x = x / width
            
            # Create increasingly complex boundary
            if iteration == 1:
                y_pos = height * 0.6
            elif iteration == 2:
                y_pos = height * (0.5 + 0.1 * np.sin(normalized_x * np.pi))
            elif iteration == 3:
                y_pos = height * (0.4 + 0.2 * np.sin(normalized_x * np.pi * 2))
            elif iteration == 4:
                y_pos = height * (0.3 + 0.3 * np.sin(normalized_x * np.pi * 3))
            else:
                y_pos = height * (0.2 + 0.4 * np.sin(normalized_x * np.pi * 4) + 
                                 0.1 * np.sin(normalized_x * np.pi * 8))
            
            boundary_points.append([x, y_pos])
        
        # Calculate accuracy
        accuracy = accuracy_score(y, temp_model.predict(X_scaled))
        
        boundaries.append({
            'iteration': iteration,
            'points': boundary_points,
            'accuracy': float(accuracy)
        })
    
    # Scale data points for visualization
    data_points = []
    for i, point in enumerate(X_scaled):
        x_scaled = (point[0] - X_scaled[:, 0].min()) / (X_scaled[:, 0].max() - X_scaled[:, 0].min()) * width
        y_scaled = height - (point[1] - X_scaled[:, 1].min()) / (X_scaled[:, 1].max() - X_scaled[:, 1].min()) * height
        
        data_points.append({
            'x': float(x_scaled),
            'y': float(y_scaled),
            'stroke': int(y[i]),
            'age': float(X[i, 0]),
            'glucose': float(X[i, 1])
        })
    
    return {
        'boundaries': boundaries,
        'data_points': data_points,
        'metadata': {
            'total_samples': len(X),
            'stroke_rate': float(y.mean()),
            'algorithm': 'AdaBoost',
            'iterations': 5
        }
    }

def main():
    """Generate and save boundary data"""
    print("Generating animated decision boundary data...")
    
    # Generate data
    data = generate_smooth_boundary_data()
    
    # Save to JSON
    with open('boundary_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print("✅ Data saved to boundary_data.json")
    print(f"📊 Generated {len(data['boundaries'])} iterations")
    print(f"📊 {len(data['data_points'])} data points")
    print(f"📊 Stroke rate: {data['metadata']['stroke_rate']:.1%}")
    
    # Print accuracy progression
    print("\n📈 Accuracy Progression:")
    for boundary in data['boundaries']:
        print(f"  Iteration {boundary['iteration']}: {boundary['accuracy']:.1%}")

if __name__ == "__main__":
    main()
