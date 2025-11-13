from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import pickle
import json
import asyncio
import os
import io
import base64
from typing import Dict, Any, Optional
import socketio
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime

# Import matplotlib for real visualizations
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns

# Import ML libraries with fallbacks
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Helper function to convert numpy types to native Python types for JSON serialization
def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def generate_decision_boundary_plot(stage="initial"):
    """Generate high-quality decision boundary plots that match the dataset"""
    try:
        # Set dark theme style
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a2e')
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        
        # Generate realistic Titanic-like data
        np.random.seed(42)
        n_samples = 200
        
        # Create realistic passenger data
        pclass = np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5])  # More 3rd class
        age = np.random.normal(35, 15, n_samples)  # Age distribution
        age = np.clip(age, 5, 80)  # Reasonable age range
        
        # Create survival probability based on realistic patterns
        survival_prob = np.zeros(n_samples)
        for i in range(n_samples):
            prob = 0.3  # Base survival rate
            
            # Higher class = higher survival
            if pclass[i] == 1:
                prob += 0.4
            elif pclass[i] == 2:
                prob += 0.2
            
            # Children and women more likely to survive
            if age[i] < 16:
                prob += 0.3
            elif age[i] > 60:
                prob -= 0.2
            
            # Add some randomness
            prob += np.random.normal(0, 0.1)
            survival_prob[i] = np.clip(prob, 0, 1)
        
        # Generate actual survival labels
        survived = np.random.random(n_samples) < survival_prob
        
        # Create decision boundaries based on stage with bright colors
        if stage == "initial":
            # Simple vertical split at Pclass = 2.5
            ax.axvline(x=2.5, color='#60a5fa', linestyle='--', linewidth=4, alpha=1.0)
            ax.text(2.6, 75, 'Pclass ≤ 2.5', fontsize=12, color='#ffffff', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='#1e293b', alpha=0.9, edgecolor='#60a5fa', linewidth=2))
            
        elif stage == "second_tree":
            # Add horizontal split at Age = 60
            ax.axvline(x=2.5, color='#60a5fa', linestyle='--', linewidth=4, alpha=1.0)
            ax.axhline(y=60, color='#fbbf24', linestyle='--', linewidth=4, alpha=1.0)
            ax.text(2.6, 75, 'Pclass ≤ 2.5', fontsize=12, color='#ffffff', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='#1e293b', alpha=0.9, edgecolor='#60a5fa', linewidth=2))
            ax.text(1.2, 62, 'Age ≤ 60', fontsize=12, color='#ffffff', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='#1e293b', alpha=0.9, edgecolor='#fbbf24', linewidth=2))
            
        else:  # final_ensemble
            # Multiple decision boundaries
            ax.axvline(x=2.5, color='#60a5fa', linestyle='--', linewidth=4, alpha=1.0)
            ax.axhline(y=60, color='#fbbf24', linestyle='--', linewidth=4, alpha=1.0)
            ax.axhline(y=35, color='#34d399', linestyle='--', linewidth=4, alpha=1.0)
            ax.text(2.6, 75, 'Pclass ≤ 2.5', fontsize=12, color='#ffffff', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='#1e293b', alpha=0.9, edgecolor='#60a5fa', linewidth=2))
            ax.text(1.2, 62, 'Age ≤ 60', fontsize=12, color='#ffffff', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='#1e293b', alpha=0.9, edgecolor='#fbbf24', linewidth=2))
            ax.text(1.2, 37, 'Age ≤ 35', fontsize=12, color='#ffffff', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='#1e293b', alpha=0.9, edgecolor='#34d399', linewidth=2))
        
        # Plot data points with bright colors for dark theme
        colors = ['#f87171' if not s else '#4ade80' for s in survived]
        scatter = ax.scatter(pclass, age, c=colors, alpha=0.8, s=60, edgecolors='#ffffff', linewidth=1)
        
        # Set dark theme styling with bright text
        ax.set_xlabel('Passenger Class (Pclass)', fontsize=14, fontweight='bold', color='#ffffff')
        ax.set_ylabel('Age (years)', fontsize=14, fontweight='bold', color='#ffffff')
        ax.set_title(f'Decision Boundary Analysis - {stage.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold', pad=20, color='#ffffff')
        
        # Set axis colors for dark theme
        ax.tick_params(colors='#ffffff', labelsize=11)
        ax.spines['bottom'].set_color('#ffffff')
        ax.spines['top'].set_color('#ffffff')
        ax.spines['right'].set_color('#ffffff')
        ax.spines['left'].set_color('#ffffff')
        
        # Set axis limits and ticks
        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(0, 85)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['1st Class', '2nd Class', '3rd Class'], color='#ffffff')
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='#ffffff')
        
        # Add professional legend with dark theme
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4ade80', label='Survived (1)', alpha=0.8),
            Patch(facecolor='#f87171', label='Not Survived (0)', alpha=0.8)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
                 frameon=True, fancybox=True, shadow=True, facecolor='#1e293b', 
                 edgecolor='#60a5fa', labelcolor='#ffffff')
        
        # Add statistics text box with dark theme
        survival_rate = np.mean(survived) * 100
        stats_text = f'Survival Rate: {survival_rate:.1f}%\nTotal Passengers: {n_samples}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', color='#ffffff',
               bbox=dict(boxstyle="round,pad=0.5", 
               facecolor='#1e293b', alpha=0.9, edgecolor='#60a5fa', linewidth=2))
        
        # Convert to high-quality base64 with dark background
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='#1a1a2e', edgecolor='none')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return plot_data
        
    except Exception as e:
        print(f"Error generating plot: {e}")
        return None

def make_grid(X, padding=0.1, steps=400):
    """Create dense grid for decision boundary visualization"""
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    dx = (x_max - x_min) * padding
    dy = (y_max - y_min) * padding
    xs = np.linspace(x_min - dx, x_max + dx, steps)
    ys = np.linspace(y_min - dy, y_max + dy, steps)
    grid_x, grid_y = np.meshgrid(xs, ys[::-1])  # reverse for display orientation
    coords = np.c_[grid_x.ravel(), grid_y.ravel()]
    return grid_x, grid_y, coords

def export_probability_grid(model, X_2d, steps=400):
    """Export high-accuracy probability grid for decision boundary visualization"""
    try:
        grid_x, grid_y, coords_2d = make_grid(X_2d, steps=steps)
        
        # Compute probabilities (class 1)
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(coords_2d)[:, 1]  # shape (steps*steps,)
        else:
            # Fallback for models without predict_proba
            predictions = model.predict(coords_2d)
            probs = predictions.astype(float)
        
        probs2d = probs.reshape(grid_x.shape)
        
        return grid_x, grid_y, probs2d
        
    except Exception as e:
        print(f"Error computing probability grid: {e}")
        return None, None, None

def generate_boosting_decision_boundary(algorithm="adaboost", n_estimators=1, stage="early"):
    """Simple scatter plot with decision boundary showing stroke vs no-stroke"""
    try:
        # Dark theme matplotlib setup
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a2e')
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        
        # Use the actual loaded dataset
        if data_store['X_train'] is None or data_store['y_train'] is None:
            raise Exception("No dataset loaded. Please load a dataset first.")
        
        # Get the actual training data
        X_train = data_store['X_train']
        y_train = data_store['y_train']
        
        # Select two features for visualization
        feature_names = data_store.get('feature_names', [])
        categorical_features = data_store.get('categorical_features', [])
        numerical_features = data_store.get('numerical_features', [])
        
        # Ensure we only use numerical features for visualization
        # Get indices of numerical features
        numerical_indices = [i for i, name in enumerate(feature_names) if name in numerical_features]
        
        # Pick the best two features (prefer numerical features)
        if 'age' in feature_names and 'avg_glucose_level' in feature_names:
            age_idx = feature_names.index('age')
            glucose_idx = feature_names.index('avg_glucose_level')
            # Verify these are numerical
            if age_idx in numerical_indices and glucose_idx in numerical_indices:
                X_2d = X_train.iloc[:, [age_idx, glucose_idx]].values
                x_label, y_label = 'Age', 'Average Glucose Level'
            else:
                # Fall back to first two numerical features
                if len(numerical_indices) >= 2:
                    X_2d = X_train.iloc[:, [numerical_indices[0], numerical_indices[1]]].values
                    x_label, y_label = feature_names[numerical_indices[0]], feature_names[numerical_indices[1]]
                else:
                    raise Exception("Not enough numerical features for visualization")
        elif 'age' in feature_names and 'bmi' in feature_names:
            age_idx = feature_names.index('age')
            bmi_idx = feature_names.index('bmi')
            # Verify these are numerical
            if age_idx in numerical_indices and bmi_idx in numerical_indices:
                X_2d = X_train.iloc[:, [age_idx, bmi_idx]].values
                x_label, y_label = 'Age', 'BMI'
            else:
                # Fall back to first two numerical features
                if len(numerical_indices) >= 2:
                    X_2d = X_train.iloc[:, [numerical_indices[0], numerical_indices[1]]].values
                    x_label, y_label = feature_names[numerical_indices[0]], feature_names[numerical_indices[1]]
                else:
                    raise Exception("Not enough numerical features for visualization")
        elif 'CreditScore' in feature_names and 'Age' in feature_names:
            # Churn dataset: use CreditScore and Age
            credit_idx = feature_names.index('CreditScore')
            age_idx = feature_names.index('Age')
            if credit_idx in numerical_indices and age_idx in numerical_indices:
                X_2d = X_train.iloc[:, [credit_idx, age_idx]].values
                x_label, y_label = 'Credit Score', 'Age'
            else:
                # Fall back to first two numerical features
                if len(numerical_indices) >= 2:
                    X_2d = X_train.iloc[:, [numerical_indices[0], numerical_indices[1]]].values
                    x_label, y_label = feature_names[numerical_indices[0]], feature_names[numerical_indices[1]]
                else:
                    raise Exception("Not enough numerical features for visualization")
        else:
            # Use first two NUMERICAL features (not categorical)
            if len(numerical_indices) >= 2:
                X_2d = X_train.iloc[:, [numerical_indices[0], numerical_indices[1]]].values
                x_label, y_label = feature_names[numerical_indices[0]], feature_names[numerical_indices[1]]
            elif len(numerical_indices) == 1:
                # Only one numerical feature, use it twice (not ideal but better than error)
                X_2d = X_train.iloc[:, [numerical_indices[0], numerical_indices[0]]].values
                x_label, y_label = feature_names[numerical_indices[0]], feature_names[numerical_indices[0]]
            else:
                raise Exception("No numerical features available for visualization")
        
        # Get labels
        y_labels = y_train.values
        
        # IMPORTANT: Model trains on ALL data (no sampling limit)
        # This allows training with any number of samples for better decision boundaries
        # Train a simple model
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_2d, y_labels)  # Uses all available training samples
        
        # Create a grid for decision boundary
        h = 0.5
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Get predictions for the grid
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary with bright colors for dark theme
        from matplotlib.colors import ListedColormap
        cmap_light = ListedColormap(['#3b82f6', '#10b981'])
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
        ax.contour(xx, yy, Z, colors='#ffffff', linewidths=3, alpha=0.9)
        
        # Plot the actual data points
        # For large datasets, sample points for visualization clarity, but train on all data
        stroke_points = X_2d[y_labels == 1]
        no_stroke_points = X_2d[y_labels == 0]
        
        # Balance visualization samples - take equal numbers from each class
        # This ensures good visualization even with imbalanced original data
        max_viz_points = 2000
        min_class_viz = min(len(stroke_points), len(no_stroke_points), max_viz_points)
        
        # Sample equal numbers from each class for balanced visualization
        if len(stroke_points) > min_class_viz:
            stroke_points_viz = stroke_points[np.random.choice(len(stroke_points), min_class_viz, replace=False)]
        else:
            stroke_points_viz = stroke_points
            
        if len(no_stroke_points) > min_class_viz:
            no_stroke_points_viz = no_stroke_points[np.random.choice(len(no_stroke_points), min_class_viz, replace=False)]
        else:
            no_stroke_points_viz = no_stroke_points
        
        # Ensure balanced visualization (same count for both classes)
        final_viz_count = min(len(stroke_points_viz), len(no_stroke_points_viz))
        if len(stroke_points_viz) > final_viz_count:
            stroke_points_viz = stroke_points_viz[np.random.choice(len(stroke_points_viz), final_viz_count, replace=False)]
        if len(no_stroke_points_viz) > final_viz_count:
            no_stroke_points_viz = no_stroke_points_viz[np.random.choice(len(no_stroke_points_viz), final_viz_count, replace=False)]
        
        # Determine label names based on dataset
        # Check if this is churn dataset by looking at feature names or y_train values
        is_churn_dataset = 'CreditScore' in feature_names or 'Geography' in feature_names
        class0_label = 'No Churn' if is_churn_dataset else 'No Stroke'
        class1_label = 'Churn' if is_churn_dataset else 'Stroke'
        
        ax.scatter(no_stroke_points_viz[:, 0], no_stroke_points_viz[:, 1], 
                  c='#f87171', label=f'{class0_label} (showing {len(no_stroke_points_viz)} of {len(no_stroke_points)})', 
                  alpha=0.9, s=60, edgecolors='#ffffff', linewidth=1.5)
        ax.scatter(stroke_points_viz[:, 0], stroke_points_viz[:, 1], 
                  c='#4ade80', label=f'{class1_label} (showing {len(stroke_points_viz)} of {len(stroke_points)})', 
                  alpha=0.9, s=60, edgecolors='#ffffff', linewidth=1.5)
        
        # Labels and title with bright colors
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold', color='#ffffff')
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold', color='#ffffff')
        ax.set_title(f'{algorithm.title()} Decision Boundary (n_estimators={n_estimators})', 
                    fontsize=14, fontweight='bold', color='#ffffff', pad=15)
        
        # Set axis colors
        ax.tick_params(colors='#ffffff', labelsize=10)
        ax.spines['bottom'].set_color('#ffffff')
        ax.spines['top'].set_color('#ffffff')
        ax.spines['right'].set_color('#ffffff')
        ax.spines['left'].set_color('#ffffff')
        
        # Legend with dark theme
        legend = ax.legend(facecolor='#1e293b', edgecolor='#60a5fa', framealpha=0.9, 
                          fontsize=11, labelcolor='#ffffff', frameon=True)
        ax.grid(True, alpha=0.2, color='#ffffff', linewidth=0.5)
        
        # Convert to base64 with dark background
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='#1a1a2e', edgecolor='none')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return plot_data
        
    except Exception as e:
        print(f"Error generating decision boundary: {e}")
        return None

def generate_3d_decision_boundary(algorithm="adaboost", n_estimators=1, stage="early"):
    """Generate 3D decision boundary plots using actual trained model and dataset"""
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        # Use dark theme matplotlib styling
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 12), facecolor='#1a1a2e')
        fig.patch.set_facecolor('#1a1a2e')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#1a1a2e')
        
        # Use the actual loaded dataset
        if data_store['X_train'] is None or data_store['y_train'] is None:
            raise Exception("No dataset loaded. Please load a dataset first.")
        
        # Get the actual training data
        X_train = data_store['X_train']
        y_train = data_store['y_train']
        
        # Select the three most important features for 3D visualization
        feature_names = data_store.get('feature_names', [])
        
        # Find the best features for 3D visualization
        if all(feat in feature_names for feat in ['age', 'avg_glucose_level', 'bmi']):
            age_idx = feature_names.index('age')
            glucose_idx = feature_names.index('avg_glucose_level')
            bmi_idx = feature_names.index('bmi')
            X_3d = X_train.iloc[:, [age_idx, glucose_idx, bmi_idx]].values
            feature_labels = ['Age (years)', 'Average Glucose Level (mg/dL)', 'BMI (kg/m²)']
        else:
            # Fallback to first three numerical features
            numerical_features = data_store.get('numerical_features', [])
            if len(numerical_features) >= 3:
                feat1_idx = feature_names.index(numerical_features[0])
                feat2_idx = feature_names.index(numerical_features[1])
                feat3_idx = feature_names.index(numerical_features[2])
                X_3d = X_train.iloc[:, [feat1_idx, feat2_idx, feat3_idx]].values
                feature_labels = [numerical_features[0], numerical_features[1], numerical_features[2]]
            else:
                # Use first three features as fallback
                X_3d = X_train.iloc[:, [0, 1, 2]].values
                feature_labels = [feature_names[0], feature_names[1], feature_names[2]]
        
        # Use actual labels
        y_labels = y_train.values
        
        # Create a model based on the stage
        if algorithm == "adaboost":
            from sklearn.ensemble import AdaBoostClassifier
            if stage == "early" or n_estimators <= 2:
                model = AdaBoostClassifier(n_estimators=2, random_state=42)
                boundary_label = f'Early AdaBoost (n_estimators = 2)'
            elif stage == "mid" or n_estimators <= 5:
                model = AdaBoostClassifier(n_estimators=5, random_state=42)
                boundary_label = f'Mid AdaBoost (n_estimators = 5)'
            else:
                model = AdaBoostClassifier(n_estimators=10, random_state=42)
                boundary_label = f'Late AdaBoost (n_estimators = 10)'
        else:
            # Default to AdaBoost
            from sklearn.ensemble import AdaBoostClassifier
            model = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
            boundary_label = f'{algorithm.title()} (n_estimators = {n_estimators})'
        
        # Train the model on the 3D features
        model.fit(X_3d, y_labels)
        
        # Create 3D scatter plot with bright colors
        colors = ['#f87171' if not s else '#4ade80' for s in y_labels]
        scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=colors, alpha=0.8, s=70, 
                           edgecolors='#ffffff', linewidth=1)
        
        # Set labels and title with bright colors
        ax.set_xlabel(feature_labels[0], fontsize=14, fontweight='bold', color='#ffffff')
        ax.set_ylabel(feature_labels[1], fontsize=14, fontweight='bold', color='#ffffff')
        ax.set_zlabel(feature_labels[2], fontsize=14, fontweight='bold', color='#ffffff')
        ax.set_title(f'3D {algorithm.title()} Decision Boundary - {boundary_label}', 
                    fontsize=16, fontweight='bold', pad=20, color='#ffffff')
        
        # Set tick colors
        ax.tick_params(colors='#ffffff', labelsize=11)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#ffffff')
        ax.yaxis.pane.set_edgecolor('#ffffff')
        ax.zaxis.pane.set_edgecolor('#ffffff')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        # Add legend with dark theme
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4ade80', label='Stroke (1)', alpha=0.9),
            Patch(facecolor='#f87171', label='No Stroke (0)', alpha=0.9)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12,
                 facecolor='#1e293b', edgecolor='#60a5fa', labelcolor='#ffffff', framealpha=0.9)
        
        # Add statistics
        stroke_rate = np.mean(y_labels) * 100
        model_accuracy = model.score(X_3d, y_labels)
        stats_text = f'Stroke Rate: {stroke_rate:.1f}%\nTotal Patients: {len(y_labels)}\nModel Accuracy: {model_accuracy:.1%}'
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', color='#ffffff',
                 bbox=dict(boxstyle="round,pad=0.5", 
                 facecolor='#1e293b', alpha=0.9, edgecolor='#60a5fa', linewidth=2))
        
        # Convert to high-quality base64 with dark background
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='#1a1a2e', edgecolor='none')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return plot_data
        
    except Exception as e:
        print(f"Error generating 3D decision boundary: {e}")
        return None

# Global variables for data and models
data_store = {
    'X_train': None,
    'X_test': None,
    'y_train': None,
    'y_test': None,
    'feature_names': None,
    'categorical_features': None,
    'numerical_features': None,
    'dataset': None,
    'model_trained': False,  # Track if a model has been trained
    'trained_model': None,  # Store the trained model
    'trained_algorithm': None  # Store which algorithm was trained
}

# Socket.IO server
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode='asgi')
app = FastAPI(title="Boosting Algorithms Demo API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create Socket.IO app
socket_app = socketio.ASGIApp(sio, app)

# Try to mount static files (for serving React build)
import os
from fastapi.responses import FileResponse

static_path = os.path.join(os.path.dirname(__file__), 'static')
serve_static = False

# Real training data endpoints
@app.get("/training-data/{algorithm}")
async def get_training_data(algorithm: str):
    """Get real training data for visualization"""
    try:
        if data_store['X_train'] is None or data_store['y_train'] is None:
            raise Exception("No dataset loaded")
        
        X_train = data_store['X_train']
        y_train = data_store['y_train']
        feature_names = data_store.get('feature_names', [])
        
        return {
            "X_train": X_train.values.tolist(),
            "y_train": y_train.values.tolist(),
            "feature_names": feature_names,
            "algorithm": algorithm
        }
        
    except Exception as e:
        print(f"Error getting training data: {e}")
        return {"error": str(e)}

@app.get("/model-predictions/{algorithm}")
async def get_model_predictions(algorithm: str, iteration: int = 0):
    """Get real model predictions for specific iteration"""
    try:
        if data_store['X_train'] is None or data_store['y_train'] is None:
            raise Exception("No dataset loaded")
        
        X_train = data_store['X_train']
        y_train = data_store['y_train']
        feature_names = data_store.get('feature_names', [])
        
        # Select two features for visualization
        if 'age' in feature_names and 'avg_glucose_level' in feature_names:
            age_idx = feature_names.index('age')
            glucose_idx = feature_names.index('avg_glucose_level')
            X_2d = X_train.iloc[:, [age_idx, glucose_idx]]
        else:
            X_2d = X_train.iloc[:, [0, 1]]
        
        # Train model with specific number of iterations
        if algorithm.lower() == 'adaboost':
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.tree import DecisionTreeClassifier
            model = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=iteration + 1,
                learning_rate=1.0,
                random_state=42
            )
        elif algorithm.lower() == 'xgboost':
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=iteration + 1,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=iteration + 1,
                learning_rate=0.1,
                random_state=42
            )
        
        # Fit the model
        model.fit(X_2d, y_train)
        
        # Get predictions
        predictions = model.predict(X_2d)
        accuracy = model.score(X_2d, y_train)
        
        # Generate decision boundary data
        h = 0.5
        x_min, x_max = X_2d.iloc[:, 0].min() - 1, X_2d.iloc[:, 0].max() + 1
        y_min, y_max = X_2d.iloc[:, 1].min() - 1, X_2d.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Get predictions for grid
        if hasattr(model, 'predict_proba'):
            Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        else:
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Create boundary path
        boundary_points = []
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                if abs(Z[i, j] - 0.5) < 0.1:  # Near decision boundary
                    boundary_points.append([xx[i, j], yy[i, j]])
        
        # Create SVG path
        if boundary_points:
            boundary_path = f"M {boundary_points[0][0]},{boundary_points[0][1]}"
            for point in boundary_points[1:]:
                boundary_path += f" L {point[0]},{point[1]}"
        else:
            boundary_path = ""
        
        return {
            "predictions": predictions.tolist(),
            "accuracy": float(accuracy),
            "decision_boundary_data": {
                "boundary_path": boundary_path,
                "grid_x": xx.tolist(),
                "grid_y": yy.tolist(),
                "predictions_grid": Z.tolist()
            },
            "iteration": iteration,
            "algorithm": algorithm
        }
        
    except Exception as e:
        print(f"Error getting model predictions: {e}")
        return {"error": str(e)}

if os.path.exists(static_path):
    try:
        app.mount("/static", StaticFiles(directory=static_path), name="static")
        serve_static = True
        print("Static files directory found and mounted")
    except Exception as e:
        print(f"Warning: Could not mount static files: {e}")
        serve_static = False

# Load initial dataset
def load_default_dataset():
    """Load the default stroke dataset"""
    try:
        # Try multiple paths for the dataset
        dataset_paths = [
            'data/stroke_data_balanced.csv',
            '../data/stroke_data_balanced.csv',
            './stroke_data_balanced.csv'
        ]
        
        df = None
        for path in dataset_paths:
            try:
                df = pd.read_csv(path)
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            raise FileNotFoundError("Could not find stroke_data_balanced.csv in any expected location")
        print(f"Loaded dataset with shape: {df.shape}")
        
        # Data preparation (reuse logic from existing scripts)
        X = df.drop(['id', 'stroke'], axis=1)
        y = df['stroke']
        
        # Identify features
        categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        numerical_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
        
        # Encode categorical variables
        X_encoded = X.copy()
        for feature in categorical_features:
            le = LabelEncoder()
            X_encoded[feature] = le.fit_transform(X_encoded[feature].astype(str))
        
        # Balance classes before splitting to ensure equal representation
        # This prevents imbalanced datasets (e.g., 500 no-stroke vs 100 stroke)
        stroke_indices = y[y == 1].index
        no_stroke_indices = y[y == 0].index
        
        # Find the smaller class and balance to that size
        min_class_size = min(len(stroke_indices), len(no_stroke_indices))
        
        if min_class_size == 0:
            print("Warning: One class has zero samples. Using all available data.")
            balanced_indices = np.concatenate([stroke_indices, no_stroke_indices])
        else:
            # Sample equal numbers from each class
            balanced_stroke_indices = np.random.choice(stroke_indices, min_class_size, replace=False)
            balanced_no_stroke_indices = np.random.choice(no_stroke_indices, min_class_size, replace=False)
            
            # Combine balanced indices
            balanced_indices = np.concatenate([balanced_stroke_indices, balanced_no_stroke_indices])
        
        np.random.shuffle(balanced_indices)  # Shuffle for randomness
        
        # Create balanced dataset
        X_balanced = X_encoded.loc[balanced_indices].reset_index(drop=True)
        y_balanced = y.loc[balanced_indices].reset_index(drop=True)
        
        print(f"Original dataset: {len(y)} samples ({y.sum()} stroke, {len(y) - y.sum()} no-stroke)")
        print(f"Balanced dataset: {len(y_balanced)} samples ({y_balanced.sum()} stroke, {len(y_balanced) - y_balanced.sum()} no-stroke)")
        
        # Split balanced data
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
        )
        
        # Store in global data_store
        data_store.update({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X_encoded.columns.tolist(),
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'dataset': df
        })
        
        return True
    except Exception as e:
        print(f"Error loading default dataset: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_default_dataset()
    yield
    # Shutdown

app.router.lifespan_context = lifespan

# Load dataset on startup
load_default_dataset()

# REST API endpoints - Root endpoint will be added later if not serving static files

@app.get("/algorithms")
async def get_available_algorithms():
    """Get list of available boosting algorithms"""
    algorithms = {
        "adaboost": "AdaBoost Classifier",
        "gradient_boosting": "Gradient Boosting Classifier",
        "xgboost": "XGBoost Classifier" if XGBOOST_AVAILABLE else None,
        "lightgbm": "LightGBM Classifier" if LIGHTGBM_AVAILABLE else None,
        "catboost": "CatBoost Classifier" if CATBOOST_AVAILABLE else None
    }
    
    # Filter out None values (unavailable algorithms)
    available = {k: v for k, v in algorithms.items() if v is not None}
    return {"algorithms": available}

@app.get("/dataset/preview")
async def get_dataset_preview(rows: int = 10, dataset: str = "stroke"):
    """Get first N rows of the dataset"""
    try:
        # Load the requested dataset
        df = load_dataset_by_name(dataset)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found")
        
        preview = df.head(rows)
        return {
            "data": preview.to_dict('records'),
            "columns": preview.columns.tolist(),
            "shape": df.shape,
            "dataset_name": dataset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")

def load_dataset_by_name(dataset_name: str):
    """Load a dataset by name (for preview only)"""
    try:
        if dataset_name == "stroke":
            # Load stroke dataset
            dataset_paths = [
                'data/stroke_data_balanced.csv',
                '../data/stroke_data_balanced.csv',
                './stroke_data_balanced.csv'
            ]
            for path in dataset_paths:
                try:
                    return pd.read_csv(path)
                except FileNotFoundError:
                    continue
            return None
        elif dataset_name == "churn" or dataset_name == "boosting_small":
            # Load churn/boosting small dataset
            dataset_paths = [
                'boosting_small_dataset.csv',
                'data/boosting_small_dataset.csv',
                '../boosting_small_dataset.csv'
            ]
            for path in dataset_paths:
                try:
                    return pd.read_csv(path)
                except FileNotFoundError:
                    continue
            return None
        else:
            return None
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None

def load_dataset_for_training(dataset_name: str, n_samples: int = None):
    """Load and prepare a dataset for training
    
    Args:
        dataset_name: Name of the dataset to load
        n_samples: Number of samples to use (None means use all available balanced data)
                   If specified, samples balanced data (equal from each class when possible)
    """
    try:
        # Clear previous data to ensure we start fresh
        data_store['X_train'] = None
        data_store['X_test'] = None
        data_store['y_train'] = None
        data_store['y_test'] = None
        data_store['model_trained'] = False
        data_store['trained_model'] = None
        data_store['trained_algorithm'] = None
        
        df = load_dataset_by_name(dataset_name)
        if df is None:
            return False
        
        print(f"Loading dataset '{dataset_name}' with n_samples={n_samples}")
        
        if dataset_name == "stroke":
            # Stroke dataset processing
            X = df.drop(['id', 'stroke'], axis=1)
            y = df['stroke']
            
            categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
            numerical_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
            
            # Encode categorical variables
            X_encoded = X.copy()
            for feature in categorical_features:
                le = LabelEncoder()
                X_encoded[feature] = le.fit_transform(X_encoded[feature].astype(str))
            
            # Balance classes
            stroke_indices = y[y == 1].index
            no_stroke_indices = y[y == 0].index
            min_class_size = min(len(stroke_indices), len(no_stroke_indices))
            
            if min_class_size == 0:
                balanced_indices = np.concatenate([stroke_indices, no_stroke_indices])
            else:
                # If n_samples is specified, sample balanced data
                if n_samples is not None and n_samples > 0:
                    samples_per_class = n_samples // 2
                    # Don't exceed available samples per class
                    samples_per_class = min(samples_per_class, min_class_size)
                    if samples_per_class > 0:
                        balanced_stroke_indices = np.random.choice(stroke_indices, samples_per_class, replace=False)
                        balanced_no_stroke_indices = np.random.choice(no_stroke_indices, samples_per_class, replace=False)
                        balanced_indices = np.concatenate([balanced_stroke_indices, balanced_no_stroke_indices])
                    else:
                        balanced_indices = np.concatenate([stroke_indices, no_stroke_indices])
                else:
                    # Use all balanced data
                    balanced_stroke_indices = np.random.choice(stroke_indices, min_class_size, replace=False)
                    balanced_no_stroke_indices = np.random.choice(no_stroke_indices, min_class_size, replace=False)
                    balanced_indices = np.concatenate([balanced_stroke_indices, balanced_no_stroke_indices])
            
            np.random.shuffle(balanced_indices)
            X_balanced = X_encoded.loc[balanced_indices].reset_index(drop=True)
            y_balanced = y.loc[balanced_indices].reset_index(drop=True)
            
            # For small datasets, adjust test_size to ensure we have enough training data
            # If n_samples was specified, we already have the right amount
            if len(X_balanced) < 20:
                # Very small dataset - use 80% for training, 20% for test
                test_size = 0.2
            else:
                test_size = 0.2
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced, test_size=test_size, random_state=42, stratify=y_balanced
            )
            
            # Log the actual sizes for debugging
            print(f"Training set size: {len(X_train)} samples ({y_train.sum()} stroke, {len(y_train) - y_train.sum()} no-stroke)")
            print(f"Test set size: {len(X_test)} samples ({y_test.sum()} stroke, {len(y_test) - y_test.sum()} no-stroke)")
            
            # IMPORTANT: Only store the sampled training/test data, NOT the full dataset
            data_store.update({
                'X_train': X_train,  # This is the sampled subset (after train_test_split)
                'X_test': X_test,   # This is the sampled test subset
                'y_train': y_train,  # Labels for sampled training data
                'y_test': y_test,   # Labels for sampled test data
                'feature_names': X_encoded.columns.tolist(),
                'categorical_features': categorical_features,
                'numerical_features': numerical_features,
                'dataset': X_balanced  # Store the balanced subset, not the full df
            })
            
            # Verify the stored data
            print(f"VERIFIED: data_store['X_train'] now has {len(data_store['X_train'])} samples")
            
        elif dataset_name == "churn" or dataset_name == "boosting_small":
            # Churn dataset processing
            X = df.drop(['CustomerID', 'Churn'], axis=1)
            y = df['Churn']
            
            # Identify categorical and numerical features
            categorical_features = []
            numerical_features = []
            
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)
            
            # Encode categorical variables
            X_encoded = X.copy()
            for feature in categorical_features:
                if feature in X_encoded.columns:
                    le = LabelEncoder()
                    X_encoded[feature] = le.fit_transform(X_encoded[feature].astype(str))
            
            # Ensure all columns are numeric (convert any remaining object types)
            for col in X_encoded.columns:
                if X_encoded[col].dtype == 'object':
                    # Try to convert to numeric, if fails then encode
                    try:
                        X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce')
                    except:
                        le = LabelEncoder()
                        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            
            # Verify all columns are numeric
            non_numeric_cols = [col for col in X_encoded.columns if X_encoded[col].dtype not in ['int64', 'float64', 'int32', 'float32']]
            if non_numeric_cols:
                print(f"Warning: Non-numeric columns after encoding: {non_numeric_cols}")
                # Force conversion
                for col in non_numeric_cols:
                    X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce').fillna(0)
            
            # Balance classes
            churn_indices = y[y == 1].index
            no_churn_indices = y[y == 0].index
            min_class_size = min(len(churn_indices), len(no_churn_indices))
            
            if min_class_size == 0:
                balanced_indices = np.concatenate([churn_indices, no_churn_indices])
            else:
                # If n_samples is specified, sample balanced data
                if n_samples is not None and n_samples > 0:
                    samples_per_class = n_samples // 2
                    # Don't exceed available samples per class
                    samples_per_class = min(samples_per_class, min_class_size)
                    if samples_per_class > 0:
                        balanced_churn_indices = np.random.choice(churn_indices, samples_per_class, replace=False)
                        balanced_no_churn_indices = np.random.choice(no_churn_indices, samples_per_class, replace=False)
                        balanced_indices = np.concatenate([balanced_churn_indices, balanced_no_churn_indices])
                    else:
                        balanced_indices = np.concatenate([churn_indices, no_churn_indices])
                else:
                    # Use all balanced data
                    balanced_churn_indices = np.random.choice(churn_indices, min_class_size, replace=False)
                    balanced_no_churn_indices = np.random.choice(no_churn_indices, min_class_size, replace=False)
                    balanced_indices = np.concatenate([balanced_churn_indices, balanced_no_churn_indices])
            
            np.random.shuffle(balanced_indices)
            X_balanced = X_encoded.loc[balanced_indices].reset_index(drop=True)
            y_balanced = y.loc[balanced_indices].reset_index(drop=True)
            
            # For very small datasets (< 10 rows), use all data for training
            if len(X_balanced) < 10:
                # Use all data for training, create minimal test set
                X_train = X_balanced
                y_train = y_balanced
                X_test = X_balanced.head(1) if len(X_balanced) > 0 else X_balanced
                y_test = y_balanced.head(1) if len(y_balanced) > 0 else y_balanced
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
                )
            
            # IMPORTANT: Only store the sampled training/test data, NOT the full dataset
            data_store.update({
                'X_train': X_train,  # This is the sampled subset (after train_test_split)
                'X_test': X_test,   # This is the sampled test subset
                'y_train': y_train,  # Labels for sampled training data
                'y_test': y_test,   # Labels for sampled test data
                'feature_names': X_encoded.columns.tolist(),
                'categorical_features': categorical_features,
                'numerical_features': numerical_features,
                'dataset': X_balanced  # Store the balanced subset, not the full df
            })
            
            # Verify the stored data
            print(f"VERIFIED: data_store['X_train'] now has {len(data_store['X_train'])} samples")
        
        print(f"Loaded dataset '{dataset_name}' for training - Final training set: {len(data_store['X_train'])} samples")
        return True
        
    except Exception as e:
        print(f"Error loading dataset for training: {e}")
        return False

@app.get("/plot/decision-boundary")
async def get_decision_boundary_plot(stage: str = "initial"):
    """Generate and return high-quality decision boundary plots"""
    try:
        # Generate the high-quality plot
        plot_data = generate_decision_boundary_plot(stage)
        
        if plot_data is None:
            raise HTTPException(status_code=500, detail="Failed to generate plot")
        
        return {"plot_data": plot_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating plot: {str(e)}")

@app.get("/plot/boosting-boundary")
async def get_boosting_decision_boundary(algorithm: str = "adaboost", n_estimators: int = 1):
    """Generate 2D decision boundary plots for boosting algorithms showing improvement"""
    try:
        # Determine stage based on n_estimators
        if n_estimators <= 2:
            stage = "early"
        elif n_estimators <= 5:
            stage = "mid"
        else:
            stage = "late"
        
        # Generate the boosting decision boundary plot
        plot_data = generate_boosting_decision_boundary(algorithm, n_estimators, stage)
        
        if plot_data is None:
            raise HTTPException(status_code=500, detail="Failed to generate boosting boundary plot")
        
        return {
            "plot_data": plot_data,
            "algorithm": algorithm,
            "n_estimators": n_estimators,
            "stage": stage,
            "type": "2D"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating boosting boundary plot: {str(e)}")

@app.get("/plot/boosting-boundary-3d")
async def get_boosting_3d_decision_boundary(algorithm: str = "adaboost", n_estimators: int = 1):
    """Generate 3D decision boundary plots for boosting algorithms"""
    try:
        # Determine stage based on n_estimators
        if n_estimators <= 2:
            stage = "early"
        elif n_estimators <= 5:
            stage = "mid"
        else:
            stage = "late"
        
        # Generate the 3D boosting decision boundary plot
        plot_data = generate_3d_decision_boundary(algorithm, n_estimators, stage)
        
        if plot_data is None:
            raise HTTPException(status_code=500, detail="Failed to generate 3D boosting boundary plot")
        
        return {
            "plot_data": plot_data,
            "algorithm": algorithm,
            "n_estimators": n_estimators,
            "stage": stage,
            "type": "3D"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating 3D boosting boundary plot: {str(e)}")

@app.get("/plot/boosting-grid")
async def get_boosting_probability_grid(algorithm: str = "adaboost", n_estimators: int = 1, steps: int = 400):
    """Export probability grid data for frontend rendering"""
    try:
        # Determine stage based on n_estimators
        if n_estimators <= 2:
            stage = "early"
        elif n_estimators <= 5:
            stage = "mid"
        else:
            stage = "late"
        
        # Use the actual loaded dataset
        if data_store['X_train'] is None or data_store['y_train'] is None:
            raise HTTPException(status_code=400, detail="No dataset loaded. Please load a dataset first.")
        
        # Get the actual training data
        X_train = data_store['X_train']
        y_train = data_store['y_train']
        
        # Select the two most important features for 2D visualization
        feature_names = data_store.get('feature_names', [])
        
        # Find the best features for 2D visualization
        if 'age' in feature_names and 'avg_glucose_level' in feature_names:
            age_idx = feature_names.index('age')
            glucose_idx = feature_names.index('avg_glucose_level')
            X_2d = X_train.iloc[:, [age_idx, glucose_idx]].values
            feature_labels = ['Age', 'Average Glucose Level']
        elif 'age' in feature_names and 'bmi' in feature_names:
            age_idx = feature_names.index('age')
            bmi_idx = feature_names.index('bmi')
            X_2d = X_train.iloc[:, [age_idx, bmi_idx]].values
            feature_labels = ['Age', 'BMI']
        else:
            # Fallback to first two numerical features
            numerical_features = data_store.get('numerical_features', [])
            if len(numerical_features) >= 2:
                feat1_idx = feature_names.index(numerical_features[0])
                feat2_idx = feature_names.index(numerical_features[1])
                X_2d = X_train.iloc[:, [feat1_idx, feat2_idx]].values
                feature_labels = [numerical_features[0], numerical_features[1]]
            else:
                # Use first two features as fallback
                X_2d = X_train.iloc[:, [0, 1]].values
                feature_labels = [feature_names[0], feature_names[1]]
        
        # Use actual labels
        survived = y_train.values
        
        # Create model based on stage
        from sklearn.tree import DecisionTreeClassifier
        if stage == "early":
            model = DecisionTreeClassifier(max_depth=1, random_state=42)
        elif stage == "mid":
            model = DecisionTreeClassifier(max_depth=2, random_state=42)
        else:
            model = DecisionTreeClassifier(max_depth=3, random_state=42)
        
        model.fit(X_2d, survived)
        
        # Generate probability grid
        grid_x, grid_y, probs2d = export_probability_grid(model, X_2d, steps=steps)
        
        if probs2d is None:
            raise HTTPException(status_code=500, detail="Failed to generate probability grid")
        
        # Convert to JSON-serializable format
        grid_data = {
            "grid_x": grid_x.tolist(),
            "grid_y": grid_y.tolist(),
            "probabilities": probs2d.tolist(),
            "data_points": {
                "x": X_2d[:, 0].tolist(),
                "y": X_2d[:, 1].tolist(),
                "labels": survived.tolist()
            },
            "bounds": {
                "x_min": float(grid_x.min()),
                "x_max": float(grid_x.max()),
                "y_min": float(grid_y.min()),
                "y_max": float(grid_y.max())
            },
            "feature_labels": feature_labels,
            "algorithm": algorithm,
            "n_estimators": n_estimators,
            "stage": stage,
            "model_accuracy": float(model.score(X_2d, survived))
        }
        
        return grid_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating probability grid: {str(e)}")

@app.post("/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a new CSV dataset"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Read the uploaded file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Validate dataset has target column
        if 'stroke' not in df.columns:
            raise HTTPException(status_code=400, detail="Dataset must have 'stroke' column as target")
        
        # No limit on dataset size - use all available data for better training
        # Users can train with any number of samples
        
        # Store the new dataset
        data_store['dataset'] = df
        
        # Process the dataset
        X = df.drop(['id', 'stroke'] if 'id' in df.columns else ['stroke'], axis=1)
        y = df['stroke']
        
        categorical_features = []
        numerical_features = []
        
        for col in X.columns:
            if X[col].dtype == 'object':
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        # Encode categorical variables
        X_encoded = X.copy()
        for feature in categorical_features:
            le = LabelEncoder()
            X_encoded[feature] = le.fit_transform(X_encoded[feature].astype(str))
        
        # Balance classes before splitting to ensure equal representation
        # This prevents imbalanced datasets (e.g., 500 no-stroke vs 100 stroke)
        stroke_indices = y[y == 1].index
        no_stroke_indices = y[y == 0].index
        
        # Find the smaller class and balance to that size
        min_class_size = min(len(stroke_indices), len(no_stroke_indices))
        
        if min_class_size == 0:
            raise HTTPException(status_code=400, detail="Dataset must contain both stroke and no-stroke cases")
        
        # Sample equal numbers from each class
        balanced_stroke_indices = np.random.choice(stroke_indices, min_class_size, replace=False)
        balanced_no_stroke_indices = np.random.choice(no_stroke_indices, min_class_size, replace=False)
        
        # Combine balanced indices
        balanced_indices = np.concatenate([balanced_stroke_indices, balanced_no_stroke_indices])
        np.random.shuffle(balanced_indices)  # Shuffle for randomness
        
        # Create balanced dataset
        X_balanced = X_encoded.loc[balanced_indices].reset_index(drop=True)
        y_balanced = y.loc[balanced_indices].reset_index(drop=True)
        
        print(f"Original dataset: {len(y)} samples ({y.sum()} stroke, {len(y) - y.sum()} no-stroke)")
        print(f"Balanced dataset: {len(y_balanced)} samples ({y_balanced.sum()} stroke, {len(y_balanced) - y_balanced.sum()} no-stroke)")
        
        # Split balanced data
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
        )
        
        data_store.update({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X_encoded.columns.tolist(),
            'categorical_features': categorical_features,
            'numerical_features': numerical_features
        })
        
        return {
            "message": "Dataset uploaded successfully",
            "shape": df.shape,
            "columns": df.columns.tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {str(e)}")

@app.post("/predict")
async def predict_stroke(data: Dict[str, Any]):
    """Predict stroke for synthetic data based on dataset features"""
    try:
        # Check if a model has been trained
        if not data_store.get('model_trained', False):
            raise HTTPException(status_code=400, detail="Model hasn't been trained yet. Please train a model first before making predictions.")
        
        if data_store['X_train'] is None:
            raise HTTPException(status_code=404, detail="No dataset loaded")
        
        # Load the original dataset to get proper encoders
        dataset_paths = [
            'data/stroke_data_balanced.csv',
            '../data/stroke_data_balanced.csv',
            './stroke_data_balanced.csv'
        ]
        
        df = None
        for path in dataset_paths:
            try:
                df = pd.read_csv(path)
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            raise HTTPException(status_code=404, detail="Original dataset not found")
        
        # Prepare the input data
        input_data = {
            'gender': data.get('gender', 'Male'),
            'age': float(data.get('age', 45.0)),
            'hypertension': int(data.get('hypertension', 0)),
            'heart_disease': int(data.get('heart_disease', 0)),
            'ever_married': data.get('ever_married', 'Yes'),
            'work_type': data.get('work_type', 'Private'),
            'Residence_type': data.get('Residence_type', 'Urban'),
            'avg_glucose_level': float(data.get('avg_glucose_level', 95.0)),
            'bmi': float(data.get('bmi', 25.0)),
            'smoking_status': data.get('smoking_status', 'never smoked')
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Get the same preprocessing as the training data
        categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        numerical_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
        
        # Encode categorical variables (using the same logic as training)
        input_encoded = input_df.copy()
        for feature in categorical_features:
            if feature in input_encoded.columns:
                # Get unique values from original dataset for consistent encoding
                unique_values = sorted(df[feature].astype(str).unique())
                le = LabelEncoder()
                le.fit(unique_values)
                
                # Handle unseen categories
                if input_data[feature] in le.classes_:
                    input_encoded[feature] = le.transform([input_data[feature]])
                else:
                    # Default to first category if not seen
                    input_encoded[feature] = le.transform([unique_values[0]])
        
        # Use AdaBoost for prediction (simplest to implement here)
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        
        # Create and train a quick model for prediction
        quick_model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=50,
            learning_rate=1.0,
            random_state=42
        )
        
        X_train = data_store['X_train']
        y_train = data_store['y_train']
        quick_model.fit(X_train, y_train)
        
        # Make prediction
        prediction = quick_model.predict(input_encoded)[0]
        prediction_proba = quick_model.predict_proba(input_encoded)[0]
        
        return {
            "prediction": int(prediction),
            "prediction_label": "Stroke" if prediction == 1 else "No Stroke",
            "probability_no_stroke": float(prediction_proba[0]),
            "probability_stroke": float(prediction_proba[1]),
            "confidence": float(max(prediction_proba)),
            "input_data": input_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/churn")
async def predict_churn(data: Dict[str, Any]):
    """Predict churn for customer data based on dataset features"""
    try:
        # Check if a model has been trained
        if not data_store.get('model_trained', False):
            raise HTTPException(status_code=400, detail="Model hasn't been trained yet. Please train a model first before making predictions.")
        
        if data_store['X_train'] is None:
            raise HTTPException(status_code=404, detail="No dataset loaded")
        
        # Load the original dataset to get proper structure
        dataset_paths = [
            'boosting_small_dataset.csv',
            'data/boosting_small_dataset.csv',
            '../boosting_small_dataset.csv'
        ]
        
        df = None
        for path in dataset_paths:
            try:
                df = pd.read_csv(path)
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            raise HTTPException(status_code=404, detail="Original churn dataset not found")
        
        # Prepare the input data
        input_data = {
            'CreditScore': float(data.get('CreditScore', 650.0)),
            'Geography': data.get('Geography', 'France'),
            'Gender': data.get('Gender', 'Male'),
            'Age': float(data.get('Age', 35.0)),
            'Tenure': float(data.get('Tenure', 5.0)),
            'Balance': float(data.get('Balance', 1000.0)),
            'NumOfProducts': float(data.get('NumOfProducts', 2.0)),
            'HasCrCard': int(data.get('HasCrCard', 1)),
            'IsActiveMember': int(data.get('IsActiveMember', 1)),
            'EstimatedSalary': float(data.get('EstimatedSalary', 50000.0))
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables if they exist (Geography, Gender)
        input_encoded = input_df.copy()
        categorical_features = []
        if 'Geography' in df.columns:
            categorical_features.append('Geography')
        if 'Gender' in df.columns:
            categorical_features.append('Gender')
        
        for feature in categorical_features:
            if feature in input_encoded.columns:
                # Get unique values from original dataset for consistent encoding
                unique_values = sorted(df[feature].astype(str).unique())
                le = LabelEncoder()
                le.fit(unique_values)
                
                # Handle unseen categories
                if str(input_data[feature]) in le.classes_:
                    input_encoded[feature] = le.transform([str(input_data[feature])])
                else:
                    # Default to first category if not seen
                    input_encoded[feature] = le.transform([unique_values[0]])
        
        # Use the trained model if available, otherwise retrain
        if data_store.get('trained_model') is not None:
            model = data_store['trained_model']
        else:
            # Fallback: retrain a quick model
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.tree import DecisionTreeClassifier
            
            model = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=50,
                learning_rate=1.0,
                random_state=42
            )
            
            X_train = data_store['X_train']
            y_train = data_store['y_train']
            model.fit(X_train, y_train)
        
        # Make prediction
        prediction = model.predict(input_encoded)[0]
        prediction_proba = model.predict_proba(input_encoded)[0]
        
        return {
            "prediction": int(prediction),
            "prediction_label": "Churn" if prediction == 1 else "No Churn",
            "probability_no_churn": float(prediction_proba[0]),
            "probability_churn": float(prediction_proba[1]),
            "confidence": float(max(prediction_proba)),
            "input_data": input_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Algorithm implementations with progress tracking
class ProgressTracker:
    def __init__(self, algorithm_name: str, sid: str):
        self.algorithm_name = algorithm_name
        self.sid = sid
        self.current_iter = 0
        self.total_iters = 0
        self.loss_history = []
        self.metrics = {}
    
    async def emit_progress(self, iteration: int, loss: float = None, metrics: Dict = None, tree_info: Dict = None, decision_boundary: str = None):
        self.current_iter = iteration
        if loss is not None:
            self.loss_history.append(loss)
        
        if metrics:
            self.metrics.update(metrics)
        
        # Convert all data to JSON-serializable types
        progress_data = {
            'algorithm': self.algorithm_name,
            'iteration': iteration,
            'total_iterations': self.total_iters,
            'loss': loss,
            'metrics': metrics,
            'tree_info': tree_info,
            'decision_boundary': decision_boundary,
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert numpy types to native Python types
        progress_data = convert_numpy_types(progress_data)
        
        await sio.emit('training_progress', progress_data, room=self.sid)

async def train_adaboost(params: Dict, tracker: ProgressTracker):
    """Train AdaBoost with progress tracking"""
    X_train, X_test = data_store['X_train'], data_store['X_test']
    y_train, y_test = data_store['y_train'], data_store['y_test']
    
    n_estimators = params.get('n_estimators', 50)
    learning_rate = params.get('learning_rate', 1.0)
    max_depth = params.get('max_depth', 1)
    
    # Validate parameters
    if learning_rate <= 0:
        learning_rate = 1.0  # Default for AdaBoost
    learning_rate = max(0.1, min(2.0, learning_rate))  # Clamp between 0.1 and 2.0
    
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=max_depth),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )
    
    tracker.total_iters = n_estimators
    
    # Custom AdaBoost implementation to track real losses
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=max_depth),
        learning_rate=learning_rate,
        random_state=42
    )
    
    # We'll implement incremental training to track real losses
    X_train_array = X_train.values
    y_train_array = y_train.values
    X_test_array = X_test.values
    y_test_array = y_test.values
    
    # Initialize AdaBoost components
    model.n_estimators = n_estimators
    model.estimators_ = []
    model.estimator_weights_ = []
    model.estimator_errors_ = []
    loss_history = []  # Track loss history for visualization
    
    # Initialize sample weights
    sample_weight = np.ones(X_train.shape[0]) / X_train.shape[0]
    
    for i in range(n_estimators):
        # Fit weak learner
        estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=42 + i)
        estimator.fit(X_train_array, y_train_array, sample_weight=sample_weight)
        
        # Predict and calculate error
        y_pred_iter = estimator.predict(X_train_array)
        incorrect = (y_pred_iter != y_train_array)
        error = np.average(incorrect, weights=sample_weight)
        
        # Handle edge cases for numerical stability
        if error <= 0:
            error = 1e-10
        elif error >= 0.5:
            # If error is too high, cap alpha to prevent extreme weights
            alpha = learning_rate * 1.0  # Reasonable default
        else:
            # Calculate alpha (estimator weight) with bounds
            alpha = learning_rate * np.log((1 - error) / error) / 2
            # Cap alpha to prevent numerical issues
            alpha = np.clip(alpha, -10.0, 10.0)
        
        # Store estimator and weights
        model.estimators_.append(estimator)
        model.estimator_weights_.append(alpha)
        model.estimator_errors_.append(error)
        
        # Update sample weights (AdaBoost formula)
        # Convert predictions to -1/+1 format for AdaBoost
        y_pred_iter_binary = 2 * y_pred_iter - 1
        y_train_binary = 2 * y_train_array - 1
        sample_weight *= np.exp(-alpha * y_train_binary * y_pred_iter_binary)
        sample_weight /= sample_weight.sum()
        
        # Always calculate and store loss history
        # AdaBoost uses weighted sum in -1/+1 format
        ensemble_pred_train = np.zeros(len(X_train_array))
        ensemble_pred_test = np.zeros(len(X_test_array))
        
        for j, (est, weight) in enumerate(zip(model.estimators_, model.estimator_weights_)):
            # Convert predictions to -1/+1 format for proper AdaBoost ensemble
            pred_train_01 = est.predict(X_train_array)
            pred_test_01 = est.predict(X_test_array)
            pred_train_binary = 2 * pred_train_01 - 1  # Convert 0/1 to -1/+1
            pred_test_binary = 2 * pred_test_01 - 1
            
            ensemble_pred_train += weight * pred_train_binary
            ensemble_pred_test += weight * pred_test_binary
        
        # Convert back to 0/1 predictions for evaluation
        final_pred_train = (ensemble_pred_train > 0).astype(int)
        final_pred_test = (ensemble_pred_test > 0).astype(int)
        
        # Calculate metrics
        train_error = 1.0 - accuracy_score(y_train_array, final_pred_train)
        val_accuracy = accuracy_score(y_test_array, final_pred_test)
        
        # Store loss history for all iterations
        loss_history.append(train_error)
        
        # Emit progress every 5 iterations or on final iteration
        if (i + 1) % 5 == 0 or i == n_estimators - 1:
            # Get detailed tree information for visualization
            feature_names = data_store.get('feature_names', [])
            tree_info = None
            
            if feature_names and len(model.estimators_) > 0:
                latest_estimator = model.estimators_[-1]
                if hasattr(latest_estimator, 'tree_'):
                    tree_ = latest_estimator.tree_
                    
                    # Get tree structure details
                    def extract_tree_structure(tree, feature_names):
                        def build_node(node_id, depth=0, max_depth=4):
                            if node_id == -1 or depth >= max_depth:
                                return None
                                
                            if tree.children_left[node_id] == tree.children_right[node_id]:  # Leaf node
                                prediction = int(tree.value[node_id].argmax())
                                return {
                                    'node_id': node_id,
                                    'is_leaf': True,
                                    'prediction': prediction,
                                    'depth': depth,
                                    'samples': int(tree.n_node_samples[node_id])
                                }
                            else:  # Internal node
                                feature_idx = tree.feature[node_id]
                                feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f'feature_{feature_idx}'
                                threshold = tree.threshold[node_id]
                                
                                return {
                                    'node_id': node_id,
                                    'is_leaf': False,
                                    'feature': feature_name,
                                    'threshold': round(float(threshold), 2),
                                    'depth': depth,
                                    'samples': int(tree.n_node_samples[node_id]),
                                    'children': {
                                        'left': build_node(tree.children_left[node_id], depth + 1, max_depth),
                                        'right': build_node(tree.children_right[node_id], depth + 1, max_depth)
                                    }
                                }
                        
                        root = build_node(0, 0, 4)  # Limit to depth 4 for visualization
                        return root
                    
                    tree_structure = extract_tree_structure(tree_, feature_names)
                    
                    # Get root node info for backward compatibility
                    if tree_ and len(tree_.feature) > 0 and tree_.feature[0] != -2:
                        feature_idx = tree_.feature[0]
                        feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f'feature_{feature_idx}'
                        threshold = tree_.threshold[0]
                        tree_depth = latest_estimator.get_depth()
                        
                        tree_info = {
                            'feature': feature_name,
                            'threshold': round(threshold, 2),
                            'weight': round(float(alpha), 4),
                            'error': round(float(error), 4),
                            'depth': tree_depth,
                            'structure': tree_structure,
                            'type': 'adaboost_stump' if max_depth == 1 else 'decision_tree'
                        }
            
            # Generate decision boundary plot for key stages
            decision_boundary_plot = None
            if (i + 1) == 1 or (i + 1) == 3 or (i + 1) == 8 or i == n_estimators - 1:
                try:
                    decision_boundary_plot = generate_boosting_decision_boundary("adaboost", i + 1, "early" if i < 2 else "mid" if i < 5 else "late")
                except Exception as e:
                    print(f"Error generating decision boundary: {e}")
                    decision_boundary_plot = None
            
            await tracker.emit_progress(i + 1, loss=train_error, metrics={'accuracy': val_accuracy}, 
                                      tree_info=tree_info, decision_boundary=decision_boundary_plot)
    
    # Convert to numpy arrays
    model.estimator_weights_ = np.array(model.estimator_weights_)
    model.estimator_errors_ = np.array(model.estimator_errors_)
    
    # Calculate feature importances manually
    feature_importances = np.zeros(X_train.shape[1])
    for estimator, weight in zip(model.estimators_, model.estimator_weights_):
        feature_importances += weight * estimator.feature_importances_
    feature_importances = feature_importances / feature_importances.sum()
    
    # Store feature importances as a custom attribute
    model._custom_feature_importances_ = feature_importances
    
    # Add proper predict method for our custom AdaBoost model
    def custom_predict(X):
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X
            
        ensemble_pred = np.zeros(len(X_array))
        for est, weight in zip(model.estimators_, model.estimator_weights_):
            pred_01 = est.predict(X_array)
            pred_binary = 2 * pred_01 - 1  # Convert to -1/+1 format
            ensemble_pred += weight * pred_binary
        
        return (ensemble_pred > 0).astype(int)
    
    model.predict = custom_predict
    
    # Update tracker with the collected loss history
    tracker.loss_history = loss_history
    
    # Final evaluation using our custom predict method
    y_pred = custom_predict(X_test)
    final_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    await tracker.emit_progress(n_estimators, metrics=final_metrics)
    return model, final_metrics

async def train_gradient_boosting(params: Dict, tracker: ProgressTracker):
    """Train Gradient Boosting with progress tracking"""
    X_train, X_test = data_store['X_train'], data_store['X_test']
    y_train, y_test = data_store['y_train'], data_store['y_test']
    
    n_estimators = params.get('n_estimators', 100)
    learning_rate = params.get('learning_rate', 0.1)
    max_depth = params.get('max_depth', 3)
    
    tracker.total_iters = n_estimators
    loss_history = []
    
    # Train incrementally by creating new models with increasing n_estimators
    for i in range(10, n_estimators + 1, 10):
        # Create model with current number of estimators
        temp_model = GradientBoostingClassifier(
            n_estimators=i,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        
        # Train the temporary model
        temp_model.fit(X_train, y_train)
        
        # Calculate training loss (1 - accuracy on training set)
        train_pred = temp_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        train_loss = 1.0 - train_accuracy
        
        # Calculate validation accuracy
        val_pred = temp_model.predict(X_test)
        val_accuracy = accuracy_score(y_test, val_pred)
        
        # Store loss history
        loss_history.append(train_loss)
        
        # Get detailed tree information for Gradient Boosting
        tree_info = None
        if hasattr(temp_model, 'estimators_') and len(temp_model.estimators_) > 0:
            feature_names = data_store.get('feature_names', [])
            # Get the last tree in the latest estimator
            latest_estimator = temp_model.estimators_[-1, 0]  # GradientBoosting uses shape (n_trees, n_classes)
            if hasattr(latest_estimator, 'tree_'):
                tree_ = latest_estimator.tree_
                
                # Extract detailed tree structure for Gradient Boosting
                def extract_gb_tree_structure(tree, feature_names, max_depth=4):
                    def build_node(node_id, depth=0):
                        if node_id == -1 or depth >= max_depth:
                            return None
                            
                        if tree.children_left[node_id] == tree.children_right[node_id]:  # Leaf node
                            # For regression trees, use the value directly
                            value = tree.value[node_id][0, 0] if tree.value[node_id].shape[1] == 1 else tree.value[node_id][0, :]
                            prediction = 1 if (isinstance(value, (int, float)) and value > 0) else 0
                            
                            return {
                                'node_id': node_id,
                                'is_leaf': True,
                                'prediction': prediction,
                                'depth': depth,
                                'value': float(value),
                                'samples': int(tree.n_node_samples[node_id])
                            }
                        else:  # Internal node
                            feature_idx = tree.feature[node_id]
                            feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f'feature_{feature_idx}'
                            threshold = tree.threshold[node_id]
                            
                            return {
                                'node_id': node_id,
                                'is_leaf': False,
                                'feature': feature_name,
                                'threshold': round(float(threshold), 2),
                                'depth': depth,
                                'samples': int(tree.n_node_samples[node_id]),
                                'children': {
                                    'left': build_node(tree.children_left[node_id], depth + 1),
                                    'right': build_node(tree.children_right[node_id], depth + 1)
                                }
                            }
                    
                    root = build_node(0, 0)
                    return root
                
                tree_structure = extract_gb_tree_structure(tree_, feature_names)
                
                if len(tree_.feature) > 0 and tree_.feature[0] != -2:
                    feature_idx = tree_.feature[0]
                    feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f'feature_{feature_idx}'
                    threshold = tree_.threshold[0]
                    depth = latest_estimator.get_depth()
                    
                    tree_info = {
                        'feature': feature_name,
                        'threshold': round(threshold, 2),
                        'depth': depth,
                        'structure': tree_structure,
                        'type': 'gradient_boosting_tree',
                        'weight': learning_rate  # Use learning rate as weight for GB
                    }
        
        # Generate decision boundary plot for key stages
        decision_boundary_plot = None
        if i == 10 or i == 30 or i == 60 or i == n_estimators:
            try:
                decision_boundary_plot = generate_boosting_decision_boundary("gradient_boosting", i, "early" if i <= 30 else "mid" if i <= 60 else "late")
            except Exception as e:
                print(f"Error generating decision boundary: {e}")
                decision_boundary_plot = None
        
        await tracker.emit_progress(i, loss=train_loss, metrics={'accuracy': val_accuracy}, tree_info=tree_info, decision_boundary=decision_boundary_plot)
    
    # Also emit progress for the final model if n_estimators is not divisible by 10
    if n_estimators % 10 != 0:
        temp_model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        temp_model.fit(X_train, y_train)
        
        train_pred = temp_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        train_loss = 1.0 - train_accuracy
        
        val_pred = temp_model.predict(X_test)
        val_accuracy = accuracy_score(y_test, val_pred)
        
        loss_history.append(train_loss)
        
        # Generate decision boundary for final model
        decision_boundary_plot = None
        try:
            decision_boundary_plot = generate_boosting_decision_boundary("gradient_boosting", n_estimators, "late")
        except Exception as e:
            print(f"Error generating decision boundary: {e}")
            decision_boundary_plot = None
        
        await tracker.emit_progress(n_estimators, loss=train_loss, metrics={'accuracy': val_accuracy}, decision_boundary=decision_boundary_plot)
    
    # Create the final model
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Update tracker with the collected loss history
    tracker.loss_history = loss_history
    
    # Final evaluation
    y_pred = model.predict(X_test)
    final_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # Generate final decision boundary
    final_decision_boundary = None
    try:
        final_decision_boundary = generate_boosting_decision_boundary("gradient_boosting", n_estimators, "late")
    except Exception as e:
        print(f"Error generating final decision boundary: {e}")
    
    await tracker.emit_progress(n_estimators, metrics=final_metrics, decision_boundary=final_decision_boundary)
    return model, final_metrics

async def train_xgboost(params: Dict, tracker: ProgressTracker):
    """Train XGBoost with progress tracking"""
    if not XGBOOST_AVAILABLE:
        raise HTTPException(status_code=400, detail="XGBoost not available. Install with: pip install xgboost")
    
    X_train, X_test = data_store['X_train'], data_store['X_test']
    y_train, y_test = data_store['y_train'], data_store['y_test']
    
    n_estimators = params.get('n_estimators', 100)
    learning_rate = params.get('learning_rate', 0.1)
    max_depth = params.get('max_depth', 3)
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
        eval_metric='logloss'
    )
    
    tracker.total_iters = n_estimators
    
    # Train with evaluation set for progress tracking
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    # Get training history
    results = model.evals_result()
    train_loss = results['validation_0']['logloss']
    val_loss = results['validation_1']['logloss']
    
    # Emit progress for each evaluation
    for i in range(len(train_loss)):
        if i % 5 == 0 or i == len(train_loss) - 1:
            val_pred = model.predict(X_test)
            val_accuracy = accuracy_score(y_test, val_pred)
            
            # Generate decision boundary plot for key stages
            decision_boundary_plot = None
            iteration_num = i + 1
            if iteration_num == 1 or iteration_num == 20 or iteration_num == 50 or iteration_num == len(train_loss):
                try:
                    decision_boundary_plot = generate_boosting_decision_boundary("xgboost", iteration_num, "early" if iteration_num <= 20 else "mid" if iteration_num <= 50 else "late")
                except Exception as e:
                    print(f"Error generating decision boundary: {e}")
                    decision_boundary_plot = None
            
            await tracker.emit_progress(iteration_num, loss=val_loss[i], metrics={'accuracy': val_accuracy}, decision_boundary=decision_boundary_plot)
    
    # Final evaluation
    y_pred = model.predict(X_test)
    final_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # Generate final decision boundary
    final_decision_boundary = None
    try:
        final_decision_boundary = generate_boosting_decision_boundary("xgboost", n_estimators, "late")
    except Exception as e:
        print(f"Error generating final decision boundary: {e}")
    
    await tracker.emit_progress(n_estimators, metrics=final_metrics, decision_boundary=final_decision_boundary)
    return model, final_metrics

async def train_lightgbm(params: Dict, tracker: ProgressTracker):
    """Train LightGBM with progress tracking"""
    if not LIGHTGBM_AVAILABLE:
        raise HTTPException(status_code=400, detail="LightGBM not available. Install with: pip install lightgbm")
    
    X_train, X_test = data_store['X_train'], data_store['X_test']
    y_train, y_test = data_store['y_train'], data_store['y_test']
    
    n_estimators = params.get('n_estimators', 100)
    learning_rate = params.get('learning_rate', 0.1)
    max_depth = params.get('max_depth', 3)
    
    tracker.total_iters = n_estimators
    
    # Train incrementally for progress tracking
    for i in range(10, n_estimators + 1, 10):
        temp_model = lgb.LGBMClassifier(
            n_estimators=i,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            verbose=-1
        )
        temp_model.fit(X_train, y_train)
        
        val_pred = temp_model.predict(X_test)
        val_accuracy = accuracy_score(y_test, val_pred)
        train_loss = 1.0 - accuracy_score(y_train, temp_model.predict(X_train))
        
        # Generate decision boundary plot for key stages
        decision_boundary_plot = None
        if i == 10 or i == 30 or i == 60 or i == n_estimators:
            try:
                decision_boundary_plot = generate_boosting_decision_boundary("lightgbm", i, "early" if i <= 30 else "mid" if i <= 60 else "late")
            except Exception as e:
                print(f"Error generating decision boundary: {e}")
                decision_boundary_plot = None
        
        await tracker.emit_progress(i, loss=train_loss, metrics={'accuracy': val_accuracy}, decision_boundary=decision_boundary_plot)
    
    # Create final model
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    # Get feature importance and final metrics
    y_pred = model.predict(X_test)
    final_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # Generate final decision boundary
    final_decision_boundary = None
    try:
        final_decision_boundary = generate_boosting_decision_boundary("lightgbm", n_estimators, "late")
    except Exception as e:
        print(f"Error generating final decision boundary: {e}")
    
    # Emit final results
    await tracker.emit_progress(n_estimators, metrics=final_metrics, decision_boundary=final_decision_boundary)
    return model, final_metrics

async def train_catboost(params: Dict, tracker: ProgressTracker):
    """Train CatBoost with progress tracking"""
    if not CATBOOST_AVAILABLE:
        raise HTTPException(status_code=400, detail="CatBoost not available. Install with: pip install catboost")
    
    X_train, X_test = data_store['X_train'], data_store['X_test']
    y_train, y_test = data_store['y_train'], data_store['y_test']
    
    n_estimators = params.get('n_estimators', 100)
    learning_rate = params.get('learning_rate', 0.1)
    max_depth = params.get('max_depth', 3)
    
    tracker.total_iters = n_estimators
    
    # Train incrementally for progress tracking
    for i in range(10, n_estimators + 1, 10):
        temp_model = cb.CatBoostClassifier(
            iterations=i,
            learning_rate=learning_rate,
            depth=max_depth,
            random_seed=42,
            verbose=False
        )
        temp_model.fit(X_train, y_train)
        
        val_pred = temp_model.predict(X_test)
        val_accuracy = accuracy_score(y_test, val_pred)
        train_loss = 1.0 - accuracy_score(y_train, temp_model.predict(X_train))
        
        # Generate decision boundary plot for key stages
        decision_boundary_plot = None
        if i == 10 or i == 30 or i == 60 or i == n_estimators:
            try:
                decision_boundary_plot = generate_boosting_decision_boundary("catboost", i, "early" if i <= 30 else "mid" if i <= 60 else "late")
            except Exception as e:
                print(f"Error generating decision boundary: {e}")
                decision_boundary_plot = None
        
        await tracker.emit_progress(i, loss=train_loss, metrics={'accuracy': val_accuracy}, decision_boundary=decision_boundary_plot)
    
    # Create final model
    model = cb.CatBoostClassifier(
        iterations=n_estimators,
        learning_rate=learning_rate,
        depth=max_depth,
        random_seed=42,
        verbose=False
    )
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True
    )
    
    # Get feature importance and final metrics
    y_pred = model.predict(X_test)
    final_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # Generate final decision boundary
    final_decision_boundary = None
    try:
        final_decision_boundary = generate_boosting_decision_boundary("catboost", n_estimators, "late")
    except Exception as e:
        print(f"Error generating final decision boundary: {e}")
    
    # Emit final results
    await tracker.emit_progress(n_estimators, metrics=final_metrics, decision_boundary=final_decision_boundary)
    return model, final_metrics

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    print(f"Client {sid} connected")
    await sio.emit('connected', {'message': 'Connected to Boosting Demo Server'}, room=sid)

@sio.event
async def disconnect(sid):
    print(f"Client {sid} disconnected")

@sio.event
async def start_training(sid, data):
    """Handle start training request"""
    try:
        algorithm = data.get('algorithm')
        params = data.get('params', {})
        dataset_name = data.get('dataset', 'stroke')  # Default to stroke if not specified
        n_samples = params.get('n_samples', None)  # Get number of samples to use
        
        # Reset training flag when starting new training
        data_store['model_trained'] = False
        data_store['trained_model'] = None
        data_store['trained_algorithm'] = None
        
        # Load the requested dataset for training with specified number of samples
        if not load_dataset_for_training(dataset_name, n_samples):
            await sio.emit('error', {'message': f'Failed to load dataset: {dataset_name}'}, room=sid)
            return
        
        if data_store['X_train'] is None:
            await sio.emit('error', {'message': 'No dataset loaded'}, room=sid)
            return
        
        tracker = ProgressTracker(algorithm, sid)
        
        # Start training based on algorithm
        training_funcs = {
            'adaboost': train_adaboost,
            'gradient_boosting': train_gradient_boosting,
            'xgboost': train_xgboost,
            'lightgbm': train_lightgbm,
            'catboost': train_catboost
        }
        
        if algorithm not in training_funcs:
            await sio.emit('error', {'message': f'Unknown algorithm: {algorithm}'}, room=sid)
            return
        
        # Send training data to frontend for visualization
        training_data = {
            'X_train': data_store['X_train'].values.tolist(),
            'y_train': data_store['y_train'].values.tolist(),
            'feature_names': data_store.get('feature_names', [])
        }
        await sio.emit('training_started', {
            'algorithm': algorithm, 
            'params': params,
            'training_data': convert_numpy_types(training_data)
        }, room=sid)
        
        try:
            model, metrics = await training_funcs[algorithm](params, tracker)
            
            # Get feature importances
            feature_importances = None
            if hasattr(model, '_custom_feature_importances_'):
                feature_importances = dict(zip(data_store['feature_names'], model._custom_feature_importances_))
            
            # Convert all data to JSON-serializable types
            completion_data = {
                'algorithm': algorithm,
                'metrics': metrics,  # These are the real metrics from test set
                'feature_importances': feature_importances,
                'loss_history': tracker.loss_history,
                'total_iterations': params.get('n_estimators', 50)
            }
            completion_data = convert_numpy_types(completion_data)
            
            # Mark model as trained
            data_store['model_trained'] = True
            data_store['trained_model'] = model
            data_store['trained_algorithm'] = algorithm
            
            await sio.emit('training_completed', completion_data, room=sid)
            
        except Exception as e:
            await sio.emit('training_error', {'message': str(e), 'algorithm': algorithm}, room=sid)
            
    except Exception as e:
        await sio.emit('error', {'message': str(e)}, room=sid)

# Add static file serving for React app (after API routes)
if serve_static:
    @app.get("/")
    async def serve_react_app():
        index_path = os.path.join(static_path, 'index.html')
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Boosting Algorithms Demo API", "status": "running"}
            
    @app.get("/{full_path:path}")
    async def serve_react_routes(full_path: str):
        # Only serve React app for non-API routes
        if not any(full_path.startswith(prefix) for prefix in ['algorithms', 'dataset', 'static']):
            index_path = os.path.join(static_path, 'index.html')
            if os.path.exists(index_path):
                return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Not found")
else:
    @app.get("/")
    async def root():
        return {"message": "Boosting Algorithms Demo API", "status": "running"}

if __name__ == "__main__":
    uvicorn.run(socket_app, host="0.0.0.0", port=8000, reload=False)
