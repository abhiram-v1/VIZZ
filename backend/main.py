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
from typing import Dict, Any, Optional
import socketio
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime

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

# Global variables for data and models
data_store = {
    'X_train': None,
    'X_test': None,
    'y_train': None,
    'y_test': None,
    'feature_names': None,
    'categorical_features': None,
    'numerical_features': None,
    'dataset': None
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
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
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
async def get_dataset_preview(rows: int = 10):
    """Get first N rows of the dataset"""
    if data_store['dataset'] is None:
        raise HTTPException(status_code=404, detail="No dataset loaded")
    
    preview = data_store['dataset'].head(rows)
    return {
        "data": preview.to_dict('records'),
        "columns": preview.columns.tolist(),
        "shape": data_store['dataset'].shape
    }

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
        
        # Limit dataset size for demo (max 10k rows)
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42)
        
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
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
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

# Algorithm implementations with progress tracking
class ProgressTracker:
    def __init__(self, algorithm_name: str, sid: str):
        self.algorithm_name = algorithm_name
        self.sid = sid
        self.current_iter = 0
        self.total_iters = 0
        self.loss_history = []
        self.metrics = {}
    
    async def emit_progress(self, iteration: int, loss: float = None, metrics: Dict = None, tree_info: Dict = None):
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
            
            await tracker.emit_progress(i + 1, loss=train_error, metrics={'accuracy': val_accuracy}, tree_info=tree_info)
    
    # Convert to numpy arrays
    model.estimator_weights_ = np.array(model.estimator_weights_)
    model.estimator_errors_ = np.array(model.estimator_errors_)
    
    # Add feature_importances_ property
    feature_importances = np.zeros(X_train.shape[1])
    for estimator, weight in zip(model.estimators_, model.estimator_weights_):
        feature_importances += weight * estimator.feature_importances_
    model.feature_importances_ = feature_importances / feature_importances.sum()
    
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
        
        await tracker.emit_progress(i, loss=train_loss, metrics={'accuracy': val_accuracy}, tree_info=tree_info)
    
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
        await tracker.emit_progress(n_estimators, loss=train_loss, metrics={'accuracy': val_accuracy})
    
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
    
    await tracker.emit_progress(n_estimators, metrics=final_metrics)
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
            await tracker.emit_progress(i + 1, loss=val_loss[i], metrics={'accuracy': val_accuracy})
    
    # Final evaluation
    y_pred = model.predict(X_test)
    final_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
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
    
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
        verbose=-1
    )
    
    tracker.total_iters = n_estimators
    
    # Train with evaluation set
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    
    # Get feature importance and final metrics
    y_pred = model.predict(X_test)
    final_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # Emit final results
    await tracker.emit_progress(n_estimators, metrics=final_metrics)
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
    
    model = cb.CatBoostClassifier(
        iterations=n_estimators,
        learning_rate=learning_rate,
        depth=max_depth,
        random_seed=42,
        verbose=False
    )
    
    tracker.total_iters = n_estimators
    
    # Train with evaluation set
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
    
    # Emit final results
    await tracker.emit_progress(n_estimators, metrics=final_metrics)
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
        
        await sio.emit('training_started', {'algorithm': algorithm, 'params': params}, room=sid)
        
        try:
            model, metrics = await training_funcs[algorithm](params, tracker)
            
            # Get feature importances
            feature_importances = None
            if hasattr(model, 'feature_importances_'):
                feature_importances = dict(zip(data_store['feature_names'], model.feature_importances_))
            
            # Convert all data to JSON-serializable types
            completion_data = {
                'algorithm': algorithm,
                'metrics': metrics,
                'feature_importances': feature_importances,
                'loss_history': tracker.loss_history
            }
            completion_data = convert_numpy_types(completion_data)
            
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
