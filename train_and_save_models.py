import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TRAINING AND SAVING WEAK CLASSIFIERS FOR BOOSTING")
print("=" * 60)

# Load the balanced dataset
df = pd.read_csv('stroke_data_balanced.csv')
print(f"Dataset shape: {df.shape}")
print(f"Class distribution:")
print(df['stroke'].value_counts())

# 1. Data Preparation
print("\n1. DATA PREPARATION")
print("-" * 30)

# Separate features and target
X = df.drop(['id', 'stroke'], axis=1)
y = df['stroke']

# Identify categorical and numerical features
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numerical_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

print(f"Categorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")

# Encode categorical variables
X_encoded = X.copy()
le_dict = {}

for feature in categorical_features:
    le = LabelEncoder()
    X_encoded[feature] = le.fit_transform(X_encoded[feature].astype(str))
    le_dict[feature] = le

print("Categorical encoding completed")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale numerical features for some classifiers
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

print("Feature scaling completed")

# 2. Define and Train Weak Classifiers
print("\n2. TRAINING WEAK CLASSIFIERS")
print("-" * 30)

classifiers = {
    'Decision Tree': DecisionTreeClassifier(
        max_depth=3,  # Shallow tree for weak classifier
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform'
    )
}

# Train and evaluate each classifier
results = {}
trained_models = {}

for name, classifier in classifiers.items():
    print(f"\nTraining {name}...")
    
    # Use scaled data for Logistic Regression and KNN
    if name in ['Logistic Regression', 'K-Nearest Neighbors']:
        X_train_use = X_train_scaled
        X_test_use = X_test_scaled
    else:
        X_train_use = X_train
        X_test_use = X_test
    
    # Train the classifier
    classifier.fit(X_train_use, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test_use)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store results and trained model
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred
    }
    
    # Store the trained model with its preprocessing info
    trained_models[name] = {
        'model': classifier,
        'use_scaled_data': name in ['Logistic Regression', 'K-Nearest Neighbors'],
        'feature_names': X_encoded.columns.tolist()
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

# 3. Save All Models and Preprocessing Objects
print("\n3. SAVING MODELS AND PREPROCESSING OBJECTS")
print("-" * 30)

# Save trained models
with open('trained_models.pkl', 'wb') as f:
    pickle.dump(trained_models, f)
print("Trained models saved to 'trained_models.pkl'")

# Save label encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(le_dict, f)
print("Label encoders saved to 'label_encoders.pkl'")

# Save scaler
with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Feature scaler saved to 'feature_scaler.pkl'")

# Save training/test data for future use
data_dict = {
    'X_train': X_train,
    'X_test': X_test,
    'X_train_scaled': X_train_scaled,
    'X_test_scaled': X_test_scaled,
    'y_train': y_train,
    'y_test': y_test,
    'feature_names': X_encoded.columns.tolist(),
    'categorical_features': categorical_features,
    'numerical_features': numerical_features
}

with open('training_data.pkl', 'wb') as f:
    pickle.dump(data_dict, f)
print("Training data saved to 'training_data.pkl'")

# 4. Create Model Loading Utility
print("\n4. CREATING MODEL LOADING UTILITY")
print("-" * 30)

model_loader_code = '''
import pickle
import pandas as pd
import numpy as np

def load_models_and_preprocessing():
    """Load all trained models and preprocessing objects"""
    
    # Load models
    with open('trained_models.pkl', 'rb') as f:
        models = pickle.load(f)
    
    # Load encoders
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    # Load scaler
    with open('feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load training data
    with open('training_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    return models, encoders, scaler, data

def predict_with_model(model_name, X_new, models, encoders, scaler):
    """Make predictions with a specific trained model"""
    
    # Get model info
    model_info = models[model_name]
    model = model_info['model']
    use_scaled = model_info['use_scaled_data']
    
    # Encode categorical variables
    X_encoded = X_new.copy()
    for feature, encoder in encoders.items():
        if feature in X_encoded.columns:
            X_encoded[feature] = encoder.transform(X_encoded[feature].astype(str))
    
    # Scale if needed
    if use_scaled:
        numerical_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
        X_encoded[numerical_features] = scaler.transform(X_encoded[numerical_features])
    
    # Make prediction
    prediction = model.predict(X_encoded)
    return prediction

def get_all_predictions(X_new, models, encoders, scaler):
    """Get predictions from all models"""
    predictions = {}
    for model_name in models.keys():
        predictions[model_name] = predict_with_model(model_name, X_new, models, encoders, scaler)
    return predictions

# Example usage:
# models, encoders, scaler, data = load_models_and_preprocessing()
# predictions = get_all_predictions(X_test, models, encoders, scaler)
'''

with open('model_loader.py', 'w') as f:
    f.write(model_loader_code)
print("Model loading utility saved to 'model_loader.py'")

# 5. Test Model Loading
print("\n5. TESTING MODEL LOADING")
print("-" * 30)

# Test loading
with open('trained_models.pkl', 'rb') as f:
    loaded_models = pickle.load(f)

print("Successfully loaded models:")
for name in loaded_models.keys():
    print(f"  - {name}")

# Test prediction with loaded model
test_sample = X_test.iloc[:5]  # First 5 test samples
dt_model = loaded_models['Decision Tree']['model']
dt_pred = dt_model.predict(test_sample)
print(f"\nTest prediction with Decision Tree: {dt_pred}")

# 6. Results Summary
print("\n6. RESULTS SUMMARY")
print("-" * 30)

results_df = pd.DataFrame({
    'Classifier': list(results.keys()),
    'Accuracy': [results[name]['accuracy'] for name in results.keys()],
    'Precision': [results[name]['precision'] for name in results.keys()],
    'Recall': [results[name]['recall'] for name in results.keys()],
    'F1-Score': [results[name]['f1_score'] for name in results.keys()]
})

print(results_df.round(4))

# Save results
results_df.to_csv('weak_classifiers_results.csv', index=False)
print("\nResults saved to 'weak_classifiers_results.csv'")

print(f"\n" + "=" * 60)
print("MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("=" * 60)
print("Files created:")
print("  - trained_models.pkl (trained models)")
print("  - label_encoders.pkl (categorical encoders)")
print("  - feature_scaler.pkl (numerical scaler)")
print("  - training_data.pkl (train/test data)")
print("  - model_loader.py (loading utility)")
print("  - weak_classifiers_results.csv (performance metrics)")
print("\nReady for boosting ensemble methods!")
