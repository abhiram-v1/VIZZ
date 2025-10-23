# Boosting Algorithms Interactive Demo

A real-time, interactive demo that runs all major boosting algorithms (AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost) with live training progress, metrics, and visualizations.

## Features

- **Real-time Training**: Watch algorithms train with live progress updates via WebSocket
- **Individual Algorithm Pages**: Dedicated pages for each boosting algorithm
- **Live Visualizations**: Training loss curves and feature importance charts
- **Dataset Preview**: View first 10 rows of the dataset
- **Comparison View**: Compare all algorithms side-by-side
- **Parameter Controls**: Adjust hyperparameters (n_estimators, learning_rate, max_depth)
- **Upload Support**: Upload your own CSV datasets (with size limits for demo)

## Tech Stack

- **Frontend**: React with React Router, Socket.IO client, Recharts for visualizations
- **Backend**: FastAPI with Socket.IO server, scikit-learn, XGBoost, LightGBM, CatBoost
- **Real-time**: WebSocket communication for live training progress
- **Styling**: Custom CSS with responsive design

## Quick Start

### Option 1: Docker (Recommended)

1. **Clone and run with Docker Compose:**
   ```bash
   git clone <repository>
   cd boosting-algorithms-demo
   docker-compose up
   ```

2. **Access the application:**
   - Open http://localhost:8000 in your browser

### Option 2: Manual Setup

#### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the backend server:**
   ```bash
   python main.py
   ```

#### Frontend Setup

1. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

2. **Start the frontend development server:**
   ```bash
   npm start
   ```

3. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

## Usage

### Individual Algorithm Pages

Navigate to any algorithm page:
- `/adaboost` - AdaBoost Classifier
- `/gradient-boosting` - Gradient Boosting Classifier  
- `/xgboost` - XGBoost Classifier
- `/lightgbm` - LightGBM Classifier
- `/catboost` - CatBoost Classifier

Each page shows:
- Dataset preview (first 10 rows)
- Training controls with parameter adjustment
- Real-time training progress and logs
- Final metrics (accuracy, precision, recall, F1-score)
- Training loss curve (updated in real-time)
- Feature importance visualization

### Compare All Algorithms

Visit `/all` to:
- Train all available algorithms simultaneously
- Compare performance metrics side-by-side
- View algorithm status and progress
- See performance comparison charts

### Controls

- **Start Training**: Begin training with current parameters
- **Stop**: Stop training (partial results shown)
- **Clear Logs**: Clear the training logs
- **Parameters**: Adjust n_estimators, learning_rate, and max_depth

## Dataset

The demo uses a stroke prediction dataset (`data/stroke_data_balanced.csv`) with the following features:
- `age`, `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`
- `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`
- Target: `stroke` (binary classification)

You can upload your own CSV files, but they must have a `stroke` column as the target variable.

## API Endpoints

### REST API
- `GET /` - API status
- `GET /algorithms` - List available algorithms
- `GET /dataset/preview?rows=10` - Get dataset preview
- `POST /dataset/upload` - Upload new dataset

### WebSocket Events
- `connect` - Connection established
- `start_training` - Start algorithm training
- `training_progress` - Real-time training updates
- `training_completed` - Training finished with results
- `training_error` - Training failed with error

## Requirements

### Python Dependencies
- FastAPI 0.104.1
- python-socketio 5.10.0
- pandas 2.1.3
- numpy 1.24.3
- scikit-learn 1.3.2
- xgboost 2.0.2
- lightgbm 4.1.0
- catboost 1.2.2

### Node.js Dependencies
- React 18.2.0
- React Router 6.8.1
- socket.io-client 4.7.4
- recharts 2.8.0

## Architecture

```
Frontend (React)          Backend (FastAPI)
├── Pages/                ├── main.py
│   ├── AdaBoostPage      ├── Socket.IO server
│   ├── XGBoostPage       ├── Algorithm implementations
│   └── AllAlgorithmsPage └── Real-time progress tracking
├── Components/
│   └── AlgorithmPage     WebSocket Communication
└── Services/
    ├── socketService.js  ├── training_progress events
    └── apiService.js     └── REST API calls
```

## Development

### Adding New Algorithms

1. **Backend**: Add algorithm implementation in `backend/main.py`
2. **Frontend**: Create new page component in `src/pages/`
3. **Routes**: Add route in `src/App.js`

### Customizing Parameters

Edit the `defaultParams` object in each algorithm page component to change default hyperparameters.

## Troubleshooting

### Backend Issues
- Ensure all Python dependencies are installed
- Check that the dataset file exists in the correct path
- Verify port 8000 is not in use

### Frontend Issues  
- Ensure Node.js and npm are installed
- Check that the backend is running on port 8000
- Verify WebSocket connection in browser dev tools

### Missing ML Libraries
If certain algorithms show as unavailable:
```bash
pip install xgboost lightgbm catboost
```

## License

This project is for educational and demonstration purposes.
