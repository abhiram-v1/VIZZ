# Boosting Algorithms Interactive Demo & ML Learning Journey

A comprehensive, interactive web application that combines real-time machine learning algorithm demonstrations with an engaging, story-driven educational experience. This project teaches machine learning concepts through hands-on experimentation and an immersive learning journey.

![ML Learning Journey Homepage](i1.png)
*Figure 1: ML Learning Journey homepage - The main entry point featuring Alex's learning journey through machine learning concepts*

## 🎯 Project Overview

This application serves two main purposes:

1. **Interactive Algorithm Demo**: Real-time training and visualization of boosting algorithms (AdaBoost, Gradient Boosting, and XGBoost) with live progress tracking
2. **ML Learning Journey**: A story-driven educational experience that teaches machine learning concepts step-by-step through the journey of Alex, a curious student learning ML

## ✨ Key Features

### 1. ML Learning Journey (Story-Driven Education)
- **Interactive Story**: Follow Alex as he learns machine learning from basics to advanced techniques
- **5 Chapters**: Progressive learning from Decision Trees to Boosting Algorithms
- **Visual Decision Tree**: Interactive D3.js visualization showing decision-making process
- **Search Functionality**: Smart search (Ctrl+K) that navigates directly to algorithm explanations
- **AdaBoost Step-by-Step**: Auto-play slideshow (4.5s intervals) with visual walkthrough of how AdaBoost works

![Decision Tree Visualization](i2.png)
*Figure 2: Decision Tree visualization - Interactive example showing "Is it raining?" decision tree with clear visual structure*

### 2. Real-Time Algorithm Training
- **Live Training**: Watch algorithms train in real-time with WebSocket updates
- **Progress Tracking**: Real-time loss curves and metrics updates
- **Multiple Algorithms**: Support for AdaBoost, Gradient Boosting, and XGBoost
- **Parameter Controls**: Adjust hyperparameters (n_estimators, learning_rate, max_depth) on the fly
- **Dataset Selection**: Choose between Stroke Prediction or Customer Churn datasets

![Algorithm Training Interface](i3.png)
*Figure 3: Algorithm training page - Real-time training with parameter controls, live metrics, and prediction form*

### 3. Decision Boundary Visualizations
- **Real Decision Boundaries**: Visualize actual decision boundaries from trained models using matplotlib
- **Iterative Progression**: See how decision boundaries evolve through training iterations
- **Auto-Play Slideshow**: Play button automatically advances through decision boundaries (2-second intervals)
- **Manual Control**: Pause/play and navigate manually with previous/next buttons

![Decision Boundary Visualization](i5.png)
*Figure 4: Decision boundary carousel - Visual representation of how models separate different classes in feature space*

### 4. AdaBoost Interactive Explanation
- **Step-by-Step Walkthrough**: Visual explanation of AdaBoost algorithm with student examples
- **Auto-Play Mode**: Automatically advances through 6 steps (4.5 seconds per step)
- **Weight Visualization**: See how misclassified examples get higher weights
- **Smart Scrolling**: Automatically adjusts scroll position to show full content of each slide

![AdaBoost Step-by-Step Explanation](i4.png)
*Figure 5: AdaBoost explanation slideshow - Interactive walkthrough showing how AdaBoost learns from mistakes*

## 🚀 Quick Start

### Automated Startup (Recommended)

**Windows:**
```bash
python start_servers.py
# or
start_servers.bat
start_servers.ps1
```

**Linux/Mac:**
```bash
python start_servers.py
```

The `start_servers.py` script automatically:
- Starts FastAPI backend on port 8000
- Starts React frontend on port 3000
- Opens browser to http://localhost:3000
- Monitors both servers and handles cleanup

### Manual Setup

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python main.py
```

**Frontend:**
```bash
npm install
npm start
```

## 📚 Learning Journey Structure

### Chapter 1: Machine Learning Basics
- Supervised, Unsupervised, and Reinforcement Learning
- Real-world analogies and examples

### Chapter 2: Decision Trees
- Flowchart-like structures for making decisions
- Interactive "Is it raining?" example with visual tree
- Concepts: Entropy, Information Gain, Splitting

### Chapter 3: Ensemble Learning
- Power of combining multiple models
- Voting mechanisms and weak learners
- Why teams work better than individuals

### Chapter 4: Boosting Algorithms
- **AdaBoost**: Adaptive boosting with weighted examples (interactive slideshow)
- **Gradient Boosting**: Step-by-step error correction
- **XGBoost**: Fast, optimized gradient boosting
- Visual explanations with student learning analogies

### Chapter 5: Experiments & Comparison
- Performance comparison of all algorithms
- Metrics: Accuracy, Precision, Recall, F1-Score
- Real results on stroke prediction dataset

## 🔬 Algorithm Training Features

### Available Algorithms
- **AdaBoost**: Adaptive boosting classifier
- **Gradient Boosting**: Sequential error correction
- **XGBoost**: Extreme gradient boosting

### Training Interface
Each algorithm page provides:
- **Dataset Preview**: First 10 rows of selected dataset
- **Parameter Controls**: Adjust n_estimators, learning_rate, max_depth
- **Real-time Progress**: Live updates via WebSocket
- **Metrics Display**: Accuracy, Precision, Recall, F1-Score
- **Loss Curves**: Real-time training loss visualization
- **Feature Importance**: Charts showing which features matter most
- **Prediction Form**: Interactive form to predict outcomes for new data
- **Validation**: Clear error message if predicting before training

### Datasets
- **Stroke Prediction**: Medical data with features like age, glucose level, BMI
- **Customer Churn**: Business data with features like credit score, geography, gender

## 🎨 Design Features

- **Dark Theme**: Modern purple/blue gradient background
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Large Text**: Optimized font sizes (20px base) for readability
- **Smooth Animations**: Polished transitions and hover effects
- **Auto-Play Features**: Hands-free navigation through visualizations
- **Smart Scrolling**: Automatically adjusts to show full content

## 🛠️ Tech Stack

### Frontend
- React 18.2.0 with React Router
- D3.js for data visualizations
- Socket.IO Client for real-time communication
- Custom CSS (9500+ lines)

### Backend
- FastAPI with WebSocket support
- scikit-learn and XGBoost
- Matplotlib for decision boundary generation
- Pandas & NumPy for data manipulation

## 📊 API Endpoints

### REST API
- `GET /algorithms` - List available algorithms
- `GET /dataset/preview` - Get dataset preview
- `POST /predict` - Predict outcomes (stroke or churn)
- `POST /predict/churn` - Predict churn specifically
- `GET /plot/boosting-boundary` - Get decision boundary visualization

### WebSocket Events
- `start_training` - Start algorithm training
- `training_progress` - Real-time training updates
- `training_completed` - Training finished with results

## 🎓 How to Use

### Navigation
- **Homepage**: Start at the ML Learning Journey (default route)
- **Search**: Press `Ctrl+K` (or `Cmd+K` on Mac) to search for concepts
- **Direct Links**: Navigate to specific algorithms via navbar

### Interactive Elements
- **Decision Tree**: Click through the interactive visualization
- **AdaBoost Slideshow**: Use play button for auto-advance or navigate manually
- **Decision Boundaries**: Auto-play through iterations or use manual controls
- **Algorithm Training**: Adjust parameters and watch real-time training

### Tips
- Use auto-play features for hands-free learning
- Search functionality helps jump directly to concepts
- Each algorithm page allows independent training and prediction
- Decision boundaries update as training progresses

## 🐛 Troubleshooting

**Backend Issues:**
- Port 8000 in use: Change port in `backend/main.py` or stop conflicting service
- Missing dependencies: `pip install -r backend/requirements.txt`
- Dataset not found: Ensure `data/` directory contains required CSV files

**Frontend Issues:**
- Port 3000 in use: React will prompt to use another port
- WebSocket connection failed: Ensure backend is running on port 8000
- Build errors: Run `npm install` to ensure all dependencies are installed

## 📁 Project Structure

```
Project/
├── backend/                    # FastAPI backend
│   ├── main.py                # Main API server
│   └── requirements.txt       # Python dependencies
├── src/                       # React frontend
│   ├── pages/                 # Page components
│   ├── components/            # Reusable components
│   └── services/              # API services
├── data/                      # Datasets
│   ├── stroke_data_balanced.csv
│   └── boosting_small_dataset.csv
├── start_servers.py           # Automated startup
└── README.md                  # This file
```

## 🎯 Learning Outcomes

After using this application, users will understand:
1. **Machine Learning Basics** - Types and applications
2. **Decision Trees** - How they work through interactive examples
3. **Ensemble Learning** - Why combining models is powerful
4. **Boosting Algorithms** - How they learn from mistakes
5. **Algorithm Differences** - AdaBoost vs Gradient Boosting vs XGBoost
6. **Real-world Application** - Practical use cases with medical and business data

## 📄 License

This project is for educational and demonstration purposes.

---

**Note**: This is a comprehensive educational tool designed to make machine learning concepts accessible and engaging through interactive visualizations and story-driven learning.
