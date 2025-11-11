# Boosting Algorithms Interactive Demo & ML Learning Journey

A comprehensive, interactive web application that combines real-time machine learning algorithm demonstrations with an engaging, story-driven educational experience. This project teaches machine learning concepts through hands-on experimentation and an immersive learning journey.

## 🎯 Project Overview

This application serves two main purposes:

1. **Interactive Algorithm Demo**: Real-time training and visualization of boosting algorithms (AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost) with live progress tracking
2. **ML Learning Journey**: A story-driven educational experience that teaches machine learning concepts step-by-step through the journey of Alex, a curious student learning ML

## ✨ Key Features

### 1. ML Learning Journey (Story-Driven Education)
- **Interactive Story**: Follow Alex, a male student, as he learns machine learning from basics to advanced techniques
- **5 Chapters**: Progressive learning from Decision Trees to Boosting Algorithms
- **Visual Decision Tree**: Interactive D3.js visualization showing a simple decision tree (rain/umbrella example)
- **Search Functionality**: Smart search that navigates directly to algorithm explanations and concepts
- **Concept Explanations**: Detailed explanations of ML concepts with real-world analogies
- **AdaBoost Step-by-Step**: Visual walkthrough of how AdaBoost works with weighted examples
- **Algorithm Comparisons**: Side-by-side comparison of different boosting algorithms

### 2. Real-Time Algorithm Training
- **Live Training**: Watch algorithms train in real-time with WebSocket updates
- **Progress Tracking**: Real-time loss curves and metrics updates
- **Multiple Algorithms**: Support for AdaBoost, Gradient Boosting, XGBoost, LightGBM, and CatBoost
- **Parameter Controls**: Adjust hyperparameters (n_estimators, learning_rate, max_depth) on the fly
- **Training Logs**: Detailed console logs showing training progress

### 3. Decision Boundary Visualizations
- **Real Decision Boundaries**: Visualize actual decision boundaries from trained models using matplotlib
- **Iterative Progression**: See how decision boundaries evolve through training iterations
- **Carousel Navigation**: Browse through different iterations of the decision boundary
- **Auto-Play Slideshow**: Play button automatically advances through decision boundaries (2-second intervals)
- **Manual Control**: Pause/play and navigate manually with previous/next buttons or dot indicators
- **Dark Theme**: Beautiful dark-themed visualizations optimized for readability
- **Real Model Integration**: Fetches actual decision boundaries from backend API
- **Balanced Visualization**: Automatically samples equal numbers from each class for clear visualization

### 4. Tree Visualizations
- **Interactive Decision Trees**: Visualize individual trees from boosting algorithms
- **Animated Tree Building**: Watch trees being constructed step-by-step
- **Feature Importance**: See which features matter most in predictions
- **Tree Structure**: Understand how decision trees make predictions

### 5. User Interface Features
- **Responsive Design**: Works on desktop and mobile devices
- **Dark Theme**: Modern dark theme with purple/blue gradients
- **Large, Readable Text**: All text sizes optimized for visibility (base font: 20px)
- **Smooth Animations**: Polished transitions and hover effects
- **Keyboard Shortcuts**: Ctrl+K to open search, Escape to close

## 📚 What This Project Explains

### Chapter 1: Machine Learning Basics
- **Supervised Learning**: Learning from labeled examples
- **Unsupervised Learning**: Finding patterns without labels
- **Reinforcement Learning**: Learning through trial and error
- **Real-world Analogy**: Teacher learning patterns from student cases

### Chapter 2: Decision Trees
- **What are Decision Trees**: Flowchart-like structures for making decisions
- **How They Work**: Asking yes/no questions to reach conclusions
- **Interactive Example**: "Is it raining?" → "Do we have umbrella?" → Decision
- **Concepts**: Entropy, Information Gain, Splitting
- **Visual Decision Tree**: Large, interactive D3.js visualization

### Chapter 3: Ensemble Learning
- **Power of Teams**: Combining multiple trees for better predictions
- **Voting Mechanism**: How ensemble methods make decisions
- **Weak Learners**: Simple models that work together
- **Comparison**: Single tree vs. ensemble performance

### Chapter 4: Boosting Algorithms
- **Learning from Mistakes**: How boosting improves by focusing on errors
- **AdaBoost**: Adaptive boosting with weighted examples
- **Gradient Boosting**: Step-by-step error correction
- **XGBoost**: Fast, optimized gradient boosting
- **Visual Explanations**: Step-by-step walkthroughs with student examples
- **Team Learning Analogy**: Students building on each other's work

### Chapter 5: Experiments & Comparison
- **Performance Comparison**: Compare all algorithms side-by-side
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Real Results**: Actual performance on stroke prediction dataset

## 🏗️ Project Structure

```
Project/
├── backend/                    # FastAPI backend server
│   ├── main.py                # Main API server with WebSocket support
│   └── requirements.txt       # Python dependencies
├── src/                       # React frontend
│   ├── pages/                 # Main page components
│   │   ├── MLLearningJourneyPage.js    # Main learning journey (Alex's story)
│   │   ├── AdaBoostPage.js             # AdaBoost algorithm page
│   │   ├── GradientBoostingPage.js     # Gradient Boosting page
│   │   ├── XGBoostPage.js              # XGBoost page
│   │   ├── TreeVisualizationPage.js    # Decision tree visualization
│   │   └── DecisionBoundaryPage.js     # Decision boundary visualization
│   ├── components/            # Reusable components
│   │   ├── RealDecisionBoundaryCarousel.js  # Decision boundary carousel
│   │   ├── RealTrainingVisualization.js      # Training visualization
│   │   └── AlgorithmPage.js                  # Algorithm training interface
│   ├── services/              # API and WebSocket services
│   │   ├── apiService.js      # REST API calls
│   │   └── socketService.js  # WebSocket communication
│   ├── App.js                 # Main app component with routing
│   └── App.css                # Global styles (9500+ lines)
├── data/                      # Datasets
│   └── stroke_data_balanced.csv  # Stroke prediction dataset
├── start_servers.py           # Automated server startup script
└── README.md                  # This file
```

## 🚀 Quick Start

### Option 1: Automated Startup (Recommended)

**To launch servers, run the `start_servers.py` script:**

**Windows:**
```bash
python start_servers.py
```

**Or use batch files:**
```bash
start_servers.bat
# or
start_servers.ps1
```

**Linux/Mac:**
```bash
python start_servers.py
# or
./start.sh
```

**What `start_servers.py` does:**
- Automatically starts the FastAPI backend server on port 8000
- Automatically starts the React frontend on port 3000
- Opens your browser to http://localhost:3000
- Monitors both servers and handles cleanup
- Checks for dependencies and installs them if needed
- Provides real-time server output monitoring
- Gracefully shuts down both servers when you press Ctrl+C

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
   Backend runs on: http://localhost:8000

#### Frontend Setup

1. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

2. **Start the frontend development server:**
   ```bash
   npm start
   ```
   Frontend runs on: http://localhost:3000

## 🎓 How to Use the Learning Journey

### Navigation
- **Home Page**: Start at the ML Learning Journey homepage
- **Chapter Navigation**: Use the progress bar or navigation buttons to move between chapters
- **Search**: Press `Ctrl+K` (or `Cmd+K` on Mac) to search for concepts, algorithms, or topics
- **Direct Links**: Search for "gradient boost", "adaboost", etc. to jump directly to algorithm explanations

### Learning Path
1. **Chapter 1**: Learn what Machine Learning is and its types
2. **Chapter 2**: Understand Decision Trees with the rain/umbrella example
3. **Chapter 3**: Discover Ensemble Learning and why teams work better
4. **Chapter 4**: Master Boosting Algorithms (AdaBoost, Gradient Boosting, XGBoost)
5. **Chapter 5**: Run experiments and compare algorithm performance

### Interactive Elements
- **Decision Tree Diagram**: Compact, readable visualization showing decision-making process with optimized text sizes (16px text, 32px icons)
- **AdaBoost Steps**: Auto-play slideshow (4.5 seconds per step) or manually navigate through step-by-step AdaBoost training process
- **Play Button**: Start auto-play to automatically advance through AdaBoost explanation steps - stops at last step automatically
- **Step Indicator**: Visual indicator showing current step (e.g., "Step 1 / 6")
- **Algorithm Tabs**: Switch between Overview, AdaBoost, Gradient Boosting, and XGBoost explanations
- **Search Results**: Click any search result to navigate directly to that section
- **Smart Auto-Play**: Auto-play automatically stops when manually navigating, can be resumed with play button

## 🔬 Algorithm Training Features

### Individual Algorithm Pages
Navigate to:
- `/adaboost` - AdaBoost Classifier
- `/gradient-boosting` - Gradient Boosting Classifier
- `/xgboost` - XGBoost Classifier
- `/lightgbm` - LightGBM Classifier
- `/catboost` - CatBoost Classifier

Each page provides:
- **Dataset Preview**: First 10 rows of the stroke dataset
- **Parameter Controls**: Adjust n_estimators, learning_rate, max_depth
- **Training Controls**: Start/Stop training, clear logs
- **Real-time Progress**: Live updates via WebSocket
- **Metrics Display**: Accuracy, Precision, Recall, F1-Score with improved visibility on dark background
- **Loss Curves**: Real-time training loss visualization
- **Feature Importance**: Charts showing which features matter most
- **Stroke Prediction**: Interactive form to predict stroke risk for new patients
- **Prediction Validation**: Clear error message if attempting to predict before training model

### Decision Boundary Visualization
- **Real Boundaries**: See actual decision boundaries from trained models (matplotlib-generated PNGs)
- **Iteration Progression**: Navigate through training iterations with smooth transitions
- **Auto-Play Mode**: Play button automatically cycles through all decision boundaries (2-second intervals)
- **Manual Navigation**: Previous/Next buttons and dot indicators for direct navigation
- **Metadata Display**: View accuracy, iteration number, data points, and algorithm type
- **Clean Visualization**: No overlapping text, optimized layout with balanced class sampling
- **Auto-Stop on Manual**: Auto-play stops when manually navigating, resumes on play button click
- **Looping**: Carousel loops continuously when auto-playing

### Tree Visualization
- **Interactive Trees**: Visualize individual decision trees
- **Animated Building**: Watch trees being constructed
- **Feature Splits**: See how features are used for splitting
- **Prediction Paths**: Understand how predictions are made

## 🎨 Design & Styling

### Theme
- **Dark Background**: Deep purple/blue gradient background
- **Color Scheme**: Purple (#667eea), Blue (#8b5cf6), Cyan (#00d4ff)
- **Typography**: Large, readable fonts (base: 20px, increased from 16px)
- **Icons**: Custom icon system with Flaticon integration

### Text Sizes
- **Base Font**: 20px (25% larger than standard)
- **Decision Tree Text**: 16px inside boxes (optimized for fit), 32px icons
- **Diagram Text**: 20-32px for axis labels and titles
- **Branch Labels**: 18px for Yes/No indicators
- **All text optimized**: Every text element sized for maximum readability and proper fit within diagram boxes

### Responsive Design
- **Mobile Friendly**: Adapts to different screen sizes
- **Flexible Layouts**: Grid systems that adjust to content
- **Touch Support**: Works on tablets and mobile devices

## 🔍 Search Functionality

### How It Works
- **Smart Matching**: Searches through all chapters, concepts, and algorithms
- **Algorithm Detection**: Recognizes "gradient boost", "adaboost", "xgboost" variations
- **Direct Navigation**: Clicking a result navigates directly to that section
- **Tab Switching**: Automatically opens the correct algorithm tab when searching for boosting algorithms

### Search Examples
- Search "gradient boost" → Navigates to Gradient Boosting tab in Chapter 4
- Search "decision tree" → Goes to Chapter 2 with decision tree visualization
- Search "entropy" → Finds entropy concept explanation
- Search "adaboost" → Opens AdaBoost explanation with step-by-step guide

## 📊 Dataset

### Stroke Prediction Dataset
- **Location**: `data/stroke_data_balanced.csv`
- **Features**: 
  - Numerical: `age`, `avg_glucose_level`, `bmi`
  - Categorical: `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`
  - Medical: `hypertension`, `heart_disease`
- **Target**: `stroke` (binary: 0 = No Stroke, 1 = Stroke)
- **Size**: Balanced dataset for fair model comparison

## 🔌 API Endpoints

### REST API (Backend)
- `GET /` - API status and health check
- `GET /algorithms` - List all available algorithms
- `GET /dataset/preview?rows=10&dataset={name}` - Get dataset preview
- `POST /dataset/upload` - Upload new dataset (CSV)
- `GET /plot/boosting-boundary?algorithm={alg}&n_estimators={n}` - Get decision boundary plot (base64 PNG)
- `POST /predict` - Predict stroke risk for new patient data (requires trained model)

### WebSocket Events (Real-time)
- `connect` - Client connects to server
- `start_training` - Start algorithm training with parameters
- `training_progress` - Real-time training updates (loss, metrics)
- `training_completed` - Training finished with final results
- `training_error` - Training failed with error message

## 🛠️ Tech Stack

### Frontend
- **React 18.2.0**: Modern React with hooks
- **React Router 6.8.1**: Client-side routing
- **D3.js 7.9.0**: Data visualization (decision trees, charts)
- **Socket.IO Client 4.7.4**: Real-time WebSocket communication
- **Recharts 2.8.0**: Chart visualizations
- **Custom CSS**: 9500+ lines of custom styling

### Backend
- **FastAPI**: Modern Python web framework
- **Python-SocketIO 5.10.0**: WebSocket server
- **scikit-learn 1.3.2**: ML algorithms (AdaBoost, Gradient Boosting)
- **XGBoost 2.0.2**: Extreme Gradient Boosting
- **LightGBM 4.1.0**: Light Gradient Boosting Machine
- **CatBoost 1.2.2**: Categorical Boosting
- **Matplotlib**: Decision boundary plotting
- **Pandas & NumPy**: Data manipulation

## 📖 Educational Content Explained

### Decision Tree Example
The application uses a simple, relatable example with an optimized, readable visualization:
- **Question**: "Is it raining?"
  - **If Yes**: "Do we have umbrella?"
    - **If Yes**: "Go out" 🚶
    - **If No**: "Don't go" 🏠
  - **If No**: "Go out" 🚶

**Visual Features**:
- Compact node boxes with properly sized text (16px)
- Clear branch labels (Yes/No) with readable font (18px)
- Icons sized appropriately (32px) for visual balance
- All text fits comfortably within diagram elements

This example teaches:
- How decision trees ask questions
- How decisions branch based on answers
- How final outcomes are reached

### AdaBoost Explanation
The app explains AdaBoost through a "student learning" analogy with an interactive slideshow:
- **Step 1**: Initial setup - all students start with equal weights
- **Step 2**: First tree makes predictions on the dataset
- **Step 3**: Weight update - increase weights for misclassified examples
- **Step 4**: Second tree focuses on previously misclassified examples
- **Step 5**: Final ensemble combines all trees with weighted voting
- **Step 6**: Result - improved accuracy through adaptive boosting

**Interactive Features**:
- **Auto-Play**: Play button automatically advances through steps (4.5 seconds per step)
- **Manual Navigation**: Previous/Next buttons for step-by-step exploration
- **Step Indicator**: Shows current step (e.g., "Step 1 / 6")

This teaches:
- Adaptive learning (focusing on mistakes)
- Weighted examples (hard examples get more attention)
- Ensemble voting (combining multiple learners)

### Boosting Concepts
- **Weak Learners**: Simple models that are slightly better than chance
- **Sequential Learning**: Building models one after another
- **Error Correction**: Each new model focuses on previous errors
- **Weighted Training**: Examples that were misclassified get higher weights

## 🎯 Key Improvements Made

### Recent Updates (Latest)
- **Auto-Play Features**: Added play buttons for decision boundary carousel (2s intervals) and AdaBoost explanation (4.5s intervals)
- **Text Size Optimization**: Reduced decision tree text to 16px for proper fit within compact boxes (200-300px width, 60-80px height)
- **Prediction Validation**: Added check to prevent predictions before model training with user-friendly error message
- **Gradient Animations**: Applied animated gradient effects to all interactive buttons (play/pause, navigation, primary buttons)
- **Improved UX**: Auto-play stops on manual navigation, resumes on play button click, loops continuously
- **Decision Boundary Updates**: Integrated real matplotlib-generated decision boundaries with balanced visualization sampling
- **Metrics Visibility**: Enhanced final metrics display with improved contrast and visibility on dark background
- **Model Training Tracking**: Backend tracks training status to prevent invalid predictions

### Text Size Optimization
- Increased base font size from 16px to 20px (25% larger)
- Decision tree text: Optimized to 16px for proper fit within compact boxes
- Icons: Sized to 32px for visual balance
- Branch labels: Reduced to 18px for readability

### User Experience
- **Auto-Play Features**: Added play buttons for decision boundary carousel (2s intervals) and AdaBoost explanation slideshow (4.5s intervals)
- **Smart Navigation**: Auto-play stops when manually navigating, resumes when play button is clicked
- **Prediction Validation**: Clear error message ("Please train the model first before making predictions") when attempting to predict before training model
- **Step Indicators**: Visual step indicators showing current position (e.g., "Step 1 / 6" for AdaBoost, "1 / 8" for decision boundaries)
- **Auto-Stop**: AdaBoost auto-play automatically stops at the last step (Step 6)
- **Looping**: Decision boundary carousel loops continuously when auto-playing
- Removed comparison tab from Chapter 4 (cleaner interface)
- Moved "REAL MODEL BOUNDARY" badge to metadata section (no overlap)
- Improved search to navigate directly to algorithm tabs
- Simplified decision tree (removed windy branch, kept rain/umbrella)

### Visual Design
- **Optimized Decision Tree**: Compact node boxes (200-300px width, 60-80px height) with readable text (16px)
- **Proper Text Fit**: All text sizes reduced to fit comfortably within diagram boxes
  - Node text: 16px (reduced from 72px)
  - Icons: 32px (reduced from 120px)
  - Branch labels: 18px (reduced from 36px)
  - Branch label ellipses: 30x20px (reduced from 50x35px)
- **Animated Buttons**: Gradient animation effects on all interactive buttons (play/pause, navigation, primary actions)
- **Metrics Display**: Enhanced visibility with improved contrast, borders, and shadows on dark background
- Enhanced spacing and padding throughout
- Better color contrast for readability

### Content Updates
- Changed Alex's pronouns from they/them to he/him throughout
- Updated decision tree to meaningful rain/umbrella example
- Improved algorithm explanations with better analogies
- Enhanced step-by-step AdaBoost walkthrough

## 🐛 Troubleshooting

### Backend Issues
- **Port 8000 in use**: Change port in `backend/main.py` or stop conflicting service
- **Missing dependencies**: Run `pip install -r backend/requirements.txt`
- **Dataset not found**: Ensure `data/stroke_data_balanced.csv` exists
- **ML libraries missing**: Install with `pip install xgboost lightgbm catboost`

### Frontend Issues
- **Port 3000 in use**: React will prompt to use another port
- **WebSocket connection failed**: Ensure backend is running on port 8000
- **Build errors**: Run `npm install` to ensure all dependencies are installed
- **Styles not loading**: Clear browser cache and restart dev server

### Common Solutions
```bash
# Clean install (if having issues)
rm -rf node_modules package-lock.json
npm install

# Python dependencies
pip install --upgrade -r backend/requirements.txt

# Check if servers are running
# Backend: http://localhost:8000/docs
# Frontend: http://localhost:3000
```

## 📝 Development Notes

### File Organization
- **Large CSS File**: `App.css` contains 9500+ lines of custom styling
- **Component Structure**: Pages contain main content, components are reusable
- **Service Layer**: API and WebSocket logic separated into services
- **State Management**: React hooks (useState, useEffect) for state

### Key Components
- **MLLearningJourneyPage**: Main educational journey (2000+ lines) with AdaBoost auto-play slideshow
- **RealDecisionBoundaryCarousel**: Decision boundary visualization with auto-play carousel
- **AlgorithmPage**: Reusable algorithm training interface with prediction validation
- **TreeVisualizationPage**: Interactive decision tree visualization

### Styling Approach
- **CSS Variables**: Used for consistent theming
- **Responsive Design**: Media queries for mobile/tablet/desktop
- **Dark Theme**: Optimized color scheme for readability
- **Animations**: Smooth transitions and hover effects

## 🎓 Learning Outcomes

After using this application, users will understand:
1. **What Machine Learning is** and its different types
2. **How Decision Trees work** through interactive examples
3. **Why Ensembles are powerful** (multiple models > single model)
4. **How Boosting works** (learning from mistakes)
5. **Differences between algorithms** (AdaBoost vs Gradient Boosting vs XGBoost)
6. **Real-world application** (stroke prediction using medical data)

## 📄 License

This project is for educational and demonstration purposes.

## 🙏 Acknowledgments

- Built with React, FastAPI, and modern ML libraries
- Uses D3.js for beautiful data visualizations
- Stroke dataset for realistic ML demonstrations
- Educational approach inspired by R2D3 and other ML education resources

---

**Note**: This is a comprehensive educational tool designed to make machine learning concepts accessible and engaging through interactive visualizations and story-driven learning.
