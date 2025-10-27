import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate } from 'react-router-dom';
import './App.css';
import AdaBoostPage from './pages/AdaBoostPage';
import GradientBoostingPage from './pages/GradientBoostingPage';
import XGBoostPage from './pages/XGBoostPage';
import LightGBMPage from './pages/LightGBMPage';
import CatBoostPage from './pages/CatBoostPage';
import AllAlgorithmsPage from './pages/AllAlgorithmsPage';
import TreeVisualizationPage from './pages/TreeVisualizationPage';
import MLLearningJourneyPage from './pages/MLLearningJourneyPage';
import DecisionBoundaryPage from './pages/DecisionBoundaryPage';
import { 
  MainLogo, 
  TreeVizIcon, 
  AdaBoostIcon, 
  GradientBoostingIcon, 
  XGBoostIcon, 
  LightGBMIcon, 
  CatBoostIcon, 
  CompareIcon 
} from './components/Logos';

function App() {
  return (
    <Router>
      <div className="App">
        <nav className="navbar">
          <div className="nav-container">
            <div className="nav-brand">
              <MainLogo size={32} className="main-logo" />
              <h1 className="nav-title">Boosting Algorithms Demo</h1>
            </div>
            <div className="nav-links">
              <Link to="/adaboost" className="nav-link">
                <AdaBoostIcon size={20} className="nav-icon" />
                <span>AdaBoost</span>
              </Link>
              <Link to="/gradient-boosting" className="nav-link">
                <GradientBoostingIcon size={20} className="nav-icon" />
                <span>Gradient Boosting</span>
              </Link>
              <Link to="/xgboost" className="nav-link">
                <XGBoostIcon size={20} className="nav-icon" />
                <span>XGBoost</span>
              </Link>
              <Link to="/lightgbm" className="nav-link">
                <LightGBMIcon size={20} className="nav-icon" />
                <span>LightGBM</span>
              </Link>
              <Link to="/catboost" className="nav-link">
                <CatBoostIcon size={20} className="nav-icon" />
                <span>CatBoost</span>
              </Link>
              <Link to="/tree-viz" className="nav-link tree-viz-link">
                <TreeVizIcon size={20} className="nav-icon" />
                <span>Live Tree Viz</span>
              </Link>
              <Link to="/ml-journey" className="nav-link ml-journey-link">
                <span className="nav-icon">🎓</span>
                <span>ML Learning Journey</span>
              </Link>
              <Link to="/all" className="nav-link">
                <CompareIcon size={20} className="nav-icon" />
                <span>Compare All</span>
              </Link>
            </div>
          </div>
        </nav>
        
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Navigate to="/ml-journey" replace />} />
            <Route path="/adaboost" element={<AdaBoostPage />} />
            <Route path="/gradient-boosting" element={<GradientBoostingPage />} />
            <Route path="/xgboost" element={<XGBoostPage />} />
            <Route path="/lightgbm" element={<LightGBMPage />} />
            <Route path="/catboost" element={<CatBoostPage />} />
            <Route path="/tree-viz" element={<TreeVisualizationPage />} />
            <Route path="/ml-journey" element={<MLLearningJourneyPage />} />
            <Route path="/decision-boundary/:algorithm" element={<DecisionBoundaryPage />} />
            <Route path="/all" element={<AllAlgorithmsPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
