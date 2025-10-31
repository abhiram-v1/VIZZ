import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate } from 'react-router-dom';
import './App.css';
import AdaBoostPage from './pages/AdaBoostPage';
import GradientBoostingPage from './pages/GradientBoostingPage';
import XGBoostPage from './pages/XGBoostPage';
import MLLearningJourneyPage from './pages/MLLearningJourneyPage';
import DecisionBoundaryPage from './pages/DecisionBoundaryPage';
import { 
  MainLogo, 
  AdaBoostIcon, 
  GradientBoostingIcon, 
  XGBoostIcon
} from './components/Logos';

function App() {
  const [showNavbar, setShowNavbar] = useState(true);

  useEffect(() => {
    const handleScroll = () => {
      // Check scroll position on window and document
      const scrollPosition = window.scrollY || window.pageYOffset || document.documentElement.scrollTop;
      setShowNavbar(scrollPosition <= 0);
    };

    // Initial check
    handleScroll();

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <Router>
      <div className="App">
        <nav className={`navbar ${showNavbar ? 'navbar-visible' : 'navbar-hidden'}`}>
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
              <Link to="/ml-journey" className="nav-link ml-journey-link">
                <span className="nav-icon"><i className="fi fi-br-graduation-cap" aria-hidden="true"></i></span>
                <span>ML Learning Journey</span>
              </Link>
            </div>
          </div>
        </nav>
        
        <main className="main-content">
          <Routes>
            <Route path="/" element={<MLLearningJourneyPage />} />
            <Route path="/adaboost" element={<AdaBoostPage />} />
            <Route path="/gradient-boosting" element={<GradientBoostingPage />} />
            <Route path="/xgboost" element={<XGBoostPage />} />
            <Route path="/ml-journey" element={<MLLearningJourneyPage />} />
            <Route path="/decision-boundary/:algorithm" element={<DecisionBoundaryPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
