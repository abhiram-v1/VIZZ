import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Icon } from '../components/Icons';

// Marvel-style Decision Boundary Visualization Page
const DecisionBoundaryPage = () => {
  const { algorithm } = useParams();
  const [plots, setPlots] = useState({});
  const [loading, setLoading] = useState(true);
  const [currentStage, setCurrentStage] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  const stages = [
    { id: 'initial', title: 'Initial Decision', subtitle: 'Simple Class-Based Split', description: 'The model starts with a basic rule: passenger class determines survival probability.' },
    { id: 'second_tree', title: 'Age Consideration', subtitle: 'Multi-Factor Analysis', description: 'Adding age as a second decision factor creates more sophisticated boundaries.' },
    { id: 'final_ensemble', title: 'Final Ensemble', subtitle: 'Optimal Classification', description: 'Complex decision boundaries achieve maximum classification accuracy.' }
  ];

  useEffect(() => {
    const fetchPlots = async () => {
      setLoading(true);
      const plotData = {};
      
      for (const stage of stages) {
        try {
          const response = await fetch(`http://localhost:8000/plot/decision-boundary?stage=${stage.id}`);
          const data = await response.json();
          plotData[stage.id] = data.plot_data;
        } catch (error) {
          console.error(`Error fetching plot for ${stage.id}:`, error);
        }
      }
      
      setPlots(plotData);
      setLoading(false);
    };

    fetchPlots();
  }, []);

  const nextStage = () => {
    if (currentStage < stages.length - 1) {
      setIsAnimating(true);
      setTimeout(() => {
        setCurrentStage(currentStage + 1);
        setIsAnimating(false);
      }, 500);
    }
  };

  const prevStage = () => {
    if (currentStage > 0) {
      setIsAnimating(true);
      setTimeout(() => {
        setCurrentStage(currentStage - 1);
        setIsAnimating(false);
      }, 500);
    }
  };

  if (loading) {
    return (
      <div className="marvel-loading">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <h2>Generating Decision Boundaries...</h2>
          <p>Creating high-quality visualizations</p>
        </div>
      </div>
    );
  }

  const currentStageData = stages[currentStage];

  return (
    <div className="decision-boundary-page">
      {/* Marvel-style Header */}
      <div className="marvel-header">
        <div className="header-content">
          <h1 className="marvel-title">
            <span className="title-main">Decision Boundary</span>
            <span className="title-accent">Analysis</span>
          </h1>
          <p className="marvel-subtitle">Advanced Machine Learning Visualization</p>
          <div className="algorithm-badge">
            <span className="badge-text">{algorithm?.toUpperCase() || 'BOOSTING'}</span>
          </div>
        </div>
        <div className="header-particles">
          <div className="particle"></div>
          <div className="particle"></div>
          <div className="particle"></div>
          <div className="particle"></div>
          <div className="particle"></div>
        </div>
      </div>

      {/* Navigation Controls */}
      <div className="navigation-controls">
        <button 
          className="nav-button prev" 
          onClick={prevStage}
          disabled={currentStage === 0}
        >
          <span className="button-icon">←</span>
          <span className="button-text">Previous</span>
        </button>
        
        <div className="stage-indicator">
          <span className="current-stage">{currentStage + 1}</span>
          <span className="stage-separator">/</span>
          <span className="total-stages">{stages.length}</span>
        </div>
        
        <button 
          className="nav-button next" 
          onClick={nextStage}
          disabled={currentStage === stages.length - 1}
        >
          <span className="button-text">Next</span>
          <span className="button-icon">→</span>
        </button>
      </div>

      {/* Main Visualization Container */}
      <div className="visualization-container">
        <div className={`stage-card ${isAnimating ? 'animating' : ''}`}>
          {/* Stage Header */}
          <div className="stage-header">
            <div className="stage-number">
              <span>{currentStage + 1}</span>
            </div>
            <div className="stage-info">
              <h2 className="stage-title">{currentStageData.title}</h2>
              <h3 className="stage-subtitle">{currentStageData.subtitle}</h3>
              <p className="stage-description">{currentStageData.description}</p>
            </div>
          </div>

          {/* Large Decision Boundary Plot */}
          <div className="plot-container">
            {plots[currentStageData.id] && (
              <img 
                src={`data:image/png;base64,${plots[currentStageData.id]}`}
                alt={`Decision Boundary - ${currentStageData.title}`}
                className="decision-boundary-plot"
              />
            )}
          </div>

          {/* Stage Details */}
          <div className="stage-details">
            <div className="detail-card">
              <div className="detail-icon"><Icon name="target" size={24} /></div>
              <div className="detail-content">
                <h4>Decision Rules</h4>
                <p>
                  {currentStage === 0 && "Pclass ≤ 2.5"}
                  {currentStage === 1 && "Pclass ≤ 2.5 AND Age ≤ 60"}
                  {currentStage === 2 && "Multiple thresholds for optimal classification"}
                </p>
              </div>
            </div>
            
            <div className="detail-card">
              <div className="detail-icon"><Icon name="chart" size={24} /></div>
              <div className="detail-content">
                <h4>Complexity</h4>
                <p>
                  {currentStage === 0 && "Simple (1 rule)"}
                  {currentStage === 1 && "Moderate (2 rules)"}
                  {currentStage === 2 && "Complex (Multiple rules)"}
                </p>
              </div>
            </div>
            
            <div className="detail-card">
              <div className="detail-icon"><Icon name="bolt" size={24} /></div>
              <div className="detail-content">
                <h4>Performance</h4>
                <p>
                  {currentStage === 0 && "Basic accuracy"}
                  {currentStage === 1 && "Improved accuracy"}
                  {currentStage === 2 && "Optimal accuracy"}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="progress-container">
        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${((currentStage + 1) / stages.length) * 100}%` }}
          ></div>
        </div>
        <div className="progress-labels">
          {stages.map((stage, index) => (
            <span 
              key={stage.id}
              className={`progress-label ${index <= currentStage ? 'active' : ''}`}
            >
              {stage.title}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};

export default DecisionBoundaryPage;
