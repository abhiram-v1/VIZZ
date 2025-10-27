import React, { useState, useEffect } from 'react';

const AnimatedDecisionBoundary = ({ algorithm = 'adaboost', isVisible = true }) => {
  const [animationPlaying, setAnimationPlaying] = useState(false);
  const [currentIteration, setCurrentIteration] = useState(0);
  const [boundaryData, setBoundaryData] = useState(null);

  // Animation effect
  useEffect(() => {
    let interval;
    if (animationPlaying) {
      interval = setInterval(() => {
        setCurrentIteration(prev => (prev + 1) % 5);
      }, 2000); // Change iteration every 2 seconds
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [animationPlaying]);

  // Generate mock boundary data for demonstration
  useEffect(() => {
    const generateMockData = () => {
      const iterations = [];
      for (let i = 0; i < 5; i++) {
        iterations.push({
          iteration: i + 1,
          accuracy: 65 + i * 8,
          boundary: `data:image/svg+xml;base64,${btoa(`
            <svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
              <rect width="400" height="300" fill="#f8f9fa" stroke="#dee2e6"/>
              <g>
                ${Array.from({length: 50}, (_, j) => {
                  const x = j * 8;
                  const y = 150 + Math.sin(x * 0.02 + i * 0.5) * 50 + i * 10;
                  return `<circle cx="${x}" cy="${y}" r="3" fill="${j % 2 === 0 ? '#e74c3c' : '#2ecc71'}"/>`;
                }).join('')}
                <path d="M0,${150 + Math.sin(0) * 50 + i * 10} ${Array.from({length: 50}, (_, j) => {
                  const x = j * 8;
                  const y = 150 + Math.sin(x * 0.02 + i * 0.5) * 50 + i * 10;
                  return `L${x},${y}`;
                }).join('')}" stroke="#2c3e50" stroke-width="3" fill="none" stroke-dasharray="5,5"/>
              </g>
            </svg>
          `)}`
        });
      }
      setBoundaryData(iterations);
    };

    generateMockData();
  }, []);

  if (!isVisible || !boundaryData) return null;

  const currentBoundary = boundaryData[currentIteration];

  return (
    <div className="animated-decision-boundary">
      <div className="boundary-header">
        <h3>🎯 Animated Decision Boundary Evolution</h3>
        <p className="boundary-description">
          Watch how the {algorithm} algorithm refines its decision boundaries as it learns
        </p>
      </div>
      
      <div className="animated-boundary-container">
        <div className="boundary-controls">
          <button 
            className="control-btn play-pause"
            onClick={() => setAnimationPlaying(!animationPlaying)}
          >
            {animationPlaying ? '⏸️ Pause' : '▶️ Play'}
          </button>
          <button 
            className="control-btn"
            onClick={() => setCurrentIteration(Math.max(0, currentIteration - 1))}
          >
            ⏮️ Previous
          </button>
          <button 
            className="control-btn"
            onClick={() => setCurrentIteration(Math.min(4, currentIteration + 1))}
          >
            ⏭️ Next
          </button>
          <button 
            className="control-btn"
            onClick={() => setCurrentIteration(0)}
          >
            🔄 Reset
          </button>
        </div>
        
        <div className="boundary-visualization">
          <div className="iteration-info">
            <div className="iteration-number">Iteration: {currentIteration + 1} / 5</div>
            <div className="accuracy-display">
              Accuracy: {currentBoundary.accuracy}%
            </div>
          </div>
          
          <div className="boundary-plot-container">
            <img 
              src={currentBoundary.boundary}
              alt={`${algorithm} decision boundary at iteration ${currentIteration + 1}`}
              className="boundary-plot"
            />
          </div>
          
          <div className="boundary-info">
            <div className="info-item">
              <span className="info-label">Algorithm:</span>
              <span className="info-value">{algorithm.toUpperCase()}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Current Iteration:</span>
              <span className="info-value">{currentIteration + 1}/5</span>
            </div>
            <div className="info-item">
              <span className="info-label">Learning Stage:</span>
              <span className="info-value">
                {currentIteration <= 1 ? 'Early Learning' : 
                 currentIteration <= 3 ? 'Mid Learning' : 'Advanced Learning'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnimatedDecisionBoundary;
