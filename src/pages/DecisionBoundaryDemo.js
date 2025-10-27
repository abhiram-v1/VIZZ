import React, { useState } from 'react';
import AnimatedDecisionBoundary from '../components/AnimatedDecisionBoundary';

const DecisionBoundaryDemo = () => {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('adaboost');
  const [showDemo, setShowDemo] = useState(false);

  const algorithms = [
    { key: 'adaboost', name: 'AdaBoost', description: 'Adaptive Boosting' },
    { key: 'xgboost', name: 'XGBoost', description: 'Extreme Gradient Boosting' },
    { key: 'gradient', name: 'Gradient Boosting', description: 'Gradient Boosting' }
  ];

  return (
    <div className="decision-boundary-demo-page">
      <div className="demo-header">
        <h1>🧠 Animated Decision Boundary Demo</h1>
        <p>Experience how boosting algorithms learn to make better predictions</p>
      </div>

      <div className="demo-controls">
        <div className="algorithm-selector">
          <h3>Choose Algorithm:</h3>
          <div className="algorithm-buttons">
            {algorithms.map(algo => (
              <button
                key={algo.key}
                className={`algorithm-btn ${selectedAlgorithm === algo.key ? 'active' : ''}`}
                onClick={() => setSelectedAlgorithm(algo.key)}
              >
                <div className="algorithm-name">{algo.name}</div>
                <div className="algorithm-desc">{algo.description}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="demo-actions">
          <button 
            className="demo-btn primary"
            onClick={() => setShowDemo(!showDemo)}
          >
            {showDemo ? '🔄 Hide Demo' : '🚀 Show Demo'}
          </button>
        </div>
      </div>

      {showDemo && (
        <div className="demo-content">
          <AnimatedDecisionBoundary 
            algorithm={selectedAlgorithm} 
            isVisible={true}
          />
        </div>
      )}

      <div className="demo-info">
        <h3>🎯 What You're Seeing</h3>
        <ul>
          <li><strong>Red Dots:</strong> Patients who had a stroke</li>
          <li><strong>Green Dots:</strong> Patients who did not have a stroke</li>
          <li><strong>Animated Line:</strong> The decision boundary that separates stroke vs no-stroke</li>
          <li><strong>Learning Process:</strong> Watch how the boundary gets more accurate with each iteration</li>
        </ul>

        <h3>🔬 The Science Behind It</h3>
        <ul>
          <li><strong>Iteration 1:</strong> Simple boundary - makes many mistakes (~65% accuracy)</li>
          <li><strong>Iteration 2-3:</strong> Learns from errors, boundary becomes more complex (~75-85% accuracy)</li>
          <li><strong>Iteration 4-5:</strong> Refined boundary - much more accurate (~90-95% accuracy)</li>
          <li><strong>Result:</strong> A model that can predict stroke risk with high accuracy</li>
        </ul>
      </div>
    </div>
  );
};

export default DecisionBoundaryDemo;
