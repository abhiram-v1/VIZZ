import React, { useState, useEffect } from 'react';
import { Icon } from './Icons';

const RealTrainingVisualization = ({ algorithm, isTraining, progressData }) => {
  const [trainingData, setTrainingData] = useState(null);
  const [currentIteration, setCurrentIteration] = useState(0);
  const [modelPredictions, setModelPredictions] = useState(null);

  // Fetch real training data from backend
  useEffect(() => {
    const fetchTrainingData = async () => {
      try {
        const response = await fetch(`http://localhost:8000/training-data/${algorithm}`);
        const data = await response.json();
        setTrainingData(data);
      } catch (error) {
        console.error('Error fetching training data:', error);
      }
    };

    if (isTraining) {
      fetchTrainingData();
    }
  }, [algorithm, isTraining]);

  // Get real model predictions for current iteration
  useEffect(() => {
    const fetchModelPredictions = async () => {
      if (trainingData && currentIteration >= 0) {
        try {
          const response = await fetch(`http://localhost:8000/model-predictions/${algorithm}?iteration=${currentIteration}`);
          const data = await response.json();
          setModelPredictions(data);
        } catch (error) {
          console.error('Error fetching model predictions:', error);
        }
      }
    };

    fetchModelPredictions();
  }, [algorithm, currentIteration, trainingData]);

  if (!isTraining || !trainingData || !modelPredictions) {
    return (
      <div className="real-training-visualization">
        <div className="loading-message">
          <h3><Icon name="sync" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Loading Real Training Data...</h3>
          <p>Connecting to actual {algorithm} model training...</p>
        </div>
      </div>
    );
  }

  const { X_train, y_train, feature_names } = trainingData;
  const { predictions, accuracy, decision_boundary_data } = modelPredictions;

  // Extract 2D features for visualization (age and glucose)
  const ageIdx = feature_names.indexOf('age');
  const glucoseIdx = feature_names.indexOf('avg_glucose_level');
  
  if (ageIdx === -1 || glucoseIdx === -1) {
    return (
      <div className="real-training-visualization">
        <div className="error-message">
          <h3><Icon name="timesCircle" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Feature Error</h3>
          <p>Required features (age, avg_glucose_level) not found in dataset</p>
        </div>
      </div>
    );
  }

  const X_2d = X_train.map(row => [row[ageIdx], row[glucoseIdx]]);
  const y_2d = y_train;

  // Scale data for visualization
  const minAge = Math.min(...X_2d.map(point => point[0]));
  const maxAge = Math.max(...X_2d.map(point => point[0]));
  const minGlucose = Math.min(...X_2d.map(point => point[1]));
  const maxGlucose = Math.max(...X_2d.map(point => point[1]));

  const scaleX = (value) => 50 + ((value - minAge) / (maxAge - minAge)) * 500;
  const scaleY = (value) => 50 + ((value - minGlucose) / (maxGlucose - minGlucose)) * 300;

  return (
    <div className="real-training-visualization">
      <div className="visualization-header">
        <h3><Icon name="target" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Real {algorithm.toUpperCase()} Training - Iteration {currentIteration + 1}</h3>
        <div className="training-stats">
          <span className="accuracy">Accuracy: {(accuracy * 100).toFixed(1)}%</span>
          <span className="samples">Samples: {X_2d.length}</span>
        </div>
      </div>

      <div className="real-boundary-plot">
        <svg width="600" height="400" viewBox="0 0 600 400" style={{background: 'white', borderRadius: '8px', border: '1px solid #ddd'}}>
          {/* Grid */}
          <defs>
            <pattern id="real-grid" width="30" height="30" patternUnits="userSpaceOnUse">
              <path d="M 30 0 L 0 0 0 30" fill="none" stroke="#f0f0f0" strokeWidth="1"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#real-grid)" />

          {/* Real decision boundary from model */}
          {decision_boundary_data && (
            <path 
              d={decision_boundary_data.boundary_path}
              stroke="#2c3e50" 
              strokeWidth="3" 
              fill="none" 
              strokeDasharray="6,3"
            />
          )}

          {/* Real data points with actual predictions */}
          {X_2d.map((point, index) => {
            const x = scaleX(point[0]);
            const y = scaleY(point[1]);
            const actualClass = y_2d[index];
            const predictedClass = predictions[index];
            const isCorrect = actualClass === predictedClass;
            
            return (
              <circle 
                key={index}
                cx={x} 
                cy={y} 
                r="4" 
                fill={actualClass === 1 ? '#e74c3c' : '#2ecc71'}
                stroke={isCorrect ? '#27ae60' : '#e74c3c'}
                strokeWidth={isCorrect ? 1 : 3}
                opacity="0.8"
              />
            );
          })}

          {/* Axes */}
          <line x1="50" y1="350" x2="550" y2="350" stroke="#333" strokeWidth="2"/>
          <line x1="50" y1="50" x2="50" y2="350" stroke="#333" strokeWidth="2"/>
          
          {/* Labels */}
          <text x="300" y="390" textAnchor="middle" fontSize="22" fill="#333" fontWeight="bold">
            Age (actual values)
          </text>
          <text x="25" y="200" textAnchor="middle" fontSize="22" fill="#333" fontWeight="bold"
                transform="rotate(-90, 25, 200)">
            Glucose Level (actual values)
          </text>
        </svg>
      </div>

      <div className="real-training-info">
        <div className="info-grid">
          <div className="info-item">
            <span className="label">Algorithm:</span>
            <span className="value">{algorithm.toUpperCase()}</span>
          </div>
          <div className="info-item">
            <span className="label">Iteration:</span>
            <span className="value">{currentIteration + 1}</span>
          </div>
          <div className="info-item">
            <span className="label">Real Accuracy:</span>
            <span className="value">{(accuracy * 100).toFixed(1)}%</span>
          </div>
          <div className="info-item">
            <span className="label">Data Points:</span>
            <span className="value">{X_2d.length}</span>
          </div>
        </div>
        
        <div className="legend">
          <div className="legend-item">
            <div className="legend-circle correct"></div>
            <span>Correct Predictions</span>
          </div>
          <div className="legend-item">
            <div className="legend-circle incorrect"></div>
            <span>Incorrect Predictions</span>
          </div>
          <div className="legend-item">
            <div className="legend-circle stroke"></div>
            <span>Stroke Patients</span>
          </div>
          <div className="legend-item">
            <div className="legend-circle no-stroke"></div>
            <span>No Stroke Patients</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RealTrainingVisualization;
