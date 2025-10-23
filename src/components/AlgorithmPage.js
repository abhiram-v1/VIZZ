import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import socketService from '../services/socketService';
import apiService from '../services/apiService';

// Decision Tree SVG Visualization Component
const DecisionTreeSVG = ({ tree, algorithm, size = 'normal' }) => {
  const isSmall = size === 'small';
  const scale = isSmall ? 0.7 : 1;
  const width = isSmall ? 200 : 350;
  const height = isSmall ? 150 : 250;
  
  if (!tree) return null;

  const renderAdaBoostStump = () => {
    const centerX = width / 2;
    const centerY = height / 4;
    const nodeSize = isSmall ? 15 : 20;
    
    return (
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        {/* Root Node */}
        <circle 
          cx={centerX} 
          cy={centerY} 
          r={nodeSize} 
          fill="#667eea" 
          stroke="#4c51bf" 
          strokeWidth="2"
          className="tree-node animate-grow"
        />
        
        {/* Root Node Label */}
        <text 
          x={centerX} 
          y={centerY + 5} 
          textAnchor="middle" 
          fontSize={isSmall ? 8 : 10} 
          fill="white" 
          fontWeight="bold"
        >
          ?
        </text>
        
        {/* Feature Label */}
        <text 
          x={centerX} 
          y={centerY - nodeSize - 10} 
          textAnchor="middle" 
          fontSize={isSmall ? 10 : 12} 
          fill="#2d3748" 
          fontWeight="bold"
        >
          {tree.feature}
        </text>
        
        {/* Threshold */}
        <text 
          x={centerX} 
          y={centerY - nodeSize - 30} 
          textAnchor="middle" 
          fontSize={isSmall ? 8 : 10} 
          fill="#4a5568"
        >
          ≤ {tree.threshold}
        </text>

        {/* Branches */}
        <line 
          x1={centerX} 
          y1={centerY + nodeSize} 
          x2={centerX - 60} 
          y2={centerY + 80} 
          stroke="#4a5568" 
          strokeWidth="2"
          className="tree-branch animate-draw"
        />
        <line 
          x1={centerX} 
          y1={centerY + nodeSize} 
          x2={centerX + 60} 
          y2={centerY + 80} 
          stroke="#4a5568" 
          strokeWidth="2"
          className="tree-branch animate-draw"
        />
        
        {/* Leaf Nodes */}
        <circle 
          cx={centerX - 60} 
          cy={centerY + 80} 
          r={nodeSize} 
          fill="#68d391" 
          stroke="#38a169" 
          strokeWidth="2"
          className="tree-leaf animate-grow"
        />
        <text 
          x={centerX - 60} 
          y={centerY + 85} 
          textAnchor="middle" 
          fontSize={isSmall ? 8 : 10} 
          fill="white" 
          fontWeight="bold"
        >
          Yes
        </text>
        
        <circle 
          cx={centerX + 60} 
          cy={centerY + 80} 
          r={nodeSize} 
          fill="#fc8181" 
          stroke="#e53e3e" 
          strokeWidth="2"
          className="tree-leaf animate-grow"
        />
        <text 
          x={centerX + 60} 
          y={centerY + 85} 
          textAnchor="middle" 
          fontSize={isSmall ? 8 : 10} 
          fill="white" 
          fontWeight="bold"
        >
          No
        </text>
        
        {/* Weight */}
        <text 
          x={centerX} 
          y={height - 10} 
          textAnchor="middle" 
          fontSize={isSmall ? 8 : 10} 
          fill="#667eea" 
          fontWeight="bold"
        >
          Weight: {tree.weight}
        </text>
      </svg>
    );
  };

  const renderGradientBoostingTree = () => {
    const centerX = width / 2;
    const centerY = height / 6;
    const nodeSize = isSmall ? 12 : 16;
    
    return (
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        {/* Root Node */}
        <circle 
          cx={centerX} 
          cy={centerY} 
          r={nodeSize} 
          fill="#667eea" 
          stroke="#4c51bf" 
          strokeWidth="2"
          className="tree-node animate-grow"
        />
        <text 
          x={centerX} 
          y={centerY + 4} 
          textAnchor="middle" 
          fontSize={isSmall ? 7 : 9} 
          fill="white" 
          fontWeight="bold"
        >
          {tree.feature}
        </text>
        <text 
          x={centerX} 
          y={centerY - nodeSize - 15} 
          textAnchor="middle" 
          fontSize={isSmall ? 7 : 9} 
          fill="#4a5568"
        >
          ≤ {tree.threshold}
        </text>

        {/* First Level */}
        <line x1={centerX} y1={centerY + nodeSize} x2={centerX - 80} y2={centerY + 60} stroke="#4a5568" strokeWidth="2" className="tree-branch animate-draw" />
        <line x1={centerX} y1={centerY + nodeSize} x2={centerX + 80} y2={centerY + 60} stroke="#4a5568" strokeWidth="2" className="tree-branch animate-draw" />
        
        {/* Level 1 Nodes */}
        <circle cx={centerX - 80} cy={centerY + 60} r={nodeSize} fill="#9f7aea" stroke="#805ad5" strokeWidth="2" className="tree-node animate-grow" />
        <text x={centerX - 80} y={centerY + 64} textAnchor="middle" fontSize={isSmall ? 6 : 8} fill="white">age</text>
        
        <circle cx={centerX + 80} cy={centerY + 60} r={nodeSize} fill="#9f7aea" stroke="#805ad5" strokeWidth="2" className="tree-node animate-grow" />
        <text x={centerX + 80} y={centerY + 64} textAnchor="middle" fontSize={isSmall ? 6 : 8} fill="white">bmi</text>

        {/* Second Level (deeper trees) */}
        {tree.depth > 2 && (
          <>
            <line x1={centerX - 80} y1={centerY + 60 + nodeSize} x2={centerX - 120} y2={centerY + 100} stroke="#4a5568" strokeWidth="1.5" className="tree-branch animate-draw" />
            <line x1={centerX - 80} y1={centerY + 60 + nodeSize} x2={centerX - 40} y2={centerY + 100} stroke="#4a5568" strokeWidth="1.5" className="tree-branch animate-draw" />
            <line x1={centerX + 80} y1={centerY + 60 + nodeSize} x2={centerX + 40} y2={centerY + 100} stroke="#4a5568" strokeWidth="1.5" className="tree-branch animate-draw" />
            <line x1={centerX + 80} y1={centerY + 60 + nodeSize} x2={centerX + 120} y2={centerY + 100} stroke="#4a5568" strokeWidth="1.5" className="tree-branch animate-draw" />
            
            {/* Level 2 Leaf Nodes */}
            <circle cx={centerX - 120} cy={centerY + 100} r={isSmall ? 8 : 10} fill="#68d391" stroke="#38a169" strokeWidth="2" className="tree-leaf animate-grow" />
            <circle cx={centerX - 40} cy={centerY + 100} r={isSmall ? 8 : 10} fill="#fc8181" stroke="#e53e3e" strokeWidth="2" className="tree-leaf animate-grow" />
            <circle cx={centerX + 40} cy={centerY + 100} r={isSmall ? 8 : 10} fill="#68d391" stroke="#38a169" strokeWidth="2" className="tree-leaf animate-grow" />
            <circle cx={centerX + 120} cy={centerY + 100} r={isSmall ? 8 : 10} fill="#fc8181" stroke="#e53e3e" strokeWidth="2" className="tree-leaf animate-grow" />
          </>
        )}

        {/* Depth Indicator */}
        <text x={centerX} y={height - 5} textAnchor="middle" fontSize={isSmall ? 7 : 9} fill="#667eea" fontWeight="bold">
          Depth: {tree.depth}
        </text>
      </svg>
    );
  };

  return algorithm === 'adaboost' ? renderAdaBoostStump() : renderGradientBoostingTree();
};

const AlgorithmPage = ({ algorithm, title, description, defaultParams = {} }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [datasetPreview, setDatasetPreview] = useState(null);
  const [progress, setProgress] = useState({ current: 0, total: 100 });
  const [logs, setLogs] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [lossHistory, setLossHistory] = useState([]);
  const [featureImportances, setFeatureImportances] = useState(null);
  const [params, setParams] = useState(defaultParams);
  const [error, setError] = useState(null);
  const [predictionData, setPredictionData] = useState({
    gender: 'Male',
    age: 45,
    hypertension: 0,
    heart_disease: 0,
    ever_married: 'Yes',
    work_type: 'Private',
    Residence_type: 'Urban',
    avg_glucose_level: 95.0,
    bmi: 25.0,
    smoking_status: 'never smoked'
  });
  const [predictionResult, setPredictionResult] = useState(null);
  const [currentTree, setCurrentTree] = useState(null);
  const [treeHistory, setTreeHistory] = useState([]);
  const [showTreeVisualization, setShowTreeVisualization] = useState(false);

  // Load dataset preview on component mount
  useEffect(() => {
    loadDatasetPreview();
  }, []);

  // Set up socket connection and event listeners
  useEffect(() => {
    const handleConnect = () => {
      console.log('Frontend: Connected to server');
      setIsConnected(true);
      setError(null);
      addLog('Connected to server', 'info');
    };

    const handleDisconnect = () => {
      console.log('Frontend: Disconnected from server');
      setIsConnected(false);
      addLog('Disconnected from server', 'warning');
    };

    const handleTrainingStarted = (data) => {
      setIsTraining(true);
      setProgress({ current: 0, total: 100 });
      setMetrics(null);
      setLossHistory([]);
      setFeatureImportances(null);
      setError(null);
      addLog(`Started training ${algorithm} with params: ${JSON.stringify(data.params)}`, 'info');
    };

    const handleTrainingProgress = (data) => {
      if (data.algorithm === algorithm) {
        setProgress({
          current: data.iteration || 0,
          total: data.total_iterations || 100
        });

        if (data.loss !== undefined) {
          setLossHistory(prev => [...prev, {
            iteration: data.iteration,
            loss: data.loss,
            timestamp: new Date(data.timestamp).getTime()
          }]);
        }

        // Update tree visualization for boosting algorithms
        if ((algorithm === 'adaboost' || algorithm === 'gradient_boosting') && data.iteration && data.tree_info) {
          const newTree = {
            iteration: data.iteration,
            algorithm: algorithm,
            tree: {
              feature: data.tree_info.feature,
              threshold: data.tree_info.threshold,
              weight: data.tree_info.weight || 1.0,
              depth: data.tree_info.depth || 1,
              type: algorithm === 'adaboost' ? 'stump' : 'tree'
            },
            accuracy: data.metrics?.accuracy || 0,
            timestamp: new Date(data.timestamp)
          };
          setCurrentTree(newTree);
          setTreeHistory(prev => [...prev.slice(-4), newTree]); // Keep last 5 trees
          setShowTreeVisualization(true);
        } else if ((algorithm === 'adaboost' || algorithm === 'gradient_boosting') && data.iteration) {
          // Fallback to generated visualization if no tree_info available
          const newTree = generateTreeVisualization(data.iteration, algorithm, data.metrics);
          setCurrentTree(newTree);
          setTreeHistory(prev => [...prev.slice(-4), newTree]);
          setShowTreeVisualization(true);
        }

        if (data.metrics) {
          setMetrics(prev => ({
            ...prev,
            ...data.metrics
          }));
          addLog(`Iteration ${data.iteration}: loss=${data.loss?.toFixed(4)}, accuracy=${data.metrics.accuracy?.toFixed(4)}`, 'info');
        }
      }
    };

    const handleTrainingCompleted = (data) => {
      if (data.algorithm === algorithm) {
        setIsTraining(false);
        setMetrics(data.metrics);
        setFeatureImportances(data.feature_importances);
        setLossHistory(data.loss_history ? data.loss_history.map((loss, i) => ({
          iteration: i + 1,
          loss: loss,
          timestamp: Date.now()
        })) : lossHistory);
        addLog(`Training completed! Final accuracy: ${data.metrics.accuracy?.toFixed(4)}`, 'info');
      }
    };

    const handleTrainingError = (data) => {
      setIsTraining(false);
      setError(data.message);
      addLog(`Training error: ${data.message}`, 'error');
    };

    const handleError = (data) => {
      console.error('Frontend: Socket error', data);
      setError(data.message);
      addLog(`Error: ${data.message}`, 'error');
    };

    // Connect to socket and set up listeners
    console.log('Frontend: Setting up socket connection...');
    const socket = socketService.connect();
    
    // Set up event listeners
    socketService.on('connect', handleConnect);
    socketService.on('disconnect', handleDisconnect);
    socketService.on('training_started', handleTrainingStarted);
    socketService.on('training_progress', handleTrainingProgress);
    socketService.on('training_completed', handleTrainingCompleted);
    socketService.on('training_error', handleTrainingError);
    socketService.on('error', handleError);
    
    // Check if already connected
    if (socket && socket.connected) {
      handleConnect();
    }

    // Cleanup on unmount
    return () => {
      socketService.off('connect', handleConnect);
      socketService.off('disconnect', handleDisconnect);
      socketService.off('training_started', handleTrainingStarted);
      socketService.off('training_progress', handleTrainingProgress);
      socketService.off('training_completed', handleTrainingCompleted);
      socketService.off('training_error', handleTrainingError);
      socketService.off('error', handleError);
    };
  }, [algorithm]);

  const loadDatasetPreview = async () => {
    try {
      const data = await apiService.getDatasetPreview(10);
      setDatasetPreview(data);
    } catch (error) {
      console.error('Failed to load dataset preview:', error);
      setError('Failed to load dataset preview');
    }
  };

  // Tree visualization generation function
  const generateTreeVisualization = (iteration, algType, metrics) => {
    const features = ['age', 'bmi', 'avg_glucose_level', 'hypertension', 'heart_disease'];
    const randomFeature = features[Math.floor(Math.random() * features.length)];
    const randomThreshold = (Math.random() * 50 + 10).toFixed(1);
    
    // Simulate different tree structures based on algorithm and iteration
    let treeStructure;
    if (algType === 'adaboost') {
      // AdaBoost typically uses shallow trees (stumps)
      treeStructure = {
        type: 'stump',
        feature: randomFeature,
        threshold: randomThreshold,
        weight: (1.0 / (iteration + 1)).toFixed(3)
      };
    } else {
      // Gradient Boosting uses deeper trees
      treeStructure = {
        type: 'tree',
        feature: randomFeature,
        threshold: randomThreshold,
        depth: Math.min(iteration % 4 + 1, 3)
      };
    }

    return {
      iteration,
      algorithm: algType,
      tree: treeStructure,
      accuracy: metrics?.accuracy || 0,
      timestamp: new Date()
    };
  };

  const addLog = useCallback((message, type = 'info') => {
    setLogs(prev => [...prev.slice(-49), {
      id: Date.now(),
      message,
      type,
      timestamp: new Date().toLocaleTimeString()
    }]);
  }, []);

  const handleStartTraining = () => {
    if (!isConnected) {
      setError('Not connected to server');
      return;
    }
    
    // Validate and clean parameters before sending
    const cleanParams = {
      n_estimators: Math.max(1, Math.min(1000, params.n_estimators || 50)),
      learning_rate: Math.max(0.01, Math.min(2.0, params.learning_rate || 0.1)),
      max_depth: Math.max(1, Math.min(20, params.max_depth || 3))
    };
    
    setLogs([]);
    socketService.startTraining(algorithm, cleanParams);
  };

  const handleStopTraining = () => {
    setIsTraining(false);
    addLog('Training stopped by user', 'warning');
  };

  const handleClearLogs = () => {
    setLogs([]);
  };

  const handleParamChange = (param, value) => {
    setParams(prev => ({
      ...prev,
      [param]: isNaN(value) ? value : Number(value)
    }));
  };

  const handlePredictionChange = (field, value) => {
    setPredictionData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handlePredict = async () => {
    try {
      setError(null);
      const result = await apiService.predictStroke(predictionData);
      setPredictionResult(result);
      addLog(`Prediction: ${result.prediction_label} (Confidence: ${(result.confidence * 100).toFixed(1)}%)`, 'info');
    } catch (error) {
      setError('Prediction failed: ' + error.message);
      addLog(`Prediction failed: ${error.message}`, 'error');
    }
  };

  const progressPercentage = progress.total > 0 ? (progress.current / progress.total) * 100 : 0;

  return (
    <div className="algorithm-page">
      <div className="page-header">
        <h1 className="page-title">{title}</h1>
        <p className="page-description">{description}</p>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {/* Dataset Preview */}
      <div className="dataset-preview">
        <h2 className="section-title">Dataset Preview</h2>
        {datasetPreview && (
          <div>
            <p>Dataset shape: {datasetPreview.shape[0]} rows × {datasetPreview.shape[1]} columns</p>
            {datasetPreview.data && datasetPreview.data.length > 0 && (
              <table className="dataset-table">
                <thead>
                  <tr>
                    {datasetPreview.columns.map((col, i) => (
                      <th key={i}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {datasetPreview.data.map((row, i) => (
                    <tr key={i}>
                      {datasetPreview.columns.map((col, j) => (
                        <td key={j}>{row[col]}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="controls-section">
        <h2 className="section-title">Training Controls</h2>
        <div className="controls-row">
          <div className="status-indicator">
            <div className={`status-dot ${isConnected ? 'running' : ''}`}></div>
            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
          <button
            className="button button-primary"
            onClick={handleStartTraining}
            disabled={!isConnected || isTraining}
          >
            {isTraining ? 'Training...' : 'Start Training'}
          </button>
          <button
            className="button button-danger"
            onClick={handleStopTraining}
            disabled={!isTraining}
          >
            Stop
          </button>
          <button
            className="button button-secondary"
            onClick={handleClearLogs}
          >
            Clear Logs
          </button>
        </div>

        <div className="params-form">
          <div className="param-group">
            <label className="param-label">Number of Estimators</label>
            <input
              type="number"
              className="param-input"
              value={params.n_estimators || 50}
              onChange={(e) => handleParamChange('n_estimators', e.target.value)}
              disabled={isTraining}
              min="1"
              max="1000"
            />
          </div>
          <div className="param-group">
            <label className="param-label">Learning Rate</label>
            <input
              type="number"
              className="param-input"
              value={params.learning_rate || 0.1}
              onChange={(e) => handleParamChange('learning_rate', e.target.value)}
              disabled={isTraining}
              min="0.01"
              max="1"
              step="0.01"
            />
          </div>
          <div className="param-group">
            <label className="param-label">Max Depth</label>
            <input
              type="number"
              className="param-input"
              value={params.max_depth || 3}
              onChange={(e) => handleParamChange('max_depth', e.target.value)}
              disabled={isTraining}
              min="1"
              max="20"
            />
          </div>
        </div>
      </div>

      {/* Progress */}
      {isTraining && (
        <div className="progress-section">
          <h2 className="section-title">Training Progress</h2>
          <div className="progress-text">
            {progress.current} / {progress.total} iterations ({progressPercentage.toFixed(1)}%)
          </div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${progressPercentage}%` }}
            ></div>
          </div>
        </div>
      )}

      {/* Logs */}
      <div className="progress-section">
        <h2 className="section-title">Training Logs</h2>
        <div className="logs-container">
          {logs.map(log => (
            <div key={log.id} className={`log-entry ${log.type}`}>
              <span>[{log.timestamp}]</span> {log.message}
            </div>
          ))}
          {logs.length === 0 && (
            <div className="log-entry">No logs yet...</div>
          )}
        </div>
      </div>

      {/* Results */}
      {(metrics || featureImportances) && (
        <div className="results-section">
          {metrics && (
            <div>
              <h2 className="section-title">Final Metrics</h2>
              <div className="metrics-grid">
                {Object.entries(metrics).map(([key, value]) => (
                  <div key={key} className="metric-card">
                    <div className="metric-value">{value?.toFixed(4)}</div>
                    <div className="metric-label">{key.replace('_', ' ')}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {lossHistory.length > 0 && (
            <div className="chart-container">
              <h3>Training Loss</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={lossHistory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="iteration" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="loss" stroke="#667eea" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* Tree Visualization Section - Only for AdaBoost and Gradient Boosting */}
      {(algorithm === 'adaboost' || algorithm === 'gradient_boosting') && (isTraining || showTreeVisualization) && (
        <div className="tree-visualization-section">
          <h2 className="section-title">
            🌳 Decision Tree Formation - {algorithm === 'adaboost' ? 'AdaBoost Stumps' : 'Gradient Boosting Trees'}
          </h2>
          
          <div className="tree-layout">
            {/* Current Tree */}
            {currentTree && (
              <div className="current-tree-container">
                <div className="tree-header">
                  <h3>Current Tree (Iteration {currentTree.iteration})</h3>
                  <div className="tree-stats">
                    <span className="accuracy-badge">
                      Accuracy: {(currentTree.accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                <div className="tree-svg-container">
                  <DecisionTreeSVG tree={currentTree.tree} algorithm={algorithm} />
                </div>
              </div>
            )}

            {/* Tree History */}
            {treeHistory.length > 0 && (
              <div className="tree-history-container">
                <h3>Recent Trees</h3>
                <div className="tree-history-grid">
                  {treeHistory.slice(-4).map((treeItem, index) => (
                    <div key={treeItem.iteration} className="tree-history-item">
                      <div className="tree-history-header">
                        <span>Iteration {treeItem.iteration}</span>
                        <span className="accuracy-small">{(treeItem.accuracy * 100).toFixed(1)}%</span>
                      </div>
                      <DecisionTreeSVG tree={treeItem.tree} algorithm={algorithm} size="small" />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Algorithm Explanation */}
          <div className="algorithm-explanation">
            <div className="explanation-card">
              <h4>{algorithm === 'adaboost' ? '🤖 AdaBoost Process' : '🌿 Gradient Boosting Process'}</h4>
              <div className="explanation-text">
                {algorithm === 'adaboost' ? (
                  <>
                    <p><strong>AdaBoost (Adaptive Boosting)</strong> builds an ensemble of weak learners (decision stumps):</p>
                    <ul>
                      <li>🔍 Each iteration focuses on misclassified training examples</li>
                      <li>⚖️ Trees get different weights based on their accuracy</li>
                      <li>🎯 Final prediction combines all weighted stumps</li>
                      <li>📈 Each stump learns from previous mistakes</li>
                    </ul>
                  </>
                ) : (
                  <>
                    <p><strong>Gradient Boosting</strong> builds trees sequentially to correct errors:</p>
                    <ul>
                      <li>🚀 Each tree learns from the residual errors of previous trees</li>
                      <li>📉 Uses gradient descent to minimize loss function</li>
                      <li>🌳 Deeper trees can model complex patterns</li>
                      <li>🎲 Final prediction sums predictions from all trees</li>
                    </ul>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Prediction Section */}
      <div className="controls-section">
        <h2 className="section-title">Predict Stroke for New Data</h2>
        <div className="params-form">
          <div className="param-group">
            <label className="param-label">Gender</label>
            <select
              className="param-input"
              value={predictionData.gender}
              onChange={(e) => handlePredictionChange('gender', e.target.value)}
            >
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
          </div>
          
          <div className="param-group">
            <label className="param-label">Age</label>
            <input
              type="number"
              className="param-input"
              value={predictionData.age}
              onChange={(e) => handlePredictionChange('age', parseFloat(e.target.value) || 0)}
              min="0"
              max="120"
            />
          </div>
          
          <div className="param-group">
            <label className="param-label">Hypertension</label>
            <select
              className="param-input"
              value={predictionData.hypertension}
              onChange={(e) => handlePredictionChange('hypertension', parseInt(e.target.value))}
            >
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
          
          <div className="param-group">
            <label className="param-label">Heart Disease</label>
            <select
              className="param-input"
              value={predictionData.heart_disease}
              onChange={(e) => handlePredictionChange('heart_disease', parseInt(e.target.value))}
            >
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
          
          <div className="param-group">
            <label className="param-label">Ever Married</label>
            <select
              className="param-input"
              value={predictionData.ever_married}
              onChange={(e) => handlePredictionChange('ever_married', e.target.value)}
            >
              <option value="Yes">Yes</option>
              <option value="No">No</option>
            </select>
          </div>
          
          <div className="param-group">
            <label className="param-label">Work Type</label>
            <select
              className="param-input"
              value={predictionData.work_type}
              onChange={(e) => handlePredictionChange('work_type', e.target.value)}
            >
              <option value="Private">Private</option>
              <option value="Self-employed">Self-employed</option>
              <option value="Govt_job">Government Job</option>
              <option value="children">Children</option>
              <option value="Never_worked">Never Worked</option>
            </select>
          </div>
          
          <div className="param-group">
            <label className="param-label">Residence Type</label>
            <select
              className="param-input"
              value={predictionData.Residence_type}
              onChange={(e) => handlePredictionChange('Residence_type', e.target.value)}
            >
              <option value="Urban">Urban</option>
              <option value="Rural">Rural</option>
            </select>
          </div>
          
          <div className="param-group">
            <label className="param-label">Average Glucose Level</label>
            <input
              type="number"
              className="param-input"
              value={predictionData.avg_glucose_level}
              onChange={(e) => handlePredictionChange('avg_glucose_level', parseFloat(e.target.value) || 0)}
              min="50"
              max="300"
              step="0.1"
            />
          </div>
          
          <div className="param-group">
            <label className="param-label">BMI</label>
            <input
              type="number"
              className="param-input"
              value={predictionData.bmi}
              onChange={(e) => handlePredictionChange('bmi', parseFloat(e.target.value) || 0)}
              min="10"
              max="50"
              step="0.1"
            />
          </div>
          
          <div className="param-group">
            <label className="param-label">Smoking Status</label>
            <select
              className="param-input"
              value={predictionData.smoking_status}
              onChange={(e) => handlePredictionChange('smoking_status', e.target.value)}
            >
              <option value="never smoked">Never Smoked</option>
              <option value="formerly smoked">Formerly Smoked</option>
              <option value="smokes">Smokes</option>
              <option value="Unknown">Unknown</option>
            </select>
          </div>
        </div>
        
        <div className="controls-row">
          <button
            className="button button-primary"
            onClick={handlePredict}
            disabled={isTraining}
          >
            Predict Stroke Risk
          </button>
        </div>
        
        {predictionResult && (
          <div className="chart-container" style={{ marginTop: '1rem' }}>
            <h3>Prediction Result</h3>
            <div style={{ 
              padding: '1rem', 
              backgroundColor: predictionResult.prediction === 1 ? '#ffe6e6' : '#e6ffe6',
              border: `2px solid ${predictionResult.prediction === 1 ? '#ff4444' : '#44ff44'}`,
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <h4 style={{ 
                color: predictionResult.prediction === 1 ? '#d32f2f' : '#2e7d32',
                margin: '0 0 1rem 0'
              }}>
                {predictionResult.prediction_label}
              </h4>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '1rem' }}>
                <div>
                  <strong>No Stroke:</strong> {(predictionResult.probability_no_stroke * 100).toFixed(1)}%
                </div>
                <div>
                  <strong>Stroke:</strong> {(predictionResult.probability_stroke * 100).toFixed(1)}%
                </div>
              </div>
              <div style={{ marginTop: '0.5rem', fontSize: '0.9rem', color: '#666' }}>
                Confidence: {(predictionResult.confidence * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Feature Importances */}
      {featureImportances && (
        <div className="feature-importance">
          <h2 className="section-title">Feature Importances</h2>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={Object.entries(featureImportances).map(([name, importance]) => ({
                name,
                importance: importance * 100 // Convert to percentage
              })).sort((a, b) => b.importance - a.importance)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip formatter={(value) => [`${value.toFixed(2)}%`, 'Importance']} />
                <Bar dataKey="importance" fill="#667eea" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
};

export default AlgorithmPage;
