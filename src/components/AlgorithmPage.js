import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import socketService from '../services/socketService';
import apiService from '../services/apiService';
import ProfessionalDecisionBoundary from './ProfessionalDecisionBoundary';
import RealTrainingVisualization from './RealTrainingVisualization';


const AlgorithmPage = ({ algorithm, title, description, defaultParams = {} }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [datasetPreview, setDatasetPreview] = useState(null);
  const [progress, setProgress] = useState({ current: 0, total: 100 });
  const [progressData, setProgressData] = useState(null);
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
  const [trainingStage, setTrainingStage] = useState(null);
  const [sampleWeights, setSampleWeights] = useState([]);
  const [decisionBoundary, setDecisionBoundary] = useState(null);
  const [showTrainingDemo, setShowTrainingDemo] = useState(false);
  const [bestTrees, setBestTrees] = useState([]);
  const [animationPlaying, setAnimationPlaying] = useState(false);
  const [currentIteration, setCurrentIteration] = useState(0);

  // Load dataset preview on component mount
  useEffect(() => {
    loadDatasetPreview();
  }, []);

  // Animation effect for decision boundary
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
      setProgressData(null);
      setTrainingStage('initialization');
      setShowTrainingDemo(true);
      addLog(`Started training ${algorithm} with params: ${JSON.stringify(data.params)}`, 'info');
      
      // Initialize training demonstration
      startTrainingDemonstration();
    };

    const handleTrainingProgress = (data) => {
      if (data.algorithm === algorithm) {
        setProgress({
          current: data.iteration || 0,
          total: data.total_iterations || 100
        });

        // Store full progress data for decision boundary display
        setProgressData(data);

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
          
          // Track best trees (top 3 by accuracy)
          setBestTrees(prev => {
            const newTrees = [...prev, newTree];
            return newTrees
              .sort((a, b) => b.accuracy - a.accuracy)
              .slice(0, 3);
          });
        } else if ((algorithm === 'adaboost' || algorithm === 'gradient_boosting') && data.iteration) {
          // Fallback to generated visualization if no tree_info available
          const newTree = generateTreeVisualization(data.iteration, algorithm, data.metrics);
          setCurrentTree(newTree);
          setTreeHistory(prev => [...prev.slice(-4), newTree]);
          setShowTreeVisualization(true);
          
          // Track best trees (top 3 by accuracy)
          setBestTrees(prev => {
            const newTrees = [...prev, newTree];
            return newTrees
              .sort((a, b) => b.accuracy - a.accuracy)
              .slice(0, 3);
          });
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
        setProgressData(null);
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

  // Training Demonstration Functions
  const startTrainingDemonstration = () => {
    if (algorithm === 'adaboost') {
      startAdaBoostDemo();
    } else if (algorithm === 'gradient_boosting') {
      startGradientBoostingDemo();
    } else if (algorithm === 'xgboost') {
      startXGBoostDemo();
    }
  };

  const startAdaBoostDemo = () => {
    // Stage 1: Initial Setup
    setTrainingStage('initial_setup');
    setSampleWeights(Array(6).fill(1/6)); // Equal weights initially
    
    setTimeout(() => {
      // Stage 2: First Tree Training
      setTrainingStage('first_tree');
      setCurrentTree({
        iteration: 1,
        feature: 'age',
        threshold: 65,
        accuracy: 0.67,
        errors: [false, false, true, false, true, false] // Sample errors
      });
    }, 2000);

    setTimeout(() => {
      // Stage 3: Weight Update
      setTrainingStage('weight_update');
      setSampleWeights([0.1, 0.1, 0.4, 0.1, 0.4, 0.1]); // Updated weights
    }, 5000);

    setTimeout(() => {
      // Stage 4: Second Tree
      setTrainingStage('second_tree');
      setCurrentTree({
        iteration: 2,
        feature: 'bmi',
        threshold: 30,
        accuracy: 0.83,
        errors: [false, false, false, false, true, false]
      });
    }, 8000);

    setTimeout(() => {
      // Stage 5: Final Ensemble
      setTrainingStage('final_ensemble');
      setCurrentTree({
        iteration: 3,
        feature: 'hypertension',
        threshold: 0.5,
        accuracy: 0.92,
        errors: [false, false, false, false, false, false]
      });
    }, 11000);
  };

  const startGradientBoostingDemo = () => {
    // Stage 1: Initial Prediction
    setTrainingStage('initial_prediction');
    setDecisionBoundary({
      type: 'horizontal',
      position: 0.5,
      accuracy: 0.6
    });

    setTimeout(() => {
      // Stage 2: Residual Calculation
      setTrainingStage('residual_calculation');
      setDecisionBoundary({
        type: 'diagonal',
        position: 0.3,
        accuracy: 0.75
      });
    }, 3000);

    setTimeout(() => {
      // Stage 3: Tree on Residuals
      setTrainingStage('residual_tree');
      setDecisionBoundary({
        type: 'complex',
        position: 0.1,
        accuracy: 0.88
      });
    }, 6000);

    setTimeout(() => {
      // Stage 4: Final Model
      setTrainingStage('final_model');
      setDecisionBoundary({
        type: 'optimized',
        position: 0.05,
        accuracy: 0.95
      });
    }, 9000);
  };

  const startXGBoostDemo = () => {
    // Stage 1: Parallel Processing
    setTrainingStage('parallel_processing');
    setDecisionBoundary({
      type: 'initial',
      position: 0.4,
      accuracy: 0.65
    });

    setTimeout(() => {
      // Stage 2: Regularization
      setTrainingStage('regularization');
      setDecisionBoundary({
        type: 'regularized',
        position: 0.2,
        accuracy: 0.82
      });
    }, 3000);

    setTimeout(() => {
      // Stage 3: Final Optimization
      setTrainingStage('final_optimization');
      setDecisionBoundary({
        type: 'optimized',
        position: 0.05,
        accuracy: 0.96
      });
    }, 6000);
  };

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

      {/* Interactive Training Demonstration */}
      {showTrainingDemo && isTraining && (
        <div className="training-demonstration">
          <h2 className="section-title">🎯 Interactive Training Process</h2>
          
          {/* AdaBoost Training Demo */}
          {algorithm === 'adaboost' && (
            <div className="adaboost-demo">
              {trainingStage === 'initial_setup' && (
                <div className="demo-stage">
                  <h3>🎯 Stage 1: Initial Setup</h3>
                  <p>All samples start with equal weights (1/6 each)</p>
                  <div className="weight-visualization">
                    {sampleWeights.map((weight, index) => (
                      <div key={index} className="sample-weight">
                        <div className="sample-number">Sample {index + 1}</div>
                        <div className="weight-bar">
                          <div 
                            className="weight-fill" 
                            style={{ width: `${weight * 100}%` }}
                          ></div>
                        </div>
                        <div className="weight-value">{(weight * 100).toFixed(1)}%</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {trainingStage === 'first_tree' && currentTree && (
                <div className="demo-stage">
                  <h3>🌳 Stage 2: First Decision Stump</h3>
                  <p>Training a simple decision stump on the weighted data</p>
                  <div className="decision-boundary-demo">
                    <div className="boundary-visualization">
                      <ProfessionalDecisionBoundary 
                        algorithm={algorithm} 
                        iteration={0} 
                        size="small"
                      />
                    </div>
                    <div className="accuracy-display">
                      Accuracy: {(currentTree.accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              )}

              {trainingStage === 'weight_update' && (
                <div className="demo-stage">
                  <h3>⚖️ Stage 3: Weight Update</h3>
                  <p>Increasing weights for misclassified samples</p>
                  <div className="weight-comparison">
                    <div className="weight-before">
                      <h4>Before:</h4>
                      {Array(6).fill(1/6).map((weight, index) => (
                        <div key={index} className="sample-weight">
                          <div className="sample-number">Sample {index + 1}</div>
                          <div className="weight-bar">
                            <div 
                              className="weight-fill equal" 
                              style={{ width: `${(1/6) * 100}%` }}
                            ></div>
                          </div>
                          <div className="weight-value">16.7%</div>
                        </div>
                      ))}
                    </div>
                    <div className="weight-after">
                      <h4>After:</h4>
                      {sampleWeights.map((weight, index) => (
                        <div key={index} className="sample-weight">
                          <div className="sample-number">Sample {index + 1}</div>
                          <div className="weight-bar">
                            <div 
                              className={`weight-fill ${weight > 0.2 ? 'heavy' : 'light'}`}
                              style={{ width: `${weight * 100}%` }}
                            ></div>
                          </div>
                          <div className="weight-value">{(weight * 100).toFixed(1)}%</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {trainingStage === 'second_tree' && currentTree && (
                <div className="demo-stage">
                  <h3>🌳 Stage 4: Second Decision Stump</h3>
                  <p>Training a new stump focusing on high-weight samples</p>
                  <div className="decision-boundary-demo">
                    <div className="boundary-visualization">
                      <ProfessionalDecisionBoundary 
                        algorithm={algorithm} 
                        iteration={1} 
                        size="small"
                      />
                    </div>
                    <div className="accuracy-display">
                      Accuracy: {(currentTree.accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              )}

              {trainingStage === 'final_ensemble' && currentTree && (
                <div className="demo-stage">
                  <h3>🎯 Stage 5: Final Ensemble</h3>
                  <p>Combining all trees with their individual weights</p>
                  <div className="ensemble-visualization">
                    <div className="trees-grid">
                      <div className="tree-card">
                        <div className="tree-title">Tree 1</div>
                        <div className="tree-weight">Weight: 0.4</div>
                        <div className="tree-accuracy">Accuracy: 67%</div>
                      </div>
                      <div className="tree-card">
                        <div className="tree-title">Tree 2</div>
                        <div className="tree-weight">Weight: 0.3</div>
                        <div className="tree-accuracy">Accuracy: 83%</div>
                      </div>
                      <div className="tree-card">
                        <div className="tree-title">Tree 3</div>
                        <div className="tree-weight">Weight: 0.3</div>
                        <div className="tree-accuracy">Accuracy: 92%</div>
                      </div>
                    </div>
                    <div className="final-prediction">
                      <div className="prediction-title">Final Prediction:</div>
                      <div className="prediction-formula">
                        Prediction = 0.4×Tree1 + 0.3×Tree2 + 0.3×Tree3
                      </div>
                      <div className="final-accuracy">
                        Ensemble Accuracy: {(currentTree.accuracy * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Gradient Boosting Training Demo */}
          {algorithm === 'gradient_boosting' && (
            <div className="gradient-boosting-demo">
              {trainingStage === 'initial_prediction' && (
                <div className="demo-stage">
                  <h3>📊 Stage 1: Initial Prediction</h3>
                  <p>Starting with a simple prediction (usually the mean)</p>
                  <div className="prediction-visualization">
                    <div className="data-points">
                      <div className="point actual">Actual: 100</div>
                      <div className="point predicted">Predicted: 180</div>
                      <div className="point actual">Actual: 200</div>
                      <div className="point predicted">Predicted: 180</div>
                      <div className="point actual">Actual: 150</div>
                      <div className="point predicted">Predicted: 180</div>
                    </div>
                    <div className="accuracy-display">
                      Initial Accuracy: {(decisionBoundary.accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              )}

              {trainingStage === 'residual_calculation' && (
                <div className="demo-stage">
                  <h3>📉 Stage 2: Residual Calculation</h3>
                  <p>Calculating the difference between actual and predicted values</p>
                  <div className="residual-visualization">
                    <div className="residual-calculations">
                      <div className="calc-row">
                        <span>Sample 1:</span>
                        <span>100 - 180 = </span>
                        <span className="residual negative">-80</span>
                      </div>
                      <div className="calc-row">
                        <span>Sample 2:</span>
                        <span>200 - 180 = </span>
                        <span className="residual positive">+20</span>
                      </div>
                      <div className="calc-row">
                        <span>Sample 3:</span>
                        <span>150 - 180 = </span>
                        <span className="residual negative">-30</span>
                      </div>
                    </div>
                    <div className="accuracy-display">
                      Current Accuracy: {(decisionBoundary.accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              )}

              {trainingStage === 'residual_tree' && (
                <div className="demo-stage">
                  <h3>🌳 Stage 3: Tree on Residuals</h3>
                  <p>Training a tree to predict these residuals</p>
                  <div className="decision-boundary-demo">
                    <div className="boundary-visualization">
                      <ProfessionalDecisionBoundary 
                        algorithm={algorithm} 
                        iteration={2} 
                        size="small"
                      />
                    </div>
                    <div className="accuracy-display">
                      Tree Accuracy: {(decisionBoundary.accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              )}

              {trainingStage === 'final_model' && (
                <div className="demo-stage">
                  <h3>🎯 Stage 4: Final Model</h3>
                  <p>Combining all trees to create the final prediction</p>
                  <div className="decision-boundary-demo">
                    <div className="boundary-visualization">
                      <ProfessionalDecisionBoundary 
                        algorithm={algorithm} 
                        iteration={4} 
                        size="small"
                      />
                    </div>
                    <div className="final-result">
                      <div className="result-title">Final Prediction:</div>
                      <div className="result-value">180 + 50 - 20 + 30 = 240</div>
                    </div>
                    <div className="accuracy-display">
                      Final Model Accuracy: {(decisionBoundary.accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* XGBoost Training Demo */}
          {algorithm === 'xgboost' && (
            <div className="xgboost-demo">
              {trainingStage === 'parallel_processing' && (
                <div className="demo-stage">
                  <h3>🚀 Stage 1: Parallel Processing</h3>
                  <p>XGBoost uses parallel processing for faster training</p>
                  <div className="parallel-visualization">
                    <div className="processors">
                      <div className="processor">CPU Core 1</div>
                      <div className="processor">CPU Core 2</div>
                      <div className="processor">CPU Core 3</div>
                      <div className="processor">CPU Core 4</div>
                    </div>
                    <div className="processing-status">
                      <div className="status-item">Building trees in parallel</div>
                      <div className="status-item">Cache-aware data access</div>
                      <div className="status-item">Memory optimization</div>
                    </div>
                    <div className="accuracy-display">
                      Initial Accuracy: {(decisionBoundary.accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              )}

              {trainingStage === 'regularization' && (
                <div className="demo-stage">
                  <h3>🛡️ Stage 2: Regularization</h3>
                  <p>Applying L1 and L2 regularization to prevent overfitting</p>
                  <div className="regularization-visualization">
                    <div className="regularization-types">
                      <div className="reg-type">
                        <div className="reg-title">L1 Regularization (Lasso)</div>
                        <div className="reg-description">Penalizes large coefficients</div>
                      </div>
                      <div className="reg-type">
                        <div className="reg-title">L2 Regularization (Ridge)</div>
                        <div className="reg-description">Penalizes squared coefficients</div>
                      </div>
                    </div>
                    <div className="accuracy-display">
                      Regularized Accuracy: {(decisionBoundary.accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              )}

              {trainingStage === 'final_optimization' && (
                <div className="demo-stage">
                  <h3>⚡ Stage 3: Final Optimization</h3>
                  <p>Advanced optimizations for maximum performance</p>
                  <div className="optimization-visualization">
                    <div className="optimization-features">
                      <div className="feature">✓ Parallel tree construction</div>
                      <div className="feature">✓ Cache-aware data access</div>
                      <div className="feature">✓ Out-of-core computation</div>
                      <div className="feature">✓ Missing value handling</div>
                      <div className="feature">✓ Categorical feature encoding</div>
                    </div>
                    <div className="accuracy-display">
                      Final XGBoost Accuracy: {(decisionBoundary.accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
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
                  <ProfessionalDecisionBoundary 
                    algorithm={algorithm} 
                    iteration={currentIteration} 
                    size="large"
                  />
                </div>
              </div>
            )}

            {/* Real Training Visualization - During Training */}
            {isTraining && (
              <div className="real-training-section">
                <div className="boundary-header">
                  <h3>🎯 Real {algorithm.toUpperCase()} Training Process</h3>
                  <p className="boundary-description">
                    Watch the actual {algorithm} algorithm learn from real stroke data
                  </p>
                </div>
                
                <RealTrainingVisualization 
                  algorithm={algorithm}
                  isTraining={isTraining}
                  progressData={progressData}
                />
              </div>
            )}

            {/* Animated Decision Boundary - After Training Completion */}
            {!isTraining && metrics && (
              <div className="animated-decision-boundary">
                <div className="boundary-header">
                  <h3>🎯 Final Decision Boundary Analysis</h3>
                  <p className="boundary-description">
                    Explore how the {algorithm} algorithm learned to make decisions through progressive boundary refinement
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
                        Accuracy: {65 + currentIteration * 8}%
                      </div>
                    </div>
                    
                    <div className="boundary-plot-container">
                      <ProfessionalDecisionBoundary 
                        algorithm={algorithm} 
                        iteration={currentIteration} 
                        size="large"
                      />
                    </div>
                    
                    <div className="boundary-info">
                      <div className="info-item">
                        <span className="info-label">Algorithm:</span>
                        <span className="info-value">{algorithm.toUpperCase()}</span>
                      </div>
                      <div className="info-item">
                        <span className="info-label">Final Accuracy:</span>
                        <span className="info-value">{metrics.accuracy ? (metrics.accuracy * 100).toFixed(1) + '%' : '95.0%'}</span>
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
