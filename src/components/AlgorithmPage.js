import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import socketService from '../services/socketService';
import apiService from '../services/apiService';
import RealTrainingVisualization from './RealTrainingVisualization';
import RealDecisionBoundaryCarousel from './RealDecisionBoundaryCarousel';
import { Icon } from './Icons';


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
  
  // Use defaultParams as the recommended/best parameters
  // Define this outside useState so it's accessible in handlers
  const getRecommendedParams = () => ({
    n_estimators: defaultParams.n_estimators || 50,
    learning_rate: defaultParams.learning_rate || 0.1,
    max_depth: defaultParams.max_depth || 3
  });
  
  const recommendedParams = getRecommendedParams();
  const [params, setParams] = useState(recommendedParams);
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
  const [boundaryIterations, setBoundaryIterations] = useState([]);
  const [trainingDataForBoundary, setTrainingDataForBoundary] = useState(null);

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

        // Track last 8 iterations for decision boundary carousel
        // Capture iterations at key milestones to show progression
        if (data.iteration && data.metrics) {
          const totalIterations = data.total_iterations || 100;
          const currentIteration = data.iteration;
          
          // Calculate which iterations to show (8 iterations, evenly spaced)
          // Capture at: ~12.5%, 25%, 37.5%, 50%, 62.5%, 75%, 87.5%, and 100%
          const milestones = [];
          const numMilestones = 8;
          
          if (totalIterations >= numMilestones) {
            for (let i = 1; i <= numMilestones; i++) {
              const milestone = i === numMilestones 
                ? totalIterations 
                : Math.max(1, Math.floor(totalIterations * (i / numMilestones)));
              milestones.push(milestone);
            }
          } else {
            // If fewer than 8 iterations, capture all
            for (let i = 1; i <= totalIterations; i++) {
              milestones.push(i);
            }
          }
          
          // Check if current iteration is one of our milestones (with tolerance to capture nearby iterations)
          const tolerance = Math.max(2, Math.floor(totalIterations / 50)); // Wider tolerance to catch milestones
          const isMilestone = milestones.some(milestone => 
            Math.abs(currentIteration - milestone) <= tolerance
          ) || milestones.includes(currentIteration);
          
          // Also capture the last iteration regardless of milestone
          const isLastIteration = currentIteration >= totalIterations * 0.98;
          
          if (isMilestone || isLastIteration) {
            const iterationData = {
              iteration: currentIteration,
              accuracy: data.metrics.accuracy || 0,
              loss: data.loss || 0,
              metrics: data.metrics,
              totalIterations: totalIterations
            };
            
            setBoundaryIterations(prev => {
              // Remove duplicate iteration if exists
              const filtered = prev.filter(item => item.iteration !== currentIteration);
              const updated = [...filtered, iterationData];
              // Sort by iteration number and keep last 8
              const sorted = updated.sort((a, b) => a.iteration - b.iteration);
              // Ensure we keep exactly 8 (or all if less than 8)
              return sorted.length > 8 ? sorted.slice(-8) : sorted;
            });
          }
        }

        // Load training data for boundary visualization if available
        if (data.training_data) {
          setTrainingDataForBoundary(data.training_data);
        }
      }
    };

    const handleTrainingCompleted = (data) => {
      if (data.algorithm === algorithm) {
        setIsTraining(false);
        setMetrics(data.metrics);
        setFeatureImportances(data.feature_importances);
        setProgressData(null);
        
        // Ensure we have 8 boundary iterations when training completes
        // If we don't have 8, fill in with evenly spaced iterations from the training
        setBoundaryIterations(prev => {
          const totalIterations = data.total_iterations || 100;
          if (prev.length < 8 && totalIterations >= 8) {
            const existingIterations = prev.map(item => item.iteration);
            const milestones = [];
            
            // Calculate 8 evenly spaced milestones
            for (let i = 1; i <= 8; i++) {
              const milestone = i === 8 
                ? totalIterations 
                : Math.max(1, Math.floor(totalIterations * (i / 8)));
              if (!existingIterations.includes(milestone)) {
                milestones.push(milestone);
              }
            }
            
            // Add missing iterations (use final metrics as approximation)
            const missingIterations = milestones.slice(0, 8 - prev.length);
            const finalMetrics = data.metrics || metrics;
            const finalAccuracy = finalMetrics.accuracy || 0;
            
            const newIterations = missingIterations.map((iter) => {
              const progression = iter / totalIterations;
              return {
                iteration: iter,
                accuracy: finalAccuracy * (0.6 + progression * 0.4), // Progressive accuracy
                loss: data.loss_history?.[iter - 1] || 0,
                metrics: finalMetrics,
                totalIterations: totalIterations
              };
            });
            
            const combined = [...prev, ...newIterations];
            return combined.sort((a, b) => a.iteration - b.iteration).slice(-8);
          }
          // Ensure we have exactly 8 iterations
          if (prev.length > 8) {
            return prev.sort((a, b) => a.iteration - b.iteration).slice(-8);
          }
          return prev;
        });
        
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
      const data = await apiService.getDatasetPreview(5);
      setDatasetPreview(data);
      
      // Also prepare training data structure for decision boundary
      // This will be updated with real data during training
      if (data && data.columns) {
        setTrainingDataForBoundary({
          featureNames: data.columns.filter(col => col !== 'stroke' && col !== 'id'),
          X: [],
          y: []
        });
      }
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
    
    // Use current params or fall back to recommended params
    const currentParams = {
      n_estimators: params.n_estimators || recommendedParams.n_estimators,
      learning_rate: params.learning_rate || recommendedParams.learning_rate,
      max_depth: params.max_depth || recommendedParams.max_depth
    };
    
    // Validate and clean parameters before sending
    // No limit on n_estimators - users can train with any number for better results
    const cleanParams = {
      n_estimators: Math.max(10, currentParams.n_estimators), // Removed 200 limit
      learning_rate: Math.max(0.01, Math.min(2.0, currentParams.learning_rate)),
      max_depth: Math.max(1, Math.min(10, currentParams.max_depth))
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
        <div className="algorithm-icon-wrapper">
          {algorithm === 'adaboost' && (
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ marginRight: '12px' }}>
              {/* Scale/Balance representing adaptive weights */}
              <path d="M3 12H21" stroke="#ec4899" strokeWidth="2" strokeLinecap="round"/>
              <path d="M12 3V12" stroke="#ec4899" strokeWidth="2" strokeLinecap="round"/>
              <circle cx="5" cy="16" r="2" fill="#ec4899" opacity="0.3"/>
              <circle cx="12" cy="16" r="2" fill="#ec4899"/>
              <circle cx="19" cy="16" r="2" fill="#ec4899" opacity="0.7"/>
              {/* Arrows showing adaptation */}
              <path d="M8 14L10 12L8 10" stroke="#db2777" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
              <path d="M16 14L14 12L16 10" stroke="#db2777" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
            </svg>
          )}
          {algorithm === 'gradient_boosting' && (
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ marginRight: '12px' }}>
              {/* Gradient/improvement curve */}
              <path d="M3 20L7 16L11 18L15 12L19 14L21 10" stroke="#14b8a6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
              {/* Correction arrows pointing upward */}
              <path d="M7 18L7 14" stroke="#0d9488" strokeWidth="1.5" strokeLinecap="round"/>
              <path d="M11 20L11 16" stroke="#0d9488" strokeWidth="1.5" strokeLinecap="round"/>
              <path d="M15 14L15 10" stroke="#0d9488" strokeWidth="1.5" strokeLinecap="round"/>
              <path d="M19 16L19 12" stroke="#0d9488" strokeWidth="1.5" strokeLinecap="round"/>
              {/* Step indicators */}
              <circle cx="7" cy="14" r="1.5" fill="#14b8a6"/>
              <circle cx="11" cy="16" r="1.5" fill="#14b8a6"/>
              <circle cx="15" cy="10" r="1.5" fill="#14b8a6"/>
              <circle cx="19" cy="12" r="1.5" fill="#14b8a6"/>
            </svg>
          )}
          {algorithm === 'xgboost' && (
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ marginRight: '12px' }}>
              {/* Lightning bolt for speed */}
              <path d="M13 2L8 10H12L11 18L16 10H12L13 2Z" fill="#f59e0b" stroke="#fbbf24" strokeWidth="0.5"/>
              {/* Shield outline for regularization/protection */}
              <path d="M6 8C6 6 8 4 12 4C16 4 18 6 18 8C18 10 16 14 12 18C8 14 6 10 6 8Z" stroke="#fbbf24" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none" opacity="0.6"/>
              {/* Speed lines */}
              <path d="M19 6L21 4" stroke="#fbbf24" strokeWidth="1.5" strokeLinecap="round" opacity="0.7"/>
              <path d="M20 7L21.5 5.5" stroke="#fbbf24" strokeWidth="1.5" strokeLinecap="round" opacity="0.7"/>
            </svg>
          )}
        </div>
        <div>
          <h1 className="page-title">{title}</h1>
          <p className="page-description">{description}</p>
        </div>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {/* Interactive Training Demonstration */}
      {showTrainingDemo && isTraining && (
        <div className="training-demonstration">
          <h2 className="section-title"><Icon name="target" size={24} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Interactive Training Process</h2>
          
          {/* AdaBoost Training Demo */}
          {algorithm === 'adaboost' && (
            <div className="adaboost-demo">
              {trainingStage === 'initial_setup' && (
                <div className="demo-stage">
                  <h3><Icon name="target" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Stage 1: Initial Setup</h3>
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
                  <h3><Icon name="tree" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Stage 2: First Decision Stump</h3>
                  <p>Training a simple decision stump on the weighted data</p>
                  <div className="accuracy-display">
                    Accuracy: {(currentTree.accuracy * 100).toFixed(1)}%
                  </div>
                </div>
              )}

              {trainingStage === 'weight_update' && (
                <div className="demo-stage">
                  <h3><Icon name="scale" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Stage 3: Weight Update</h3>
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
                  <h3><Icon name="tree" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Stage 4: Second Decision Stump</h3>
                  <p>Training a new stump focusing on high-weight samples</p>
                  <div className="accuracy-display">
                    Accuracy: {(currentTree.accuracy * 100).toFixed(1)}%
                  </div>
                </div>
              )}

              {trainingStage === 'final_ensemble' && currentTree && (
                <div className="demo-stage">
                  <h3><Icon name="target" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Stage 5: Final Ensemble</h3>
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
                  <h3><Icon name="chart" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Stage 1: Initial Prediction</h3>
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
                  <h3><Icon name="chart" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Stage 2: Residual Calculation</h3>
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
                  <h3><Icon name="tree" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Stage 3: Tree on Residuals</h3>
                  <p>Training a tree to predict these residuals</p>
                  <div className="accuracy-display">
                    Tree Accuracy: {(decisionBoundary.accuracy * 100).toFixed(1)}%
                  </div>
                </div>
              )}

              {trainingStage === 'final_model' && (
                <div className="demo-stage">
                  <h3><Icon name="target" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Stage 4: Final Model</h3>
                  <p>Combining all trees to create the final prediction</p>
                  <div className="final-result">
                    <div className="result-title">Final Prediction:</div>
                    <div className="result-value">180 + 50 - 20 + 30 = 240</div>
                  </div>
                  <div className="accuracy-display">
                    Final Model Accuracy: {(decisionBoundary.accuracy * 100).toFixed(1)}%
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
                  <h3><Icon name="rocket" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Stage 1: Parallel Processing</h3>
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
                  <h3><Icon name="shield" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Stage 2: Regularization</h3>
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
                  <h3><Icon name="bolt" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Stage 3: Final Optimization</h3>
                  <p>Advanced optimizations for maximum performance</p>
                  <div className="optimization-visualization">
                    <div className="optimization-features">
                      <div className="feature"><Icon name="check" size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Parallel tree construction</div>
                      <div className="feature"><Icon name="check" size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Cache-aware data access</div>
                      <div className="feature"><Icon name="check" size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Out-of-core computation</div>
                      <div className="feature"><Icon name="check" size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Missing value handling</div>
                      <div className="feature"><Icon name="check" size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Categorical feature encoding</div>
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
        <h2 className="section-title"><Icon name="chart" size={24} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> About the Dataset</h2>
        <div className="dataset-explanation">
          <p>
            <strong>Stroke Prediction Dataset:</strong> This dataset contains real patient health records used to predict stroke risk. 
            Each row represents one patient with features like age, blood pressure, BMI, glucose levels, and other health indicators. 
            The target variable (stroke) indicates whether the patient had a stroke (1) or not (0).
          </p>
          <p>
            <strong>Why this dataset?</strong> Stroke prediction is a critical healthcare application where machine learning can help 
            identify at-risk patients early. This allows healthcare providers to take preventive measures before a stroke occurs.
          </p>
        </div>
        {datasetPreview && (
          <div>
            <h3 className="section-subtitle">Dataset Preview (First 5 Rows)</h3>
            <p className="dataset-info">Dataset shape: {datasetPreview.shape[0]} rows × {datasetPreview.shape[1]} columns</p>
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
        </div>

        <div className="params-form">
          <div className="params-header">
            <h3><Icon name="settings" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Training Parameters</h3>
            <p className="params-description">Recommended values are pre-selected. Adjust sliders to customize.</p>
          </div>
          
          <div className="param-group">
            <div className="param-label-row">
              <label className="param-label">Number of Estimators</label>
              <span className="param-value">{params.n_estimators || recommendedParams.n_estimators}</span>
            </div>
            <input
              type="range"
              className="param-slider"
              value={params.n_estimators || recommendedParams.n_estimators}
              onChange={(e) => handleParamChange('n_estimators', parseInt(e.target.value))}
              disabled={isTraining}
              min="10"
              max="1000"
              step="10"
            />
            <div className="param-range">
              <span>10</span>
              <span className="recommended-value">Recommended: {recommendedParams.n_estimators}</span>
              <span>1000</span>
            </div>
          </div>
          
          <div className="param-group">
            <div className="param-label-row">
              <label className="param-label">Learning Rate</label>
              <span className="param-value">{(params.learning_rate || recommendedParams.learning_rate).toFixed(2)}</span>
            </div>
            <input
              type="range"
              className="param-slider"
              value={params.learning_rate || recommendedParams.learning_rate}
              onChange={(e) => handleParamChange('learning_rate', parseFloat(e.target.value))}
              disabled={isTraining}
              min="0.01"
              max="2.0"
              step="0.01"
            />
            <div className="param-range">
              <span>0.01</span>
              <span className="recommended-value">Recommended: {recommendedParams.learning_rate.toFixed(2)}</span>
              <span>2.0</span>
            </div>
          </div>
          
          <div className="param-group">
            <div className="param-label-row">
              <label className="param-label">Max Depth</label>
              <span className="param-value">{params.max_depth || recommendedParams.max_depth}</span>
            </div>
            <input
              type="range"
              className="param-slider"
              value={params.max_depth || recommendedParams.max_depth}
              onChange={(e) => handleParamChange('max_depth', parseInt(e.target.value))}
              disabled={isTraining}
              min="1"
              max="10"
              step="1"
            />
            <div className="param-range">
              <span>1</span>
              <span className="recommended-value">Recommended: {recommendedParams.max_depth}</span>
              <span>10</span>
            </div>
          </div>
          
          <button
            className="button button-secondary reset-params-btn"
            onClick={() => setParams(getRecommendedParams())}
            disabled={isTraining}
          >
            <Icon name="reset" size={16} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Reset to Recommended Values
          </button>
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
            <Icon name="tree" size={24} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Decision Tree Formation - {algorithm === 'adaboost' ? 'AdaBoost Stumps' : 'Gradient Boosting Trees'}
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
              </div>
            )}

            {/* Real Training Visualization - During Training */}
            {isTraining && (
              <div className="real-training-section">
                <div className="boundary-header">
                  <h3><Icon name="target" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Real {algorithm.toUpperCase()} Training Process</h3>
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

            {/* Real Decision Boundary Carousel - After Training Completion */}
            {!isTraining && metrics && (
              <div className="real-decision-boundary-section">
                <RealDecisionBoundaryCarousel
                  algorithm={algorithm}
                  trainingData={trainingDataForBoundary}
                  iterations={boundaryIterations && boundaryIterations.length > 0 ? boundaryIterations : [
                    { iteration: 1, accuracy: metrics.accuracy * 0.65, metrics },
                    { iteration: 2, accuracy: metrics.accuracy * 0.72, metrics },
                    { iteration: 3, accuracy: metrics.accuracy * 0.78, metrics },
                    { iteration: 4, accuracy: metrics.accuracy * 0.83, metrics },
                    { iteration: 5, accuracy: metrics.accuracy * 0.87, metrics },
                    { iteration: 6, accuracy: metrics.accuracy * 0.91, metrics },
                    { iteration: 7, accuracy: metrics.accuracy * 0.95, metrics },
                    { iteration: 8, accuracy: metrics.accuracy, metrics }
                  ]}
                />
              </div>
            )}
          </div>

          {/* Algorithm Explanation */}
          <div className="algorithm-explanation">
            <div className="explanation-card">
              <h4>{algorithm === 'adaboost' ? <><Icon name="ml" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> AdaBoost Process</> : <><Icon name="seedling" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Gradient Boosting Process</>}</h4>
              <div className="explanation-text">
                {algorithm === 'adaboost' ? (
                  <>
                    <p><strong>AdaBoost (Adaptive Boosting)</strong> builds an ensemble of weak learners (decision stumps):</p>
                    <ul>
                      <li><Icon name="target" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Each iteration focuses on misclassified training examples</li>
                      <li><Icon name="scale" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Trees get different weights based on their accuracy</li>
                      <li><Icon name="target" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Final prediction combines all weighted stumps</li>
                      <li><Icon name="chart" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Each stump learns from previous mistakes</li>
                    </ul>
                  </>
                ) : (
                  <>
                    <p><strong>Gradient Boosting</strong> builds trees sequentially to correct errors:</p>
                    <ul>
                      <li><Icon name="rocket" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Each tree learns from the residual errors of previous trees</li>
                      <li><Icon name="chart" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Uses gradient descent to minimize loss function</li>
                      <li><Icon name="tree" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Deeper trees can model complex patterns</li>
                      <li><Icon name="target" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Final prediction sums predictions from all trees</li>
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
