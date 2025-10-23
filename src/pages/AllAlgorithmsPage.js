import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import socketService from '../services/socketService';
import apiService from '../services/apiService';

const AllAlgorithmsPage = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState({});
  const [results, setResults] = useState({});
  const [availableAlgorithms, setAvailableAlgorithms] = useState({});
  const [comparisonData, setComparisonData] = useState([]);

  useEffect(() => {
    loadAvailableAlgorithms();
    
    const handleConnect = () => setIsConnected(true);
    const handleDisconnect = () => setIsConnected(false);
    
    const handleTrainingStarted = (data) => {
      setTrainingStatus(prev => ({
        ...prev,
        [data.algorithm]: { status: 'training', progress: 0 }
      }));
    };

    const handleTrainingProgress = (data) => {
      setTrainingStatus(prev => ({
        ...prev,
        [data.algorithm]: {
          status: 'training',
          progress: data.total_iterations ? (data.iteration / data.total_iterations) * 100 : 0
        }
      }));
    };

    const handleTrainingCompleted = (data) => {
      setTrainingStatus(prev => ({
        ...prev,
        [data.algorithm]: { status: 'completed', progress: 100 }
      }));
      
      setResults(prev => ({
        ...prev,
        [data.algorithm]: data.metrics
      }));
    };

    const handleTrainingError = (data) => {
      setTrainingStatus(prev => ({
        ...prev,
        [data.algorithm]: { status: 'error', progress: 0 }
      }));
    };

    socketService.connect();
    socketService.on('connect', handleConnect);
    socketService.on('disconnect', handleDisconnect);
    socketService.on('training_started', handleTrainingStarted);
    socketService.on('training_progress', handleTrainingProgress);
    socketService.on('training_completed', handleTrainingCompleted);
    socketService.on('training_error', handleTrainingError);

    return () => {
      socketService.off('connect', handleConnect);
      socketService.off('disconnect', handleDisconnect);
      socketService.off('training_started', handleTrainingStarted);
      socketService.off('training_progress', handleTrainingProgress);
      socketService.off('training_completed', handleTrainingCompleted);
      socketService.off('training_error', handleTrainingError);
    };
  }, []);

  useEffect(() => {
    // Update comparison data when results change
    const comparison = Object.entries(results).map(([algorithm, metrics]) => ({
      algorithm: availableAlgorithms[algorithm] || algorithm,
      accuracy: metrics.accuracy,
      precision: metrics.precision,
      recall: metrics.recall,
      f1_score: metrics.f1_score
    }));
    setComparisonData(comparison);
  }, [results, availableAlgorithms]);

  const loadAvailableAlgorithms = async () => {
    try {
      const data = await apiService.getAvailableAlgorithms();
      setAvailableAlgorithms(data.algorithms);
      
      // Initialize training status for all algorithms
      const initialStatus = {};
      Object.keys(data.algorithms).forEach(alg => {
        initialStatus[alg] = { status: 'idle', progress: 0 };
      });
      setTrainingStatus(initialStatus);
    } catch (error) {
      console.error('Failed to load available algorithms:', error);
    }
  };

  const handleTrainAll = () => {
    Object.keys(availableAlgorithms).forEach(algorithm => {
      if (trainingStatus[algorithm]?.status !== 'training') {
        socketService.startTraining(algorithm, {
          n_estimators: 50,
          learning_rate: 0.1,
          max_depth: 3
        });
      }
    });
  };

  const handleTrainSingle = (algorithm) => {
    socketService.startTraining(algorithm, {
      n_estimators: 50,
      learning_rate: 0.1,
      max_depth: 3
    });
  };

  const isTraining = Object.values(trainingStatus).some(status => status.status === 'training');
  const allCompleted = Object.keys(availableAlgorithms).length > 0 && 
    Object.values(trainingStatus).every(status => status.status === 'completed' || status.status === 'error');

  return (
    <div className="algorithm-page">
      <div className="page-header">
        <h1 className="page-title">Compare All Algorithms</h1>
        <p className="page-description">
          Train and compare all available boosting algorithms side by side to see their performance differences.
        </p>
      </div>

      <div className="controls-section">
        <h2 className="section-title">Training Controls</h2>
        <div className="controls-row">
          <div className="status-indicator">
            <div className={`status-dot ${isConnected ? 'running' : ''}`}></div>
            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
          <button
            className="button button-primary"
            onClick={handleTrainAll}
            disabled={!isConnected || isTraining}
          >
            {isTraining ? 'Training All...' : 'Train All Algorithms'}
          </button>
        </div>
      </div>

      {/* Algorithm Status Grid */}
      <div className="results-section">
        <div>
          <h2 className="section-title">Algorithm Status</h2>
          <div className="metrics-grid">
            {Object.entries(availableAlgorithms).map(([key, name]) => {
              const status = trainingStatus[key];
              const result = results[key];
              
              return (
                <div key={key} className="metric-card">
                  <h4>{name}</h4>
                  <div style={{ marginBottom: '0.5rem' }}>
                    Status: <span style={{ 
                      color: status?.status === 'training' ? '#28a745' : 
                             status?.status === 'completed' ? '#007bff' :
                             status?.status === 'error' ? '#dc3545' : '#6c757d'
                    }}>
                      {status?.status || 'idle'}
                    </span>
                  </div>
                  
                  {status?.status === 'training' && (
                    <div>
                      <div>Progress: {status.progress?.toFixed(1)}%</div>
                      <div className="progress-bar" style={{ marginTop: '0.5rem', height: '8px' }}>
                        <div
                          className="progress-fill"
                          style={{ width: `${status.progress || 0}%` }}
                        ></div>
                      </div>
                    </div>
                  )}
                  
                  {result && (
                    <div style={{ marginTop: '0.5rem' }}>
                      <div>Accuracy: {result.accuracy?.toFixed(4)}</div>
                      <div>F1 Score: {result.f1_score?.toFixed(4)}</div>
                    </div>
                  )}
                  
                  <button
                    className="button button-secondary"
                    style={{ marginTop: '0.5rem', fontSize: '0.8rem', padding: '0.5rem' }}
                    onClick={() => handleTrainSingle(key)}
                    disabled={status?.status === 'training' || !isConnected}
                  >
                    Train Single
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        {/* Comparison Chart */}
        {comparisonData.length > 0 && (
          <div className="chart-container">
            <h3>Performance Comparison</h3>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={comparisonData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="algorithm" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="accuracy" fill="#667eea" name="Accuracy" />
                <Bar dataKey="f1_score" fill="#764ba2" name="F1 Score" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Detailed Results Table */}
      {comparisonData.length > 0 && (
        <div className="chart-container">
          <h3>Detailed Results</h3>
          <table className="dataset-table">
            <thead>
              <tr>
                <th>Algorithm</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
              </tr>
            </thead>
            <tbody>
              {comparisonData.sort((a, b) => b.f1_score - a.f1_score).map((row, i) => (
                <tr key={i}>
                  <td>{row.algorithm}</td>
                  <td>{row.accuracy?.toFixed(4)}</td>
                  <td>{row.precision?.toFixed(4)}</td>
                  <td>{row.recall?.toFixed(4)}</td>
                  <td>{row.f1_score?.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default AllAlgorithmsPage;
