import React, { useState, useEffect, useCallback } from 'react';
import socketService from '../services/socketService';
import apiService from '../services/apiService';
import { ConnectionStatusIcon, TreeVizIcon } from '../components/Logos';
import { FaPlay, FaSpinner } from 'react-icons/fa';
import { Icon } from '../components/Icons';

// Enhanced Tree Node Component with Pixelmatters-inspired animations
const TreeNode = ({ node, x, y, width, height, level, isLeaf, feature, threshold, prediction, weight, animationDelay = 0 }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), animationDelay);
    return () => clearTimeout(timer);
  }, [animationDelay]);

  const nodeRadius = Math.max(20, 30 - level * 3);
  const nodeColor = isLeaf 
    ? (prediction === 1 ? '#22c55e' : '#ef4444') 
    : (level === 0 ? '#3b82f6' : '#8b5cf6');

  return (
    <g 
      className={`tree-node-group ${isVisible ? 'visible' : ''} ${isHovered ? 'hovered' : ''}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Glow effect for root nodes */}
      {level === 0 && (
        <circle
          cx={x}
          cy={y}
          r={nodeRadius + 8}
          fill="url(#nodeGlow)"
          opacity={isVisible ? 0.3 : 0}
          style={{
            transition: 'opacity 1.2s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
            filter: 'blur(4px)'
          }}
        />
      )}
      
      {/* Main node circle with enhanced styling */}
      <circle
        cx={x}
        cy={y}
        r={nodeRadius}
        fill={`url(#nodeGradient-${level})`}
        stroke="#ffffff"
        strokeWidth={isHovered ? "4" : "3"}
        className="tree-node"
        style={{ 
          filter: isHovered 
            ? 'drop-shadow(0 8px 16px rgba(0,0,0,0.3)) drop-shadow(0 0 20px rgba(102, 126, 234, 0.4))'
            : 'drop-shadow(0 4px 8px rgba(0,0,0,0.2))',
          transform: isVisible 
            ? (isHovered ? 'scale(1.1)' : 'scale(1)') 
            : 'scale(0)',
          transition: 'all 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275)',
          cursor: 'pointer'
        }}
      />
      
      {!isLeaf && (
        <>
          <text
            x={x}
            y={y - 8}
            textAnchor="middle"
            fontSize={Math.max(16, 22 - level)}
            fill="white"
            fontWeight="bold"
            className="feature-text"
            style={{
              opacity: isVisible ? 1 : 0,
              transform: isVisible ? 'translateY(0)' : 'translateY(10px)',
              transition: 'all 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94) 0.2s',
              textShadow: '0 2px 4px rgba(0,0,0,0.3)',
              filter: isHovered ? 'drop-shadow(0 0 8px rgba(255,255,255,0.5))' : 'none'
            }}
          >
            {feature}
          </text>
          <text
            x={x}
            y={y + 8}
            textAnchor="middle"
            fontSize={Math.max(14, 20 - level)}
            fill="white"
            className="threshold-text"
            style={{
              opacity: isVisible ? 0.9 : 0,
              transform: isVisible ? 'translateY(0)' : 'translateY(10px)',
              transition: 'all 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94) 0.4s',
              textShadow: '0 1px 2px rgba(0,0,0,0.3)'
            }}
          >
            ≤ {threshold}
          </text>
        </>
      )}
      
      {isLeaf && (
        <>
          <text
            x={x}
            y={y - 5}
            textAnchor="middle"
            fontSize={Math.max(20, 26 - level)}
            fill="white"
            fontWeight="bold"
            className="prediction-text"
            style={{
              opacity: isVisible ? 1 : 0,
              transform: isVisible ? 'translateY(0) scale(1)' : 'translateY(10px) scale(0.8)',
              transition: 'all 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94) 0.3s',
              textShadow: '0 2px 4px rgba(0,0,0,0.4)',
              filter: isHovered ? 'drop-shadow(0 0 12px rgba(255,255,255,0.6))' : 'none'
            }}
          >
            {prediction === 1 ? 'Stroke' : 'No Stroke'}
          </text>
          {weight && (
            <text
              x={x}
              y={y + 15}
              textAnchor="middle"
              fontSize="19"
              fill="white"
              opacity={isVisible ? 0.8 : 0}
              className="weight-text"
              style={{
                transform: isVisible ? 'translateY(0)' : 'translateY(5px)',
                transition: 'all 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94) 0.5s',
                textShadow: '0 1px 2px rgba(0,0,0,0.3)'
              }}
            >
              w: {weight}
            </text>
          )}
        </>
      )}
    </g>
  );
};

// Enhanced Tree Branch Component with Pixelmatters-inspired animations
const TreeBranch = ({ x1, y1, x2, y2, label, animationDelay = 0, branchType = 'left' }) => {
  const [isDrawn, setIsDrawn] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setIsDrawn(true), animationDelay);
    return () => clearTimeout(timer);
  }, [animationDelay]);

  const branchLength = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
  const strokeWidth = Math.max(2, 4 - (branchLength / 200));

  return (
    <g 
      className="tree-branch-group"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Branch glow effect */}
      <line
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke="url(#branchGlow)"
        strokeWidth={strokeWidth + 4}
        opacity={isDrawn ? (isHovered ? 0.3 : 0.1) : 0}
        style={{
          strokeDasharray: isDrawn ? '0' : '1000',
          strokeDashoffset: isDrawn ? '0' : '1000',
          transition: 'all 1.2s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
          filter: 'blur(2px)'
        }}
      />
      
      {/* Main branch line */}
      <line
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke={isHovered ? "url(#branchGradient)" : "#374151"}
        strokeWidth={isHovered ? strokeWidth + 1 : strokeWidth}
        className="tree-branch"
        style={{
          strokeDasharray: isDrawn ? '0' : '1000',
          strokeDashoffset: isDrawn ? '0' : '1000',
          transition: 'all 1.5s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
          filter: isHovered 
            ? 'drop-shadow(0 0 8px rgba(102, 126, 234, 0.4))'
            : 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))',
          cursor: 'pointer'
        }}
      />
      
      {label && (
        <text
          x={x1 + (x2 - x1) / 2}
          y={y1 + (y2 - y1) / 2 - 5}
          textAnchor="middle"
          fontSize="14"
          fill={isHovered ? "#667eea" : "#6b7280"}
          fontWeight="bold"
          className="branch-label"
          style={{
            opacity: isDrawn ? 1 : 0,
            transform: isDrawn ? 'translateY(0) scale(1)' : 'translateY(5px) scale(0.8)',
            transition: 'all 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94) 1s',
            textShadow: isHovered ? '0 0 8px rgba(102, 126, 234, 0.4)' : '0 1px 2px rgba(0,0,0,0.2)',
            filter: isHovered ? 'drop-shadow(0 0 4px rgba(102, 126, 234, 0.3))' : 'none'
          }}
        >
          {label}
        </text>
      )}
    </g>
  );
};

// R2D3-inspired Data Flow Visualization Component
const DataFlowVisualization = ({ treeData, algorithm, onDataPointSelect }) => {
  const [flowingData, setFlowingData] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);

  // Generate sample data points for visualization
  const sampleDataPoints = [
    { id: 1, age: 45, avg_glucose_level: 95, bmi: 25, hypertension: 0, prediction: 'No Stroke', color: '#22c55e' },
    { id: 2, age: 65, avg_glucose_level: 120, bmi: 30, hypertension: 1, prediction: 'Stroke', color: '#ef4444' },
    { id: 3, age: 35, avg_glucose_level: 85, bmi: 22, hypertension: 0, prediction: 'No Stroke', color: '#22c55e' },
    { id: 4, age: 70, avg_glucose_level: 140, bmi: 28, hypertension: 1, prediction: 'Stroke', color: '#ef4444' }
  ];

  const animateDataFlow = () => {
    setCurrentStep(0);
    const steps = sampleDataPoints.length;
    let step = 0;
    
    const interval = setInterval(() => {
      if (step < steps) {
        setCurrentStep(step);
        setFlowingData(prev => [...prev, sampleDataPoints[step]]);
        step++;
      } else {
        clearInterval(interval);
        setTimeout(() => {
          setFlowingData([]);
          setCurrentStep(0);
        }, 2000);
      }
    }, 1000);
  };

  return (
    <div className="data-flow-container">
      <div className="data-flow-controls">
        <button 
          className="data-flow-btn"
          onClick={animateDataFlow}
        >
          <Icon name="target" size={16} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Show Data Flow Through Tree
        </button>
      </div>
      
      <div className="data-points-display">
        {sampleDataPoints.map((point, index) => (
          <div 
            key={point.id}
            className={`data-point ${index <= currentStep ? 'active' : ''}`}
            style={{
              backgroundColor: point.color,
              transform: index <= currentStep ? 'scale(1.1)' : 'scale(1)',
              transition: 'all 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94)'
            }}
            onClick={() => onDataPointSelect(point)}
          >
            <div className="data-point-info">
              <div className="data-point-label">Age: {point.age}</div>
              <div className="data-point-label">Glucose: {point.avg_glucose_level}</div>
              <div className="data-point-label">BMI: {point.bmi}</div>
              <div className="data-point-prediction">{point.prediction}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Enhanced Decision Tree Visualization with R2D3-inspired educational features
const DecisionTreeVisualization = ({ treeData, algorithm, isAnimating }) => {
  const [animationStep, setAnimationStep] = useState(0);
  const [showDataFlow, setShowDataFlow] = useState(false);
  const [selectedDataPoint, setSelectedDataPoint] = useState(null);
  const [treeExplanation, setTreeExplanation] = useState('');

  useEffect(() => {
    if (isAnimating) {
      const interval = setInterval(() => {
        setAnimationStep(prev => Math.min(prev + 1, 10));
      }, 800);
      return () => clearInterval(interval);
    } else {
      setAnimationStep(10); // Show all immediately if not animating
    }
  }, [isAnimating, treeData]);

  // Generate educational explanation based on algorithm
  useEffect(() => {
    if (algorithm === 'adaboost') {
      setTreeExplanation('AdaBoost creates a series of weak learners (decision stumps) that focus on misclassified examples. Each stump makes a simple decision based on one feature.');
    } else if (algorithm === 'gradient_boosting') {
      setTreeExplanation('Gradient Boosting builds trees sequentially, where each new tree corrects the errors of the previous trees. This creates a strong ensemble from weak learners.');
    }
  }, [algorithm]);

  if (!treeData) {
    return (
      <div className="tree-placeholder">
        <div className="tree-loading">
          <div className="loading-spinner"></div>
          <p>Building decision tree...</p>
        </div>
      </div>
    );
  }

  const renderTree = () => {
    const width = 800;
    const height = 500;
    const nodes = [];
    const branches = [];
    
    // Generate tree structure based on algorithm and depth
    let treeDepth = algorithm === 'adaboost' ? 1 : Math.min(treeData.depth || 3, 4);
    
    // For demonstration, create a more complex tree structure
    if (!treeData.tree || treeData.tree.type === 'stump') {
      // AdaBoost stump with some enhancement
      const rootX = width / 2;
      const rootY = 80;
      const levelHeight = 150;
      
      nodes.push(
        <TreeNode
          key="root"
          x={rootX}
          y={rootY}
          width={100}
          height={60}
          level={0}
          feature={treeData.tree?.feature || 'age'}
          threshold={treeData.tree?.threshold || '65'}
          animationDelay={0}
        />
      );
      
      // Add branches
      branches.push(
        <TreeBranch
          key="left-branch"
          x1={rootX}
          y1={rootY + 30}
          x2={rootX - 120}
          y2={rootY + levelHeight}
          label="Yes"
          animationDelay={500}
          branchType="left"
        />
      );
      
      branches.push(
        <TreeBranch
          key="right-branch"
          x1={rootX}
          y1={rootY + 30}
          x2={rootX + 120}
          y2={rootY + levelHeight}
          label="No"
          animationDelay={500}
          branchType="right"
        />
      );
      
      // Add leaf nodes
      nodes.push(
        <TreeNode
          key="left-leaf"
          x={rootX - 120}
          y={rootY + levelHeight}
          width={80}
          height={40}
          level={1}
          isLeaf={true}
          prediction={0}
          weight={treeData.tree?.weight}
          animationDelay={1000}
        />
      );
      
      nodes.push(
        <TreeNode
          key="right-leaf"
          x={rootX + 120}
          y={rootY + levelHeight}
          width={80}
          height={40}
          level={1}
          isLeaf={true}
          prediction={1}
          weight={treeData.tree?.weight}
          animationDelay={1200}
        />
      );
    } else {
      // Gradient Boosting - deeper tree
      const rootX = width / 2;
      const rootY = 60;
      const levelHeight = 120;
      
      // Root node
      nodes.push(
        <TreeNode
          key="root"
          x={rootX}
          y={rootY}
          width={100}
          height={60}
          level={0}
          feature={treeData.tree?.feature || 'bmi'}
          threshold={treeData.tree?.threshold || '25'}
          animationDelay={0}
        />
      );
      
      // Level 1
      const level1X1 = rootX - 150;
      const level1X2 = rootX + 150;
      const level1Y = rootY + levelHeight;
      
      nodes.push(
        <TreeNode
          key="level1-left"
          x={level1X1}
          y={level1Y}
          width={80}
          height={50}
          level={1}
          feature="avg_glucose_level"
          threshold="140"
          animationDelay={400}
        />
      );
      
      nodes.push(
        <TreeNode
          key="level1-right"
          x={level1X2}
          y={level1Y}
          width={80}
          height={50}
          level={1}
          feature="age"
          threshold="50"
          animationDelay={600}
        />
      );
      
      // Branches to level 1
      branches.push(
        <TreeBranch
          key="root-left"
          x1={rootX}
          y1={rootY + 30}
          x2={level1X1}
          y2={level1Y - 25}
          label="≤ 25"
          animationDelay={300}
        />
      );
      
      branches.push(
        <TreeBranch
          key="root-right"
          x1={rootX}
          y1={rootY + 30}
          x2={level1X2}
          y2={level1Y - 25}
          label="> 25"
          animationDelay={500}
        />
      );
      
      // Level 2 (leaf nodes)
      const leafY = level1Y + levelHeight;
      const leafNodes = [
        { x: level1X1 - 80, y: leafY, pred: 0 },
        { x: level1X1 + 80, y: leafY, pred: 1 },
        { x: level1X2 - 80, y: leafY, pred: 0 },
        { x: level1X2 + 80, y: leafY, pred: 1 }
      ];
      
      leafNodes.forEach((leaf, i) => {
        nodes.push(
          <TreeNode
            key={`leaf-${i}`}
            x={leaf.x}
            y={leaf.y}
            width={60}
            height={40}
            level={2}
            isLeaf={true}
            prediction={leaf.pred}
            animationDelay={800 + i * 200}
          />
        );
      });
      
      // Branches to leaves
      const leafBranches = [
        { x1: level1X1, y1: level1Y + 25, x2: level1X1 - 80, y2: leafY - 20, label: "≤ 140" },
        { x1: level1X1, y1: level1Y + 25, x2: level1X1 + 80, y2: leafY - 20, label: "> 140" },
        { x1: level1X2, y1: level1Y + 25, x2: level1X2 - 80, y2: leafY - 20, label: "≤ 50" },
        { x1: level1X2, y1: level1Y + 25, x2: level1X2 + 80, y2: leafY - 20, label: "> 50" }
      ];
      
      leafBranches.forEach((branch, i) => {
        branches.push(
          <TreeBranch
            key={`leaf-branch-${i}`}
            x1={branch.x1}
            y1={branch.y1}
            x2={branch.x2}
            y2={branch.y2}
            label={branch.label}
            animationDelay={700 + i * 150}
          />
        );
      });
    }
    
    return (
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="tree-svg">
        <defs>
          {/* Enhanced gradient definitions for different node levels */}
          <linearGradient id="nodeGradient-0" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#667eea" />
            <stop offset="50%" stopColor="#764ba2" />
            <stop offset="100%" stopColor="#5a67d8" />
          </linearGradient>
          <linearGradient id="nodeGradient-1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#8b5cf6" />
            <stop offset="100%" stopColor="#7c3aed" />
          </linearGradient>
          <linearGradient id="nodeGradient-2" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#a855f7" />
            <stop offset="100%" stopColor="#9333ea" />
          </linearGradient>
          <linearGradient id="nodeGradient-3" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#c084fc" />
            <stop offset="100%" stopColor="#a855f7" />
          </linearGradient>
          
          {/* Branch gradients */}
          <linearGradient id="branchGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#667eea" />
            <stop offset="50%" stopColor="#764ba2" />
            <stop offset="100%" stopColor="#667eea" />
          </linearGradient>
          
          {/* Glow effects */}
          <radialGradient id="nodeGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#667eea" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#667eea" stopOpacity="0" />
          </radialGradient>
          
          <radialGradient id="branchGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#667eea" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#667eea" stopOpacity="0" />
          </radialGradient>
          
          {/* Enhanced filters */}
          <filter id="nodeGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
            <feMerge> 
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
          
          <filter id="branchGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
            <feMerge> 
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
          
          {/* Drop shadow filter */}
          <filter id="dropShadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="4" stdDeviation="4" floodColor="rgba(0,0,0,0.3)"/>
          </filter>
        </defs>
        
        {/* Enhanced background with gradient */}
        <defs>
          <linearGradient id="backgroundGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#f8fafc" />
            <stop offset="50%" stopColor="#e2e8f0" />
            <stop offset="100%" stopColor="#cbd5e0" />
          </linearGradient>
        </defs>
        <rect width={width} height={height} fill="url(#backgroundGradient)" />
        
        {/* Subtle grid pattern */}
        <defs>
          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#e2e8f0" strokeWidth="0.5" opacity="0.3"/>
          </pattern>
        </defs>
        <rect width={width} height={height} fill="url(#grid)" />
        
        {/* Branches first */}
        {branches}
        
        {/* Nodes on top */}
        {nodes}
        
        {/* Enhanced tree title with gradient text */}
        <defs>
          <linearGradient id="titleGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#667eea" />
            <stop offset="50%" stopColor="#764ba2" />
            <stop offset="100%" stopColor="#667eea" />
          </linearGradient>
        </defs>
        <text 
          x={width / 2} 
          y={30} 
          textAnchor="middle" 
          fontSize="32" 
          fill="url(#titleGradient)" 
          fontWeight="bold"
          style={{
            textShadow: '0 2px 4px rgba(0,0,0,0.1)',
            filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))'
          }}
        >
          {algorithm === 'adaboost' ? 'AdaBoost Stump' : 'Gradient Boosting Tree'}
        </text>

        {/* R2D3-inspired Split Point Visualization */}
        {treeData && treeData.tree && (
          <>
            {/* Decision boundary line */}
            <line
              x1={width / 2 - 100}
              y1={150}
              x2={width / 2 + 100}
              y2={150}
              stroke="url(#branchGradient)"
              strokeWidth="3"
              strokeDasharray="5,5"
              opacity="0.7"
              style={{
                animation: 'dash 2s linear infinite'
              }}
            />
            
            {/* Split point annotation */}
            <text
              x={width / 2}
              y={140}
              textAnchor="middle"
              fontSize="22"
              fill="#667eea"
              fontWeight="bold"
              style={{
                textShadow: '0 1px 2px rgba(0,0,0,0.2)'
              }}
            >
              Split Point: {treeData.tree.feature} ≤ {treeData.tree.threshold}
            </text>
            
            {/* Decision regions */}
            <rect
              x={50}
              y={160}
              width={width/2 - 100}
              height={80}
              fill="rgba(34, 197, 94, 0.1)"
              stroke="rgba(34, 197, 94, 0.3)"
              strokeWidth="2"
              strokeDasharray="3,3"
            />
            <text
              x={width/4}
              y={200}
              textAnchor="middle"
              fontSize="19"
              fill="#22c55e"
              fontWeight="bold"
            >
              No Stroke
            </text>
            
            <rect
              x={width/2 + 50}
              y={160}
              width={width/2 - 100}
              height={80}
              fill="rgba(239, 68, 68, 0.1)"
              stroke="rgba(239, 68, 68, 0.3)"
              strokeWidth="2"
              strokeDasharray="3,3"
            />
            <text
              x={3*width/4}
              y={200}
              textAnchor="middle"
              fontSize="19"
              fill="#ef4444"
              fontWeight="bold"
            >
              Stroke
            </text>
          </>
        )}
      </svg>
    );
  };

  return (
    <div className="tree-visualization-enhanced">
      {/* Educational explanation */}
      <div className="tree-explanation">
        <h3><Icon name="brain" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> How This Tree Works</h3>
        <p>{treeExplanation}</p>
      </div>
      
      {/* Main tree visualization */}
      {renderTree()}
      
      {/* R2D3-inspired data flow visualization */}
      <DataFlowVisualization 
        treeData={treeData}
        algorithm={algorithm}
        onDataPointSelect={setSelectedDataPoint}
      />
      
      {/* Selected data point details */}
      {selectedDataPoint && (
        <div className="data-point-details">
          <h4><Icon name="chart" size={18} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Data Point Analysis</h4>
          <div className="data-point-content">
            <p><strong>Age:</strong> {selectedDataPoint.age}</p>
            <p><strong>Glucose Level:</strong> {selectedDataPoint.avg_glucose_level}</p>
            <p><strong>BMI:</strong> {selectedDataPoint.bmi}</p>
            <p><strong>Prediction:</strong> {selectedDataPoint.prediction}</p>
            <p><strong>Tree Path:</strong> This data point would follow the {selectedDataPoint.age > 65 ? 'right' : 'left'} branch based on age.</p>
          </div>
        </div>
      )}
    </div>
  );
};

// R2D3-inspired Storytelling Tree Visualization Page
const TreeVisualizationPage = () => {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('adaboost');
  const [isConnected, setIsConnected] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [currentTree, setCurrentTree] = useState(null);
  const [treeHistory, setTreeHistory] = useState([]);
  const [algorithmExplanation, setAlgorithmExplanation] = useState('');
  const [ensembleVisualization, setEnsembleVisualization] = useState([]);
  const [currentStoryStep, setCurrentStoryStep] = useState(0);
  const [showInteractiveDemo, setShowInteractiveDemo] = useState(false);
  const [params, setParams] = useState({
    n_estimators: selectedAlgorithm === 'adaboost' ? 50 : 100,
    learning_rate: selectedAlgorithm === 'adaboost' ? 1.0 : 0.1,
    max_depth: selectedAlgorithm === 'adaboost' ? 1 : 3
  });

  // R2D3-style storytelling steps with everyday analogies
  const storySteps = [
    {
      title: "Think of Decision Trees Like a Game of 20 Questions",
      titleIcon: "tree",
      content: "Remember playing '20 Questions'? You ask yes/no questions to guess what someone is thinking. Decision trees work the same way! Instead of guessing an animal, we're asking questions about a patient to predict if they might have a stroke. Each question helps us get closer to the answer.",
      analogy: "Like asking 'Is it bigger than a breadbox?' in 20 Questions",
      visual: "concept"
    },
    {
      title: "The Real-World Problem: Predicting Health Risks",
      titleIcon: "hospital",
      content: "Imagine you're a doctor with 1000 patients. You want to know which ones might be at risk for a stroke. You can't examine everyone individually, but you have their health records (age, blood pressure, weight, etc.). A decision tree helps you make smart predictions based on patterns you've seen before.",
      analogy: "Like a weather forecaster using past data to predict tomorrow's weather",
      visual: "problem"
    },
    {
      title: "Finding the Best Question to Ask First",
      titleIcon: "target",
      content: "Just like in 20 Questions, you want to ask the question that gives you the most information. For stroke prediction, we might ask 'Is the patient over 65?' because age is often the most important factor. This question splits our patients into two groups, making it easier to make predictions.",
      analogy: "Like asking 'Is it alive?' first in 20 Questions - it eliminates half the possibilities",
      visual: "split"
    },
    {
      title: "Building a Complete Decision Tree",
      titleIcon: "seedling",
      content: "After the first question, we ask more questions on each branch. For patients over 65, we might ask about blood pressure. For younger patients, we might ask about weight. Each branch gets more specific questions until we can make a confident prediction.",
      analogy: "Like a flowchart that guides you through a series of yes/no decisions",
      visual: "tree"
    },
    {
      title: "Making Predictions for New Patients",
      titleIcon: "target",
      content: "When a new patient comes in, we follow the tree's path: 'Are they over 65?' → 'Do they have high blood pressure?' → 'Are they overweight?' Based on their answers, we end up at a prediction: 'High Risk' or 'Low Risk' for stroke.",
      analogy: "Like following a GPS route - each turn gets you closer to your destination",
      visual: "prediction"
    }
  ];

  useEffect(() => {
    const handleConnect = () => {
      console.log('Connected to server in Tree Viz');
      setIsConnected(true);
    };
    const handleDisconnect = () => {
      console.log('Disconnected from server in Tree Viz');
      setIsConnected(false);
    };

    // Try to connect and check if already connected
    socketService.connect();
    
    // Check if already connected with multiple attempts
    const checkConnection = () => {
      if (socketService.socket && socketService.socket.connected) {
        setIsConnected(true);
      } else {
        // Try again in 500ms if not connected
        setTimeout(checkConnection, 500);
      }
    };
    
    setTimeout(checkConnection, 500);

    socketService.on('connect', handleConnect);
    socketService.on('disconnect', handleDisconnect);

    return () => {
      socketService.off('connect', handleConnect);
      socketService.off('disconnect', handleDisconnect);
    };
  }, []);

  useEffect(() => {
    const handleTrainingProgress = (data) => {
      if (data.algorithm === selectedAlgorithm && data.tree_info) {
        const newTree = {
          iteration: data.iteration,
          algorithm: selectedAlgorithm,
          tree: {
            feature: data.tree_info.feature,
            threshold: data.tree_info.threshold,
            weight: data.tree_info.weight || 1.0,
            depth: data.tree_info.depth || 1,
            type: selectedAlgorithm === 'adaboost' ? 'stump' : 'tree'
          },
          accuracy: data.metrics?.accuracy || 0,
          timestamp: new Date(data.timestamp)
        };

        setCurrentTree(newTree);
        setTreeHistory(prev => [...prev.slice(-9), newTree]); // Keep last 10 trees
        
        // Add to ensemble visualization
        setEnsembleVisualization(prev => [...prev, {
          ...newTree,
          id: Date.now(),
          position: prev.length
        }]);
      }
    };

    const handleTrainingStarted = (data) => {
      if (data.algorithm === selectedAlgorithm) {
        setIsTraining(true);
        setTreeHistory([]);
        setEnsembleVisualization([]);
      }
    };

    const handleTrainingCompleted = (data) => {
      if (data.algorithm === selectedAlgorithm) {
        setIsTraining(false);
      }
    };

    socketService.on('training_progress', handleTrainingProgress);
    socketService.on('training_started', handleTrainingStarted);
    socketService.on('training_completed', handleTrainingCompleted);
    socketService.on('training_error', (data) => {
      console.error('Training error:', data);
      setIsTraining(false);
    });

    return () => {
      socketService.off('training_progress', handleTrainingProgress);
      socketService.off('training_started', handleTrainingStarted);
      socketService.off('training_completed', handleTrainingCompleted);
      socketService.off('training_error');
    };
  }, [selectedAlgorithm]);

  const handleStartTraining = () => {
    console.log('Attempting to start training...', { isConnected, selectedAlgorithm, params });
    
    // Try to start training
    socketService.startTraining(selectedAlgorithm, params);
    setTreeHistory([]);
    setEnsembleVisualization([]);
    setIsTraining(true); // Set training state immediately for UI feedback
    
    // If not connected, show a demo tree after a delay
    if (!isConnected) {
      setTimeout(() => {
        const demoTree = {
          iteration: 1,
          algorithm: selectedAlgorithm,
          tree: {
            feature: selectedAlgorithm === 'adaboost' ? 'age' : 'bmi',
            threshold: selectedAlgorithm === 'adaboost' ? 65 : 25,
            weight: selectedAlgorithm === 'adaboost' ? 0.8 : 0.1,
            depth: selectedAlgorithm === 'adaboost' ? 1 : 3,
            type: selectedAlgorithm === 'adaboost' ? 'stump' : 'tree'
          },
          accuracy: 0.75,
          timestamp: new Date()
        };
        setCurrentTree(demoTree);
        setIsTraining(false);
      }, 2000);
    }
  };

  const handleParamChange = (param, value) => {
    setParams(prev => ({
      ...prev,
      [param]: isNaN(value) ? value : Number(value)
    }));
  };

  return (
    <div className="tree-visualization-page">
      {/* R2D3-style storytelling progression */}
      <div className="visualization-story">
        {storySteps.map((step, index) => (
          <div key={index} className="story-step">
            <div className="story-content">
              <div className="story-text">
                <h2>{step.titleIcon && <Icon name={step.titleIcon} size={24} style={{ marginRight: '8px', verticalAlign: 'middle' }} />}{step.title}</h2>
                <p>{step.content}</p>
                {step.analogy && (
                  <div className="analogy-box">
                    <div className="analogy-icon"><Icon name="lightbulb" size={24} /></div>
                    <div className="analogy-text">
                      <strong>Think of it like this:</strong> {step.analogy}
                    </div>
                  </div>
                )}
                {index === storySteps.length - 1 && (
                  <button 
                    className="interactive-demo-btn"
                    onClick={() => setShowInteractiveDemo(true)}
                  >
                    <Icon name="rocket" size={16} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Start Interactive Demo
                  </button>
                )}
              </div>
              <div className="story-visual">
                {step.visual === 'concept' && (
                  <div className="concept-visualization">
                    <div className="decision-flow">
                      <div className="decision-node">Age &gt; 65?</div>
                      <div className="decision-branches">
                        <div className="branch yes">Yes → Stroke Risk</div>
                        <div className="branch no">No → Check BMI</div>
                      </div>
                    </div>
                  </div>
                )}
                {step.visual === 'problem' && (
                  <div className="problem-visualization">
                    <div className="data-points">
                      <div className="data-point stroke">Patient A: Age 70, BMI 30 → Stroke</div>
                      <div className="data-point no-stroke">Patient B: Age 45, BMI 22 → No Stroke</div>
                      <div className="data-point stroke">Patient C: Age 68, BMI 28 → Stroke</div>
                    </div>
                  </div>
                )}
                {step.visual === 'split' && (
                  <div className="split-visualization">
                    <div className="decision-boundary"></div>
                    <div className="split-regions">
                      <div className="region stroke-region">Stroke Risk</div>
                      <div className="region no-stroke-region">No Stroke</div>
                    </div>
                  </div>
                )}
                {step.visual === 'tree' && (
                  <div className="tree-preview">
                    <div className="tree-structure">
                      <div className="root-node">Age &gt; 65?</div>
                      <div className="tree-branches">
                        <div className="left-branch">
                          <div className="node">BMI &gt; 25?</div>
                          <div className="leaves">
                            <div className="leaf stroke">Stroke</div>
                            <div className="leaf no-stroke">No Stroke</div>
                          </div>
                        </div>
                        <div className="right-branch">
                          <div className="node">Glucose &gt; 120?</div>
                          <div className="leaves">
                            <div className="leaf stroke">Stroke</div>
                            <div className="leaf no-stroke">No Stroke</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                {step.visual === 'prediction' && (
                  <div className="prediction-flow">
                    <div className="patient-data">New Patient: Age 60, BMI 26</div>
                    <div className="flow-arrow">↓</div>
                    <div className="decision-path">Age &gt; 65? → No</div>
                    <div className="flow-arrow">↓</div>
                    <div className="decision-path">BMI &gt; 25? → Yes</div>
                    <div className="flow-arrow">↓</div>
                    <div className="prediction-result">Prediction: Stroke Risk</div>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Interactive Demo Section */}
      {showInteractiveDemo && (
        <div className="interactive-demo-section">
          <div className="demo-header">
            <h2><Icon name="target" size={24} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Interactive Decision Tree Demo</h2>
            <p>Train your own decision tree and see how it makes predictions</p>
          </div>

          <div className="viz-controls">
            <div className="algorithm-selector">
              <label>Algorithm:</label>
              <select 
                value={selectedAlgorithm} 
                onChange={(e) => setSelectedAlgorithm(e.target.value)}
              >
                <option value="adaboost">AdaBoost (Decision Stumps)</option>
                <option value="gradient_boosting">Gradient Boosting (Deeper Trees)</option>
              </select>
            </div>

            <div className="params-controls">
              <div className="param-group">
                <label>N Estimators:</label>
                <input
                  type="number"
                  value={params.n_estimators}
                  onChange={(e) => handleParamChange('n_estimators', parseInt(e.target.value))}
                  min="10"
                  max="200"
                />
              </div>
              <div className="param-group">
                <label>Learning Rate:</label>
                <input
                  type="number"
                  step="0.1"
                  value={params.learning_rate}
                  onChange={(e) => handleParamChange('learning_rate', parseFloat(e.target.value))}
                  min="0.01"
                  max="2.0"
                />
              </div>
              <div className="param-group">
                <label>Max Depth:</label>
                <input
                  type="number"
                  value={params.max_depth}
                  onChange={(e) => handleParamChange('max_depth', parseInt(e.target.value))}
                  min="1"
                  max="6"
                />
              </div>
            </div>

            <div className="button-section">
              <div className="connection-status">
                <ConnectionStatusIcon connected={isConnected} size={20} />
                <span style={{ marginLeft: '0.5rem', fontWeight: 'bold' }}>
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              <button 
                className="start-training-btn"
                onClick={handleStartTraining}
                disabled={isTraining}
                style={{ 
                  opacity: isTraining ? 0.6 : 1,
                  cursor: isTraining ? 'not-allowed' : 'pointer'
                }}
              >
                {isTraining ? (
                  <>
                    <FaSpinner 
                      size={16} 
                      style={{ 
                        animation: 'spin 1s linear infinite',
                        marginRight: '0.5rem'
                      }} 
                    />
                    <span>Training...</span>
                  </>
                ) : (
                  <>
                    <FaPlay size={16} />
                    <span style={{ marginLeft: '0.5rem' }}>Start Live Training</span>
                  </>
                )}
              </button>
            </div>
          </div>

          <div className="viz-content">
            <div className="main-tree-display">
              <div className="current-tree-section">
                <h2>
                  {isTraining ? 
                    `Building Tree (Iteration ${currentTree?.iteration || 0})` : 
                    'Current Tree'
                  }
                </h2>
                <DecisionTreeVisualization 
                  treeData={currentTree} 
                  algorithm={selectedAlgorithm}
                  isAnimating={isTraining}
                />
              </div>

              <div className="ensemble-visualization">
                <h3>Ensemble Formation ({ensembleVisualization.length} trees)</h3>
                <div className="ensemble-grid">
                  {ensembleVisualization.slice(-12).map((tree, index) => (
                    <div key={tree.id} className="ensemble-tree-item">
                      <div className="tree-number">#{index + 1}</div>
                      <DecisionTreeVisualization 
                        treeData={tree} 
                        algorithm={selectedAlgorithm}
                        isAnimating={false}
                      />
                      <div className="tree-weight">Weight: {tree.tree.weight}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="algorithm-info">
              <div className="info-card">
                <h3><Icon name="brain" size={20} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> How {selectedAlgorithm === 'adaboost' ? 'AdaBoost' : 'Gradient Boosting'} Works (In Simple Terms)</h3>
                <div className="explanation">
                  {selectedAlgorithm === 'adaboost' ? (
                    <>
                      <div className="analogy-section">
                        <h4><Icon name="target" size={18} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Think of AdaBoost Like a Team of Medical Students</h4>
                        <p>Imagine you're training a team of medical students to diagnose stroke risk. Each student is only allowed to ask ONE yes/no question (like "Is the patient over 65?"). Here's how AdaBoost works:</p>
                      </div>
                      <div className="step-by-step">
                        <div className="step">
                          <div className="step-number">1</div>
                          <div className="step-content">
                            <strong>First Student:</strong> Asks their best question and makes predictions. Some patients are diagnosed correctly, others are wrong.
                          </div>
                        </div>
                        <div className="step">
                          <div className="step-number">2</div>
                          <div className="step-content">
                            <strong>Second Student:</strong> Focuses on the patients the first student got wrong. They ask a different question to catch these mistakes.
                          </div>
                        </div>
                        <div className="step">
                          <div className="step-number">3</div>
                          <div className="step-content">
                            <strong>More Students:</strong> Each new student focuses on the patients that previous students missed. They each get a "confidence score" based on how well they do.
                          </div>
                        </div>
                        <div className="step">
                          <div className="step-number">4</div>
                          <div className="step-content">
                            <strong>Final Decision:</strong> When a new patient comes in, all students vote. Students who are more confident get more say in the final decision.
                          </div>
                        </div>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="analogy-section">
                        <h4>🌊 Think of Gradient Boosting Like Learning to Ride a Bike</h4>
                        <p>Remember learning to ride a bike? You fell down, got back up, and learned from each mistake. Gradient Boosting works the same way with decision trees:</p>
                      </div>
                      <div className="step-by-step">
                        <div className="step">
                          <div className="step-number">1</div>
                          <div className="step-content">
                            <strong>First Tree:</strong> Makes predictions on all patients. Some predictions are right, others are wrong (like your first bike ride).
                          </div>
                        </div>
                        <div className="step">
                          <div className="step-number">2</div>
                          <div className="step-content">
                            <strong>Second Tree:</strong> Looks at the mistakes the first tree made and tries to fix them. It's like adjusting your balance after falling.
                          </div>
                        </div>
                        <div className="step">
                          <div className="step-number">3</div>
                          <div className="step-content">
                            <strong>More Trees:</strong> Each new tree focuses on the remaining mistakes. It's like getting better at bike riding with each attempt.
                          </div>
                        </div>
                        <div className="step">
                          <div className="step-number">4</div>
                          <div className="step-content">
                            <strong>Final Prediction:</strong> All trees work together. The first tree's prediction + second tree's correction + third tree's fine-tuning = final answer.
                          </div>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TreeVisualizationPage;
