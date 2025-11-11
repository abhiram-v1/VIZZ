import React from 'react';

const ProfessionalDecisionBoundary = ({ algorithm, iteration = 0, size = 'large' }) => {
  const isLarge = size === 'large';
  const width = isLarge ? 600 : 300;
  const height = isLarge ? 400 : 200;
  
  // Generate realistic data points based on iteration - matching PNG style
  const generateDataPoints = () => {
    const points = [];
    const numPoints = isLarge ? 300 : 150;
    
    // Create realistic stroke dataset distribution
    for (let i = 0; i < numPoints; i++) {
      // Generate realistic age and glucose values
      const age = Math.random() * 60 + 20; // 20-80 years
      const glucose = Math.random() * 200 + 50; // 50-250 mg/dL
      
      // Scale to visualization coordinates
      const x = 50 + ((age - 20) / 60) * (width - 100);
      const y = 50 + ((glucose - 50) / 200) * (height - 100);
      
      // Create realistic stroke probability based on age and glucose
      let strokeProbability;
      if (iteration === 0) {
        // Simple rule: older age = higher stroke risk
        strokeProbability = (age - 20) / 60;
      } else if (iteration === 1) {
        // Add glucose factor
        strokeProbability = ((age - 20) / 60) * 0.7 + ((glucose - 50) / 200) * 0.3;
      } else if (iteration === 2) {
        // More complex interaction
        strokeProbability = ((age - 20) / 60) * 0.6 + ((glucose - 50) / 200) * 0.4 + 
                           Math.sin((age - 20) / 10) * 0.1;
      } else if (iteration === 3) {
        // Even more complex
        strokeProbability = ((age - 20) / 60) * 0.5 + ((glucose - 50) / 200) * 0.4 + 
                           Math.sin((age - 20) / 8) * 0.15 + Math.cos((glucose - 50) / 20) * 0.1;
      } else {
        // Most complex - realistic medical model
        strokeProbability = ((age - 20) / 60) * 0.4 + ((glucose - 50) / 200) * 0.3 + 
                           Math.sin((age - 20) / 6) * 0.2 + Math.cos((glucose - 50) / 15) * 0.15 +
                           Math.sin((age - 20) / 4) * Math.cos((glucose - 50) / 25) * 0.1;
      }
      
      // Add some noise and determine final classification
      const noise = (Math.random() - 0.5) * 0.3;
      const isStroke = (strokeProbability + noise) > 0.5;
      
      points.push({ x, y, isStroke, age, glucose });
    }
    
    return points;
  };
  
  // Generate decision boundary path - matching PNG style
  const generateBoundaryPath = () => {
    const points = [];
    const numPoints = isLarge ? 80 : 40;
    
    for (let i = 0; i <= numPoints; i++) {
      const x = 50 + (i * (width - 100) / numPoints);
      let y;
      
      if (iteration === 0) {
        // Simple horizontal line
        y = 50 + (height - 100) * 0.5;
      } else if (iteration === 1) {
        // Slight curve
        const normalizedX = (x - 50) / (width - 100);
        y = 50 + (height - 100) * (0.5 + Math.sin(normalizedX * Math.PI) * 0.1);
      } else if (iteration === 2) {
        // More complex curve
        const normalizedX = (x - 50) / (width - 100);
        y = 50 + (height - 100) * (0.45 + Math.sin(normalizedX * Math.PI * 2) * 0.15 + 
                                   Math.sin(normalizedX * Math.PI * 4) * 0.05);
      } else if (iteration === 3) {
        // Even more complex
        const normalizedX = (x - 50) / (width - 100);
        y = 50 + (height - 100) * (0.4 + Math.sin(normalizedX * Math.PI * 3) * 0.2 + 
                                   Math.sin(normalizedX * Math.PI * 6) * 0.1 + 
                                   Math.sin(normalizedX * Math.PI * 12) * 0.05);
      } else {
        // Most complex - realistic medical boundary
        const normalizedX = (x - 50) / (width - 100);
        y = 50 + (height - 100) * (0.35 + Math.sin(normalizedX * Math.PI * 4) * 0.25 + 
                                   Math.sin(normalizedX * Math.PI * 8) * 0.15 + 
                                   Math.sin(normalizedX * Math.PI * 16) * 0.1 + 
                                   Math.sin(normalizedX * Math.PI * 32) * 0.05);
      }
      
      points.push({ x, y });
    }
    
    return points;
  };
  
  const dataPoints = generateDataPoints();
  const boundaryPoints = generateBoundaryPath();
  
  // Create boundary path string
  const boundaryPath = boundaryPoints.reduce((path, point, index) => {
    return index === 0 ? `M ${point.x},${point.y}` : `${path} L ${point.x},${point.y}`;
  }, '');
  
  // Create filled region path
  const filledRegionPath = `${boundaryPath} L ${width - 50},${height - 50} L 50,${height - 50} Z`;
  
  return (
    <div className="professional-boundary-plot">
      <svg 
        width={width} 
        height={height} 
        viewBox={`0 0 ${width} ${height}`} 
        style={{background: 'white', borderRadius: '8px', border: '1px solid #ddd'}}
      >
        {/* Professional grid - matching PNG style */}
        <defs>
          <pattern id={`grid-${algorithm}-${iteration}`} width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#e0e0e0" strokeWidth="0.5"/>
          </pattern>
          <linearGradient id={`redGradient-${algorithm}-${iteration}`} x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#ffcccc" stopOpacity="0.6"/>
            <stop offset="100%" stopColor="#ff9999" stopOpacity="0.3"/>
          </linearGradient>
          <linearGradient id={`greenGradient-${algorithm}-${iteration}`} x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#ccffcc" stopOpacity="0.6"/>
            <stop offset="100%" stopColor="#99ff99" stopOpacity="0.3"/>
          </linearGradient>
        </defs>
        
        {/* Background grid */}
        <rect width="100%" height="100%" fill={`url(#grid-${algorithm}-${iteration})`} />
        
        {/* Filled regions - matching PNG style */}
        <path 
          d={filledRegionPath}
          fill={`url(#greenGradient-${algorithm}-${iteration})`}
        />
        
        {/* Data points - matching PNG style */}
        {dataPoints.map((point, index) => (
          <circle 
            key={index}
            cx={point.x} 
            cy={point.y} 
            r={isLarge ? 4 : 3} 
            fill={point.isStroke ? '#e74c3c' : '#2ecc71'}
            stroke={point.isStroke ? '#c0392b' : '#27ae60'}
            strokeWidth="1.5"
            opacity="0.9"
          />
        ))}
        
        {/* Decision boundary line - matching PNG style */}
        <path 
          d={boundaryPath}
          stroke="#2c3e50" 
          strokeWidth={isLarge ? 4 : 3} 
          fill="none" 
          strokeDasharray="8,4"
        />
        
        {/* Axes - matching PNG style */}
        <line x1="50" y1={height - 50} x2={width - 50} y2={height - 50} stroke="#333" strokeWidth="2"/>
        <line x1="50" y1="50" x2="50" y2={height - 50} stroke="#333" strokeWidth="2"/>
        
        {/* Axis labels - matching PNG style */}
        <text x={width / 2} y={height - 20} textAnchor="middle" fontSize={isLarge ? 22 : 20} fill="#333" fontWeight="bold">
          Age (scaled)
        </text>
        <text x="25" y={height / 2} textAnchor="middle" fontSize={isLarge ? 22 : 20} fill="#333" fontWeight="bold"
              transform={`rotate(-90, 25, ${height / 2})`}>
          Glucose Level (scaled)
        </text>
        
        {/* Title - matching PNG style */}
        <text x={width / 2} y="30" textAnchor="middle" fontSize={isLarge ? 26 : 22} fill="#333" fontWeight="bold">
          {algorithm.toUpperCase()} Decision Boundary - Iteration {iteration + 1}
        </text>
        
        {/* Legend - matching PNG style */}
        <g transform={`translate(${width - 140}, 60)`}>
          <circle cx="10" cy="10" r="5" fill="#e74c3c" stroke="#c0392b" strokeWidth="1.5"/>
          <text x="20" y="15" fontSize={isLarge ? 19 : 16} fill="#333" fontWeight="bold">Stroke</text>
          <circle cx="10" cy="30" r="5" fill="#2ecc71" stroke="#27ae60" strokeWidth="1.5"/>
          <text x="20" y="35" fontSize={isLarge ? 19 : 16} fill="#333" fontWeight="bold">No Stroke</text>
        </g>
      </svg>
    </div>
  );
};

export default ProfessionalDecisionBoundary;
