import React from 'react';
import { 
  // Icons we actually use
  FaBrain,
  FaLayerGroup, FaBolt,
  FaCheckCircle, FaTimesCircle, FaCubes
} from 'react-icons/fa';
import { 
  HiLightningBolt
} from 'react-icons/hi';
import { 
  MdTrendingUp
} from 'react-icons/md';

// Main Logo Component using professional brain icon
export const MainLogo = ({ size = 40, className = "" }) => (
  <FaBrain 
    size={size} 
    className={className} 
    style={{ color: '#667eea' }}
  />
);

// AdaBoost Icon - using tree icon with bolt for boosting
export const AdaBoostIcon = ({ size = 24, className = "" }) => {
  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <img src={"/decision-tree.svg"} alt={"Decision Tree"} style={{ width: size, height: size, opacity: 0.9 }} />
      <FaBolt 
        size={size * 0.4} 
        style={{ 
          position: 'absolute', 
          top: -2, 
          right: -2, 
          color: '#fbbf24' 
        }}
      />
    </div>
  );
};

// Gradient Boosting Icon - layered tree with gradient indicator
export const GradientBoostingIcon = ({ size = 24, className = "" }) => {
  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <FaLayerGroup 
        size={size} 
        className={className}
        style={{ color: '#8b5cf6' }}
      />
      <MdTrendingUp 
        size={size * 0.5} 
        style={{ 
          position: 'absolute', 
          bottom: -2, 
          right: -2, 
          color: '#10b981' 
        }}
      />
    </div>
  );
};

// XGBoost Icon - using cubes with lightning for boost
export const XGBoostIcon = ({ size = 24, className = "" }) => {
  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <FaCubes 
        size={size} 
        className={className}
        style={{ color: '#f59e0b' }}
      />
      <HiLightningBolt 
        size={size * 0.5} 
        style={{ 
          position: 'absolute', 
          top: -2, 
          right: -2, 
          color: '#ef4444' 
        }}
      />
    </div>
  );
};

// Connection Status Icons using professional library icons
export const ConnectionStatusIcon = ({ connected, size = 16 }) => {
  if (connected) {
    return (
      <FaCheckCircle 
        size={size} 
        style={{ color: '#22c55e' }} 
      />
    );
  }
  return (
    <FaTimesCircle 
      size={size} 
      style={{ color: '#ef4444' }} 
    />
  );
};
