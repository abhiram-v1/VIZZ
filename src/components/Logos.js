import React from 'react';
import { 
  // Icons we actually use
  FaBrain,
  FaTree, FaLayerGroup, FaBolt, FaFire, FaCat, FaChartBar,
  FaCheckCircle, FaTimesCircle, FaCubes
} from 'react-icons/fa';
import { 
  HiLightningBolt, HiSparkles
} from 'react-icons/hi';
import { 
  MdCompareArrows, MdTrendingUp, MdSpeed
} from 'react-icons/md';

// Main Logo Component using professional brain icon
export const MainLogo = ({ size = 40, className = "" }) => (
  <FaBrain 
    size={size} 
    className={className} 
    style={{ color: '#667eea' }}
  />
);

// Tree Visualization Icon
export const TreeVizIcon = ({ size = 24, className = "" }) => (
  <FaTree 
    size={size} 
    className={className}
    style={{ color: '#22c55e' }}
  />
);

// AdaBoost Icon - using tree icon with bolt for boosting
export const AdaBoostIcon = ({ size = 24, className = "" }) => {
  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <FaTree 
        size={size} 
        className={className}
        style={{ color: '#3b82f6' }}
      />
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

// LightGBM Icon - using sparkles for light gradient boosting
export const LightGBMIcon = ({ size = 24, className = "" }) => {
  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <HiSparkles 
        size={size} 
        className={className}
        style={{ color: '#06b6d4' }}
      />
      <MdSpeed 
        size={size * 0.4} 
        style={{ 
          position: 'absolute', 
          bottom: -2, 
          right: -2, 
          color: '#0891b2' 
        }}
      />
    </div>
  );
};

// CatBoost Icon - using cat with boost indicator
export const CatBoostIcon = ({ size = 24, className = "" }) => {
  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <FaCat 
        size={size} 
        className={className}
        style={{ color: '#ec4899' }}
      />
      <FaFire 
        size={size * 0.4} 
        style={{ 
          position: 'absolute', 
          top: -2, 
          right: -2, 
          color: '#f97316' 
        }}
      />
    </div>
  );
};

// Compare All Icon - using professional comparison icon
export const CompareIcon = ({ size = 24, className = "" }) => {
  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <FaChartBar 
        size={size} 
        className={className}
        style={{ color: '#64748b' }}
      />
      <MdCompareArrows 
        size={size * 0.5} 
        style={{ 
          position: 'absolute', 
          top: -2, 
          right: -2, 
          color: '#475569' 
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
