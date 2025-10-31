import React from 'react';
import { 
  // Robot/Machine Learning
  FaRobot, FaBrain, FaMicrochip,
  // Trees/Nature
  FaTree, FaLeaf, FaSeedling,
  // People/Team
  FaUsers, FaUser, FaUserFriends,
  // Science/Experiment
  FaFlask, FaMicroscope, FaVial,
  // Target/Goals
  FaBullseye, FaCrosshairs, FaDotCircle,
  // Charts/Data
  FaChartLine, FaChartBar, FaChartArea, FaChartPie,
  // Process/Workflow
  FaSync, FaRedo, FaCog, FaWrench,
  // Speed/Performance
  FaRocket, FaBolt,
  // Security/Shield
  FaShieldAlt,
  // Analytics/Data Science
  FaDatabase, FaTable, FaFileAlt,
  // Medical/Health
  FaHospital, FaHeartbeat, FaStethoscope,
  // Playback Controls
  FaPlay, FaPause, FaStepForward, FaStepBackward, FaRedoAlt,
  // Status indicators
  FaCheck, FaTimes, FaCheckCircle, FaTimesCircle,
  // Lightbulb/Ideas
  FaLightbulb,
  // Other
  FaVoteYea, FaBalanceScale, FaArrowDown, FaArrowUp,
  FaChevronRight, FaChevronLeft
} from 'react-icons/fa';

// Icon mapping component
export const Icons = {
  // Machine Learning
  ml: FaRobot,
  brain: FaBrain,
  chip: FaMicrochip,
  
  // Trees
  tree: FaTree,
  leaf: FaLeaf,
  seedling: FaSeedling,
  
  // People
  team: FaUsers,
  user: FaUser,
  users: FaUserFriends,
  person: FaUser,
  
  // Science
  experiment: FaFlask,
  test: FaMicroscope,
  vial: FaVial,
  
  // Target
  target: FaBullseye,
  goal: FaCrosshairs,
  dot: FaDotCircle,
  
  // Charts
  chart: FaChartLine,
  barChart: FaChartBar,
  areaChart: FaChartArea,
  pieChart: FaChartPie,
  analytics: FaChartBar,
  
  // Process
  sync: FaSync,
  reset: FaRedo,
  settings: FaCog,
  gear: FaWrench,
  gears: FaCog, // Using FaCog as replacement for gears
  
  // Speed
  rocket: FaRocket,
  bolt: FaBolt,
  zap: FaBolt, // Using FaBolt as replacement for zap
  
  // Security
  shield: FaShieldAlt,
  protect: FaShieldAlt, // Using FaShieldAlt as replacement
  
  // Data
  database: FaDatabase,
  table: FaTable,
  file: FaFileAlt,
  
  // Medical
  hospital: FaHospital,
  health: FaHeartbeat,
  medical: FaStethoscope,
  
  // Controls
  play: FaPlay,
  pause: FaPause,
  next: FaStepForward,
  previous: FaStepBackward,
  redo: FaRedoAlt,
  
  // Status
  check: FaCheck,
  cross: FaTimes,
  checkCircle: FaCheckCircle,
  timesCircle: FaTimesCircle,
  
  // Ideas
  lightbulb: FaLightbulb,
  idea: FaBrain,
  
  // Other
  vote: FaVoteYea,
  scale: FaBalanceScale,
  arrowDown: FaArrowDown,
  arrowUp: FaArrowUp,
  right: FaChevronRight,
  left: FaChevronLeft
};

// Helper component to render icons
export const Icon = ({ name, size = 20, className = "", color, ...props }) => {
  const IconComponent = Icons[name];
  if (!IconComponent) {
    console.warn(`Icon "${name}" not found`);
    return null;
  }
  return (
    <IconComponent 
      size={size} 
      className={className}
      color={color}
      {...props}
    />
  );
};

export default Icons;

