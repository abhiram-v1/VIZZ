import React from 'react';
import AlgorithmPage from '../components/AlgorithmPage';

const XGBoostPage = () => {
  const defaultParams = {
    n_estimators: 150, // Increased for more complex boundaries
    learning_rate: 0.08, // Lower for more refined learning
    max_depth: 5 // Increased depth for complex, adaptive boundaries
  };

  return (
    <AlgorithmPage
      algorithm="xgboost"
      title="XGBoost Classifier"
      description="Extreme Gradient Boosting - an advanced gradient boosting algorithm that achieves the perfect balance of speed and accuracy, optimized for real-world performance with parallel processing and regularization."
      defaultParams={defaultParams}
    />
  );
};

export default XGBoostPage;
