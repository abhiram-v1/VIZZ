import React from 'react';
import AlgorithmPage from '../components/AlgorithmPage';

const XGBoostPage = () => {
  const defaultParams = {
    n_estimators: 100,
    learning_rate: 0.1,
    max_depth: 3
  };

  return (
    <AlgorithmPage
      algorithm="xgboost"
      title="XGBoost Classifier"
      description="Extreme Gradient Boosting - an optimized gradient boosting framework that's highly efficient and scalable, with built-in regularization."
      defaultParams={defaultParams}
    />
  );
};

export default XGBoostPage;
