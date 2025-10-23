import React from 'react';
import AlgorithmPage from '../components/AlgorithmPage';

const LightGBMPage = () => {
  const defaultParams = {
    n_estimators: 100,
    learning_rate: 0.1,
    max_depth: 3
  };

  return (
    <AlgorithmPage
      algorithm="lightgbm"
      title="LightGBM Classifier"
      description="Light Gradient Boosting Machine - a fast, distributed, high-performance gradient boosting framework with leaf-wise tree growth."
      defaultParams={defaultParams}
    />
  );
};

export default LightGBMPage;
