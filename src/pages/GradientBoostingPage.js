import React from 'react';
import AlgorithmPage from '../components/AlgorithmPage';

const GradientBoostingPage = () => {
  const defaultParams = {
    n_estimators: 100,
    learning_rate: 0.1,
    max_depth: 3
  };

  return (
    <AlgorithmPage
      algorithm="gradient_boosting"
      title="Gradient Boosting Classifier"
      description="Gradient Boosting - builds models sequentially, where each new model minimizes the loss function using gradient descent."
      defaultParams={defaultParams}
    />
  );
};

export default GradientBoostingPage;
