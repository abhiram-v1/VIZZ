import React from 'react';
import AlgorithmPage from '../components/AlgorithmPage';

const GradientBoostingPage = () => {
  const defaultParams = {
    n_estimators: 150, // Increased for more complex, adaptive boundaries
    learning_rate: 0.08, // Lower learning rate allows more refinement
    max_depth: 5 // Increased depth for more complex decision boundaries
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
