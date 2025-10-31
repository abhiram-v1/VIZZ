import React from 'react';
import AlgorithmPage from '../components/AlgorithmPage';

const AdaBoostPage = () => {
  const defaultParams = {
    n_estimators: 100, // Increased for more complex boundaries
    learning_rate: 0.8, // Slightly lower for more stable learning
    max_depth: 3 // Increased from 1 to allow more complex decision boundaries
  };

  return (
    <AlgorithmPage
      algorithm="adaboost"
      title="AdaBoost Classifier"
      description="Adaptive Boosting - combines multiple weak learners sequentially, where each subsequent learner focuses on the mistakes of the previous ones."
      defaultParams={defaultParams}
    />
  );
};

export default AdaBoostPage;
