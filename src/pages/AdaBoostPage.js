import React from 'react';
import AlgorithmPage from '../components/AlgorithmPage';

const AdaBoostPage = () => {
  const defaultParams = {
    n_estimators: 50,
    learning_rate: 1.0,
    max_depth: 1
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
