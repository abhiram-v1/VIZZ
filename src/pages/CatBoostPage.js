import React from 'react';
import AlgorithmPage from '../components/AlgorithmPage';

const CatBoostPage = () => {
  const defaultParams = {
    n_estimators: 100,
    learning_rate: 0.1,
    max_depth: 3
  };

  return (
    <AlgorithmPage
      algorithm="catboost"
      title="CatBoost Classifier"
      description="Categorical Boosting - a gradient boosting framework that handles categorical features natively without preprocessing."
      defaultParams={defaultParams}
    />
  );
};

export default CatBoostPage;
