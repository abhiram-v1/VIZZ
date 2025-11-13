#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic churn dataset with realistic distributions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def generate_synthetic_churn_dataset(n_samples=1000, output_file='boosting_small_dataset.csv'):
    """
    Generate a synthetic churn dataset with realistic patterns
    
    Args:
        n_samples: Number of samples to generate
        output_file: Output CSV file path
    """
    np.random.seed(42)
    
    # Generate Customer IDs
    customer_ids = range(1, n_samples + 1)
    
    # Age: Normal distribution around 35-40, clipped to 18-80
    age = np.random.normal(38, 12, n_samples)
    age = np.clip(age, 18, 80).astype(int)
    
    # Credit Score: Normal distribution around 650, clipped to 300-850
    credit_score = np.random.normal(650, 100, n_samples)
    credit_score = np.clip(credit_score, 300, 850).astype(int)
    
    # Geography: Weighted distribution (France: 50%, Spain: 25%, Germany: 25%)
    geography = np.random.choice(['France', 'Spain', 'Germany'], 
                                 size=n_samples, 
                                 p=[0.5, 0.25, 0.25])
    
    # Gender: 50/50 split
    gender = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.5, 0.5])
    
    # Tenure: Normal distribution around 5 years, clipped to 0-10
    tenure = np.random.normal(5, 3, n_samples)
    tenure = np.clip(tenure, 0, 10).astype(int)
    
    # Balance: Right-skewed distribution (many low balances, few high)
    balance = np.random.exponential(10000, n_samples)
    balance = np.clip(balance, 0, 250000).astype(float)
    # Round to 2 decimal places
    balance = np.round(balance, 2)
    
    # Number of Products: Discrete distribution (1-4, weighted towards 1-2)
    num_products = np.random.choice([1, 2, 3, 4], 
                                    size=n_samples, 
                                    p=[0.4, 0.35, 0.15, 0.1])
    
    # Has Credit Card: 70% have credit card
    has_cr_card = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    
    # Is Active Member: 50% are active
    is_active_member = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    
    # Estimated Salary: Normal distribution around 50k, clipped to 10k-200k
    estimated_salary = np.random.normal(50000, 25000, n_samples)
    estimated_salary = np.clip(estimated_salary, 10000, 200000).astype(float)
    estimated_salary = np.round(estimated_salary, 2)
    
    # Generate Churn based on realistic patterns:
    # Higher churn probability for:
    # - Lower credit scores
    # - Inactive members
    # - Lower tenure
    # - Lower balance
    # - More products (overwhelmed customers)
    
    churn_probability = (
        0.3 * (1 - (credit_score - 300) / 550) +  # Lower credit score = higher churn
        0.2 * (1 - is_active_member) +  # Inactive = higher churn
        0.2 * (1 - tenure / 10) +  # Lower tenure = higher churn
        0.15 * (1 - np.clip(balance / 250000, 0, 1)) +  # Lower balance = higher churn
        0.15 * (num_products / 4)  # More products = slightly higher churn
    )
    
    # Add some randomness
    churn_probability = np.clip(churn_probability + np.random.normal(0, 0.1, n_samples), 0, 1)
    
    # Generate churn labels
    churn = (np.random.random(n_samples) < churn_probability).astype(int)
    
    # Ensure balanced classes (approximately 50/50)
    churn_count = churn.sum()
    no_churn_count = n_samples - churn_count
    target_churn = n_samples // 2
    
    if churn_count < target_churn:
        # Need more churn cases
        no_churn_indices = np.where(churn == 0)[0]
        additional_churn_needed = target_churn - churn_count
        if additional_churn_needed > 0 and len(no_churn_indices) > 0:
            indices_to_flip = np.random.choice(no_churn_indices, 
                                               min(additional_churn_needed, len(no_churn_indices)), 
                                               replace=False)
            churn[indices_to_flip] = 1
    elif churn_count > target_churn:
        # Need more no-churn cases
        churn_indices = np.where(churn == 1)[0]
        additional_no_churn_needed = churn_count - target_churn
        if additional_no_churn_needed > 0 and len(churn_indices) > 0:
            indices_to_flip = np.random.choice(churn_indices, 
                                               min(additional_no_churn_needed, len(churn_indices)), 
                                               replace=False)
            churn[indices_to_flip] = 0
    
    # Create DataFrame
    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary,
        'Churn': churn
    })
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"Generated synthetic churn dataset with {len(df)} samples")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nChurn distribution:")
    print(df['Churn'].value_counts())
    print(f"\nChurn rate: {df['Churn'].mean()*100:.2f}%")
    print(f"\nFirst few rows:")
    print(df.head(10))
    print(f"\nDataset saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    # Generate 1000 samples (can be adjusted)
    df = generate_synthetic_churn_dataset(n_samples=1000, output_file='boosting_small_dataset.csv')
    
    # Also save a copy to data directory
    import os
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/boosting_small_dataset.csv', index=False)
    print(f"\nAlso saved to: data/boosting_small_dataset.csv")

