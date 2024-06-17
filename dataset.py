import pandas as pd
import numpy as np

# Generate a sample dataset
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    'TransactionID': np.arange(n_samples),
    'Amount': np.random.uniform(1, 5000, n_samples),
    'Time': np.random.randint(1, 24, n_samples),
    'Age': np.random.randint(18, 70, n_samples),
    'Income': np.random.uniform(20000, 100000, n_samples),
    'MerchantID': np.random.randint(1, 100, n_samples),
    'Class': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
})

print(data.head())

# Save the dataset to a CSV file
data.to_csv('credit_card_transactions.csv', index=False)
