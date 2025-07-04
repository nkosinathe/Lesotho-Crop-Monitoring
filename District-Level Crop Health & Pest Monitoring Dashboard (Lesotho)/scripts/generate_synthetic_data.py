import numpy as np # type: ignore
import pandas as pd # type: ignore
import os

np.random.seed(42)
n_samples = 1000

def generate_sample(label):
    if label == 0:  # Healthy
        return {
            'b5': np.random.normal(1400, 100),
            'b6': np.random.normal(1600, 100),
            'b7': np.random.normal(1700, 120),
            'b11': np.random.normal(1800, 100),
            'b12': np.random.normal(1900, 100),
            'ndvi': np.random.normal(0.7, 0.05),
            'label': label
        }
    else:  # Diseased
        return {
            'b5': np.random.normal(1100, 100),
            'b6': np.random.normal(1300, 100),
            'b7': np.random.normal(1400, 120),
            'b11': np.random.normal(2100, 100),
            'b12': np.random.normal(2200, 100),
            'ndvi': np.random.normal(0.4, 0.1),
            'label': label
        }

samples = [generate_sample(np.random.choice([0, 1], p=[0.6, 0.4])) for _ in range(n_samples)]
df = pd.DataFrame(samples)
df['ndvi'] = df['ndvi'].clip(0, 1)

os.makedirs("data", exist_ok=True)
df.to_csv("data/disease_training.csv", index=False)
print("✅ Synthetic dataset saved to data/disease_training.csv")
