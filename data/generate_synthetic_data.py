import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(loc=0, scale=1, size=num_samples),
        'feature2': np.random.normal(loc=5, scale=2, size=num_samples),
        'label': np.random.randint(0, 2, size=num_samples)
    }
    df = pd.DataFrame(data)
    df.to_csv('data/sample_data.csv', index=False)
    print(f"Generated {num_samples} samples of synthetic data.")

if __name__ == "__main__":
    generate_synthetic_data()
