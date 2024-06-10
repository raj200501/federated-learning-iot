import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('../data/sample_data.csv')

# Basic data exploration
print(data.head())
print(data.describe())
print(data.info())

# Visualize data distribution
sns.pairplot(data, hue='label')
plt.show()

# Visualize correlation
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()
