import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# Load the Iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Target labels

# Step 1: Explore the dataset
print("Feature names:", data.feature_names)
print("Target classes:", data.target_names)
print("First 5 rows of features:\n", X[:5])
print("First 5 target labels:", y[:5])
