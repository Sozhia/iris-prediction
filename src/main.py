import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



# Load the Iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Target labels

# Step 1: Explore the dataset
print("Feature names:", data.feature_names)
print("Target classes:", data.target_names)
print("First 5 rows of features:\n", X[:5])
print("First 5 target labels:", y[:5])

# Step 2: Standardize the features for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = int(input("Enter the number of clusters for K-Means: "))

# Perform KMeans with the user-defined number of clusters
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualize clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)
centroids_reduced = pca.transform(centroids)
