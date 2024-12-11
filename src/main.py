import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

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

plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0],
            X_reduced[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.7,
            edgecolors='k')

plt.scatter(centroids_reduced[:, 0],
            centroids_reduced[:, 1],
            c='red',
            marker='X',
            s=200,
            label='Centroids')

plt.title(f"KMeans Clusters (k={k}) with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()

# Step 4: Train a neural network for classification
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)

# Initialize and train the neural network
nn_classifier = MLPClassifier(hidden_layer_sizes=(10, 10),
                              max_iter=500,
                              random_state=42)
nn_classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = nn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Neural Network Classification Accuracy: {accuracy:.2f}")

# Step 5: Prompt user for input and classify
print("Enter SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm:")
user_input = [float(input(f"{feature}: ")) for feature in data.feature_names]
user_input_scaled = scaler.transform([user_input])

# Predict the class
target_class = nn_classifier.predict(user_input_scaled)[0]
target_name = data.target_names[target_class]
print(f"Predicted class: {target_name}")

# Step 6: Highlight the user's input point and its nearest neighbors
user_point_reduced = pca.transform(user_input_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0],
            X_reduced[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.7,
            edgecolors='k')
plt.scatter(centroids_reduced[:, 0],
            centroids_reduced[:, 1],
            c='red',
            marker='X',
            s=200,
            label='Centroids')
plt.scatter(user_point_reduced[:, 0],
            user_point_reduced[:, 1],
            c='blue',
            marker='o',
            s=200,
            label='User Input',
            edgecolors='white')

# Compute distances to all points and find the k nearest neighbors
distances = cdist(user_input_scaled, X_scaled, metric='euclidean').flatten()
neighbors_idx = np.argsort(distances)[:k]
neighbors_reduced = X_reduced[neighbors_idx]

# Highlight the k nearest neighbors
plt.scatter(neighbors_reduced[:, 0],
            neighbors_reduced[:, 1],
            c='orange',
            marker='s',
            s=150,
            label=f'{k} Nearest Neighbors',
            edgecolors='k')

plt.title(f"KMeans Clusters (k={k}) with PCA and User Input")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()



