# Iris Dataset Clustering and Classification
Using [Iris Species](https://www.kaggle.com/datasets/uciml/iris) dataset. It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.

The columns in this dataset are:

  * Id
  * SepalLengthCm
  * SepalWidthCm
  * PetalLengthCm
  * PetalWidthCm
  * Species

This project demonstrates how to cluster and classify the Iris dataset using K-Means clustering and a Neural Network classifier. The program allows user interaction for defining the number of clusters and inputting specific data points for classification.

## Libraries Used

The following Python libraries are used in this project:

 * [NumPy](https://numpy.org/doc/): For numerical computations and handling data arrays.

 * [Matplotlib](https://matplotlib.org/stable/index.html): For visualizing clusters and classification results.

 * [scikit-learn](https://scikit-learn.org/stable/): For data preprocessing, clustering (K-Means), dimensionality 
 reduction (PCA), and neural network classification.

 * [SciPy](https://docs.scipy.org/doc/scipy/): For calculating distances between points.

## How the Code Works

### Step 1: Load the Iris Dataset

The Iris dataset is loaded using load_iris from sklearn.datasets. It contains:

 * Features: Sepal length, sepal width, petal length, and petal width.

 * Target classes: Setosa, Versicolor, and Virginica.

### Step 2: Standardize the Features

Features are standardized using StandardScaler to improve the performance of clustering and classification algorithms.

### Step 3: K-Means Clustering

 * The user is prompted to input the number of clusters (k).

 * K-Means is performed with k clusters.

 * The results are visualized using Principal Component Analysis (PCA) to reduce data to 2 dimensions for plotting.

### Step 4: Neural Network Classification

 * The dataset is split into training and testing sets using train_test_split.

 * A Multi-Layer Perceptron (MLP) classifier with two hidden layers (10 neurons each) is trained.

 * The model's accuracy is evaluated and displayed.

### Step 5: User Input and Classification

 * The user inputs values for the four features.

 * The input is standardized and classified using the trained neural network.

 * The predicted class and class name are displayed.

### Step 6: Highlight User Input and Nearest Neighbors

 * The user input point is transformed with PCA and plotted alongside clusters.

 * The nearest neighbors (based on Euclidean distance) are identified and highlighted on the plot.

## Inputs
 
1. **Number of Clusters**: Integer input for the K-Means clustering step.

2. **Feature Values**: Four floating-point numbers corresponding to sepal length, sepal width, petal length, and petal width.

## Outputs

1. **Clustering Visualization**: A scatter plot showing K-Means clusters and centroids.

2. **Classification Accuracy**: The accuracy of the neural network classifier on the test set.

3. **Predicted Class**: The predicted Iris species for the user input.

4. **Enhanced Plot**: A scatter plot highlighting the user input and the k nearest neighbors.

## How to Run

1. Ensure the required libraries are installed:
```bash
pip install numpy matplotlib scikit-learn scipy
```

2. Run the script:
```bash
python main.py
```
3. Follow the prompts to input the number of clusters and feature values.

## Key Visualizations

 * **Cluster Plot**: Displays clusters formed by K-Means with centroids marked in red.

 * **Enhanced Cluster Plot**: Highlights the user input and nearest neighbors within the cluster plot.

## Future Improvements

 * Automate parameter selection for K-Means (e.g., using the Elbow method).

 * Optimize neural network hyperparameters.

* Enhance input validation and user experience.
