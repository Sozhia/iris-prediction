import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import iris dataset
iris = datasets.load_iris()

specs = iris.data  # Sepals and petals dimensions
tags = iris.target  # Flowers type

numero_vueltas = np.ones((specs.shape[0], 1))  # Columna de unos
specs = np.hstack((specs, numero_vueltas))

# Dividir en conjunto de entrenamiento y prueba
specs_train, specs_test, tags_train, tags_test = train_test_split(
  specs, tags, test_size=0.2, random_state=42)

# Normalizar las características
scaler = StandardScaler()
specs_train = scaler.fit_transform(specs_train)
tags_test = scaler.transform(tags_test)


