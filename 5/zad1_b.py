import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Wczytanie zbioru danych Wine
wine = datasets.load_wine()
X = wine.data
y = wine.target
# Standaryzacja danych
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Redukcja wymiarów do dwóch za pomocą PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# Tworzenie wykresu
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Wine dataset - PCA')
plt.show()
