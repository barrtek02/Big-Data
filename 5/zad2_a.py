import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# wczytanie danych
housing = fetch_california_housing()

X = housing.data
y = housing.target

# standaryzacja danych
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# redukcja wymiarowości
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# wykres 2D
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
