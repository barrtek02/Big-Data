import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from matplotlib import colors

# Wczytanie zbioru danych Iris
iris = load_iris()
X = iris.data
y = iris.target
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_std, y)

plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)

plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('Iris dataset after LDA')
plt.show()
