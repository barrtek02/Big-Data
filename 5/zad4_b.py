
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Perform SVD decomposition with TruncatedSVD
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X_std)

# Visualize reduced dataset
plt.scatter(X_svd[:, 0], X_svd[:, 1], c=y)
plt.xlabel('SV 1')
plt.ylabel('SV 2')
plt.show()
