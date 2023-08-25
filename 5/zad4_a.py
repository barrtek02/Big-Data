import matplotlib.pyplot as plt
import openml
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

# Wczytaj dataset
dataset = openml.datasets.get_dataset(31)  # credit-g
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

# Konwertuj dane używając one-hot encoding, wybieramy istotne kolumny
X = pd.get_dummies(X, columns=['checking_status', 'credit_history', 'purpose', 'savings_status',
                               'employment', 'personal_status', 'other_parties', 'property_magnitude',
                               'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker'])

# Standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Perform SVD decomposition with TruncatedSVD
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X_std)
# Visualize reduced dataset
plt.scatter(X_svd[:, 0], X_svd[:, 1], c=y.cat.codes)
plt.xlabel('SV 1')
plt.ylabel('SV 2')
plt.show()
