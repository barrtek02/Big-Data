import openml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Wczytaj dataset
# dataset = openml.datasets.get_dataset(31) #credit-g
dataset = openml.datasets.get_dataset(40975) #credit-g
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

# Konwertuj dane używając one-hot encoding, wybieramy istotne kolumny
# X = pd.get_dummies(X, columns=['checking_status', 'credit_history', 'purpose', 'savings_status',
#                                'employment', 'personal_status', 'other_parties', 'property_magnitude',
#                                'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker'])
X = pd.get_dummies(X)

# Standaryzuj dane
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Zredukuj wymiary za pomocą PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)
print(X_pca)
# Zwizualizuj dane
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.cat.codes)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
