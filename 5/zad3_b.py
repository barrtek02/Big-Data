import matplotlib.pyplot as plt
import openml
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


# Wczytaj dataset
"""The model evaluates cars according to the following concept structure:

CAR                      car acceptability
. PRICE                  overall price
. . buying               buying price
. . maint                price of the maintenance
. TECH                   technical characteristics
. . COMFORT              comfort
. . . doors              number of doors
. . . persons            capacity in terms of persons to carry
. . . lug_boot           the size of luggage boot
. . safety               estimated safety of the car"""

dataset = openml.datasets.get_dataset(40975)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

# Konwertuj dane używając one-hot encoding
X = pd.get_dummies(X)

# Standaryzuj dane
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
#Dokonaj redukcji wymiarowości LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_std, y)
# Zwizualizuj dane
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y.cat.codes)

plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('Car Evaluation Database LDA')

plt.show()
