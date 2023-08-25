import numpy as np
import matplotlib.pyplot as plt


# import pandas as pd
# import seaborn as sns


class ModelLiniowy:

    def __init__(self, method='explicit'):
        self.method = method

    @staticmethod
    def __add_constant__(X):
        return np.column_stack([X, np.ones((X.shape[0], 1))])

    def fit(self, X, Y):
        X_ext = self.__add_constant__(X)
        if self.method == 'explicit':
            self.a = Y.T @ X_ext @ np.linalg.inv(X_ext.T @ X_ext)
        else:
            pass

    def show_params(self):
        for j in range(self.a.shape[1]):
            print(f'a_{j} = {self.a[0, j]:0.4f}')

    def predict(self, X):
        X = np.array(X)
        X_ext = self.__add_constant__(X)
        return X_ext @ self.a.T


# ------------------------------------------------------------------------------#
def rysuj_model_i_dane(model, X, Y):
    X_test = np.linspace(start=X.min(), stop=X.max(), num=300)
    Y_test = model.predict(X_test)
    plt.scatter(X, Y, alpha=0.5)
    plt.plot(X_test, Y_test, color='tab:orange', linewidth=3)


X = np.array([[1], [2], [3], [4]
              # ,[6]
             ])
Y = np.array([[2],[3],[3],[2]
              # ,[3]
             ])

# X = np.log(X)
# Y = np.log(Y)

lin = ModelLiniowy()
lin.fit(X, Y)
X_test = np.linspace(start=X.min(), stop=X.max(), num=300)
Y_test = lin.predict(np.log(X_test))
plt.scatter(X, Y, alpha=0.5)
plt.plot(X_test, Y_test, color='tab:orange', linewidth=3)
plt.show()