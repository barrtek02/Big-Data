import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X = np.linspace(0, 1, 100)
a_0, a_1 = random.randint(1, 10), random.randint(0, 10)
Y = a_0 * X + a_1 + np.random.normal(size=len(X), scale=0.5)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

print(f'Random params -> a_0: {a_0}, a_1: {a_1}')

X_ext = np.column_stack([X_train, np.ones((X_train.shape[0], 1))])
alpha = 0.5
params = Y_train.T @ X_ext @ np.linalg.inv(X_ext.T @ X_ext + alpha * np.eye(X_ext.shape[1]))

print(f'Estimated params -> a_0: {round(params[0], 3)}, a_1: {round(params[1], 3)}')

Y_pred = params[0] * X_test + params[1]

print(r2_score(Y_test, Y_pred))

plt.scatter(X, Y)
plt.plot(X_test, Y_pred, color='tab:orange', linewidth=3)
plt.show()