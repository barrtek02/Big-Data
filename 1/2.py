import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score



X = np.linspace(0, 1, 100)
a_0, a_1, a_2, a_3 = random.randint(1, 10), random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)
Y = a_0 * X ** 3 - a_1 * X**2 + a_2*X + a_3 + np.random.normal(size=len(X), scale=0.5)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

print(f'Random params -> a_0: {a_0}, a_1: {a_1}, a_2: {a_2}, a_3: {a_3}')

X_ext = np.column_stack([X_train**3, X_train**2, X_train, np.ones((X_train.shape[0], 1))])
params = Y_train.T @ X_ext @ np.linalg.inv(X_ext.T @ X_ext)
print(f'Estimated params -> a_0: {round(params[0], 3)}, a_1: {round(params[1], 3)}, a_2: {round(params[2], 3)}, a_3: {round(params[3], 3)}')

Y_pred = params[0]*X_test**3+params[1]*X_test**2+params[2]*X_test+params[3]
# print(r2_score(Y_test, Y_pred))


plt.scatter(X, Y)
idx = np.argsort(X_test)
x = X_test[idx]
y = Y_pred[idx]
plt.plot(x, y, color='red')
plt.show()
