import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import lagrange
from sklearn.metrics import mean_squared_error

# Interpolacja wielomianowa trzeciego stopnia polega na
# znalezieniu funkcji wielomianowej trzeciego stopnia,
# która przechodzi przez cztery punkty na płaszczyźnie.

#generowanie punktów
x = np.linspace(0, 4, 4)
mean = 2.5
std_dev = 1
y = 1 / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2))

#dopasowywanie funckji interpolacyjnej do danych
f_interpolar = lagrange(x, y)

# próbkowanie
x_resampled = np.linspace(0, 4, 30)
y_true = 1 / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-(x_resampled - mean) ** 2 / (2 * std_dev ** 2))
y_resampled = f_interpolar(x_resampled)

# wizualizacja
plt.scatter(x, y)
plt.plot(x_resampled, y_true, color='blue')
plt.plot(x_resampled, y_resampled, color='red', linestyle='--')
plt.show()

#ocena
print(mean_squared_error(y_true, y_resampled))