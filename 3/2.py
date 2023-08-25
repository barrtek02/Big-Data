import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np



#wygeneruj dane
series = np.random.normal(0, 10, size=21).reshape(-1, 1)

#przygotuj narzędzia do standaryzacji
scaler = preprocessing.StandardScaler()

# wykonaj standaryzacje
series_std = scaler.fit_transform(series)

#zilustruj wyniki
plt.plot(np.arange(len(series)), series, 'tab:blue', label='Dane')
plt.plot(np.arange(len(series)), series_std, 'tab:red', label='Dane po standaryzacji')
plt.xlabel('Chwila czasu')
plt.ylabel('Wartość')
plt.legend()
plt.tight_layout()
plt.show()

print(f'średnia: {np.round(series_std.mean(),4)}', f'odchylenie standardowe: {np.round(series_std.std(),4)}')