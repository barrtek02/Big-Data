import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np

#wygeneruj syntetyczny szereg czasowy
x = np.linspace(0, 4*np.pi, 100).reshape(-1, 1)
series = np.sin(x)

# przygotuj narzędzie do skalowania
minmax = preprocessing.MinMaxScaler(feature_range=(0,1))

# przeskaluj dane
series_S = minmax.fit_transform(series)

#zwizualizuj wyniki
plt.plot(x, series, 'tab:blue', label='Dane')
plt.plot(x, series_S, 'tab:red', label='Przeskalowane dane')
plt.plot(np.ones(int(np.max(x))+1), 'tab:green')
plt.plot(np.ones(int(np.max(x))+1) * 0, 'tab:green')
plt.title('sin(x)')
plt.xlabel('Chwila czasu')
plt.legend()
plt.tight_layout()
plt.show()
