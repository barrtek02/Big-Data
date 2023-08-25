import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from pandas import read_csv

#wczytaj dane
data = read_csv('Wroclaw.csv')
series = np.array(data['temp']).reshape(1, -1)
dates = data['datetime']
formatted_dates = []
for date in dates:
    formatted_dates.append(date[-2:])

#przygotuj normalizator
normalizer = preprocessing.Normalizer(norm='l2')

#normalizuj dane
series_N = normalizer.transform(series)

#zwizualizuj wyniki
plt.plot(formatted_dates, series.flatten(), 'tab:blue', label='Temperatury')
plt.plot(formatted_dates, series_N.flatten(), 'tab:red', label='Znormalizowane Temperatury')
plt.title('Temperatura Wrocław Luty')
plt.xlabel('Data')
plt.legend()
plt.tight_layout()
plt.show()

print(f'Długość wektora: {np.round(np.linalg.norm(series_N), 4)}')