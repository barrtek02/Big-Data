import numpy as np
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

# generowanie szeregu czasowego o 100 elementach
x = np.linspace(0, 10*np.pi, 100)
ts = np.sin(x) + np.random.normal(0, 0.15, 100)
plt.plot(ts)

# wyświetlenie wyników
print("Średnia: ", np.mean(ts))
print("Odchylenie standardowe: ", np.std(ts)) #jak wartości różnią się od średniej
print("Wartość maksymalna: ", np.max(ts))
print("Wartość minimalna: ", np.min(ts))
print("Mediana: ", np.median(ts))
print("Kurtoza: ", kurtosis(ts)) #miara ilości skrajnych odstających,
# gdy >0 to jest więcej skrajnych wartości odstających niż w rozkladzie normalnym

plt.show()
