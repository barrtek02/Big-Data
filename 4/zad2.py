import numpy as np
import nolds

# generowanie szeregu czasowego o 1000 elementach
ts = np.random.normal(0, 1, 1000)
# ts = np.linspace(0, 100, 100)

# obliczenie entropii, wymiaru fraktalnego oraz wykładnika Hursta
entropy = nolds.sampen(ts)
# miara nieporządku, miara nieprzewidywalności lub niepewności informacji,
# im większa entropia, tym większa ilość informacji przekazywana, ale trudniej przewidziec przyszlosc

correlation_dimension = nolds.corr_dim(ts, 1)
# współczynnik korelacji to jeden ze sposobów analizowania wymiaru fraktalnego
# miara złożoności, jak bardzo punkty w przestrzeni są rozproszone w róznych kierunkach
# im wyższy współczynnik korelacji tym bardziej złożona struktura systemu

# f. Weierstrassa - self similar time series
hurst_exponent = nolds.hurst_rs(ts)
# miara czy dany szereg ma tendencję do oscylowania wokół swojej średniej wartości
#  >0,5 stabilne i mniej zmienne w czasie

# wyświetlenie wyników
print("Entropia: ", entropy)
print("Correlation Dimension: ", correlation_dimension)
print("Wykładnik Hursta: ", hurst_exponent)
