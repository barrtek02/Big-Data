from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 1]
y_pred = [0.2, 0.8, 0.3, 0.6, 0.1, 0.9, 0.7, 0.4, 0.5, 0.6]

# Obliczenie FPR i TPR
fpr, tpr, thresholds = roc_curve(y_true, y_pred)

# Obliczenie AUC
roc_auc = auc(fpr, tpr)

# Rysowanie krzywej ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Krzywa ROC (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Odsetek fałszywie pozytywnych')
plt.ylabel('Odsetek prawdziwie pozytywnych')
plt.title('Krzywa ROC')
plt.legend(loc="lower right")
plt.show()
