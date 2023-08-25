from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, \
    roc_auc_score, RocCurveDisplay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Załaduj zbiory danych
iris = load_iris()
wine = load_wine()
digits = load_digits()

# Podziel zbiór na zestaw treningowy i testowy
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(iris.data, iris.target)
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(wine.data, wine.target)
X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(digits.data, digits.target)


# Funkcja klasyfikacji dla k-najbliższych sąsiadów
def classification_task(name, X_train, y_train, X_test, y_test):
    if name == 'knn':  # k-Nearest Neighbors
        classifier = KNeighborsClassifier()
        print('k-Nearest Neighbors scores:')
    elif name == 'svm':
        classifier = SVC(probability=True)
        print('Support Vector Machines scores:')
    elif name == 'dt':
        classifier = DecisionTreeClassifier()
        print('DecisionTree scores:')
    elif name == 'rf':
        classifier = RandomForestClassifier()
        print('Random Forest scores:')
    else:
        print('Wrong name')
        return

    classifier.fit(X_train, y_train)

    # Dokonaj predykcji na zestawie testowym
    y_pred = classifier.predict(X_test)

    # Oblicz i wyświetl wyniki
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    tpr, fpr, _ = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)

    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {round(accuracy * 100, 2)} %    ", f"Precision: {round(precision * 100, 2)} %     ",
          f"Recall: {round(recall * 100, 2)} %      ", f"F1 Score: {round(f1 * 100, 2)} %,      "
                                                       f"ROC Curve: {round(roc_auc, 2)}")
    print('')






classifiers = ['knn', 'svm', 'dt', 'rf']
# Run the task
print("IRIS DATASET\n")
for classifier in classifiers:
    classification_task(classifier, X_train_iris, y_train_iris, X_test_iris, y_test_iris)

print('----------------------------------------------------------------------------------------------------')
print("WINE DATASET\n")
for classifier in classifiers:
    classification_task(classifier, X_train_wine, y_train_wine, X_test_wine, y_test_wine)

print('----------------------------------------------------------------------------------------------------')
print("DIGITS DATASET\n")
for classifier in classifiers:
    classification_task(classifier, X_train_digits, y_train_digits, X_test_digits, y_test_digits)
plt.show()