import time

from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.datasets import make_blobs, make_moons, make_circles, make_classification, make_gaussian_quantiles, \
    make_multilabel_classification
from sklearn.metrics import silhouette_score, calinski_harabasz_score, homogeneity_score, adjusted_rand_score, \
    adjusted_mutual_info_score
import matplotlib.pyplot as plt


def do_clustering(X, y):

    # initialize variables
    best_k = 0
    best_score = -1

    # loop through possible values of K
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, n_init='auto')
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        if score > best_score:
            best_k = k
            best_score = score

    # fit k-means with best k
    kmeans = KMeans(n_clusters=best_k, n_init='auto')
    kmeans.fit(X)

    # fit MeanShift
    ms = MeanShift()
    ms.fit(X)

    # fit DBSCAN
    dbscan = DBSCAN()
    dbscan.fit(X)

    # compute scores
    labels = [kmeans.labels_, ms.labels_, dbscan.labels_]
    scores = [[silhouette_score(X, label), calinski_harabasz_score(X, label), adjusted_rand_score(y, label),
               homogeneity_score(y, label), adjusted_mutual_info_score(y, label)] for label in labels]

    # plot the data and the labels
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

    # plot original data with labels
    for i in range(3):
        axes[0, i].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
        axes[0, i].set_title("Original data")

    # plot k-means clustering with labels
    axes[1, 0].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
    axes[1, 0].set_title(f"K-means clustering with K={best_k}")

    # plot MeanShift clustering with labels
    axes[1, 1].scatter(X[:, 0], X[:, 1], c=ms.labels_, cmap='viridis')
    axes[1, 1].set_title("MeanShift clustering")

    # plot DBSCAN clustering with labels
    axes[1, 2].scatter(X[:, 0], X[:, 1], c=dbscan.labels_, cmap='viridis')
    axes[1, 2].set_title("DBSCAN clustering")

    plt.tight_layout()

    print(
        f'Silhouette_score        --- K-Means: {round(scores[0][0] * 100, 2)}%, Mean Shift: {round(scores[1][0] * 100, 2)}%, DBSCAN: {round(scores[2][0] * 100, 2)}%')
    print(
        f'Calinski Harabasz Score --- K-Means: {round(scores[0][1], 2)}, Mean Shift: {round(scores[1][1], 2)}, DBSCAN: {round(scores[2][1], 2)}')
    print(
        f'Adjusted Rand Score     --- K-Means: {round(scores[0][2] * 100, 2)}%, Mean Shift: {round(scores[1][2] * 100, 2)}%, DBSCAN: {round(scores[2][2] * 100, 2)}%')
    print(
        f'Homogeneity Score       --- K-Means: {round(scores[0][3] * 100, 2)}%, Mean Shift: {round(scores[1][3] * 100, 2)}%, DBSCAN: {round(scores[2][3] * 100, 2)}%')
    print(
        f'Adjusted Mutual Info    --- K-Means: {round(scores[0][4] * 100, 2)}%, Mean Shift: {round(scores[1][4] * 100, 2)}%, DBSCAN: {round(scores[2][4] * 100, 2)}%')

    plt.show()



print('---------------------------------------------------------------------------------------------------------------')
print('Make Blobs DATASET')
X_blobs, y_blobs = make_blobs(n_samples=1000, centers=4)
do_clustering(X_blobs, y_blobs)
print('---------------------------------------------------------------------------------------------------------------')
print('Make Gaussian Quantiles DATASET')
X_gaussian, y_gaussian = make_gaussian_quantiles(n_samples=1000)
do_clustering(X_gaussian, y_gaussian)
