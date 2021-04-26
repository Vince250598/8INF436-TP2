import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from Tools import analyzeCluster, showClusterDistribution, random_state


def showModelPerformanceKMeans(X_train, y_train, n_clusters):
    n_init = 10
    max_iter = 300
    tol = 0.0001

    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init,
                    max_iter=max_iter, tol=tol,
                    random_state=random_state)
    km = kmeans.fit(X_train)
    X_train_kmeansClustered = kmeans.predict(X_train)
    X_train_kmeansClustered = pd.DataFrame(data=X_train_kmeansClustered, index=X_train.index, columns=['cluster'])

    showClusterDistribution(X_train, km.labels_, 8, 'Evaporation', 'Rainfall')
    showClusterDistribution(X_train, km.labels_, 13, 'Evaporation', 'Rainfall')

    showClusterDistribution(X_train, km.labels_, 14, 'Evaporation', 'Rainfall')

    countByCluster_kMeans, countByLabel_kMeans, countMostFreq_kMeans, accuracyDF_kMeans, overallAccuracy_kMeans, accuracyByLabel_kMeans = analyzeCluster(
        X_train_kmeansClustered, y_train)
    print("Accuracy by cluster from K-Means: \n", accuracyByLabel_kMeans)
    print("Overall accuracy from K-Means clustering: ", overallAccuracy_kMeans)
    print("Standard deviation from K-Means: ", accuracyByLabel_kMeans.std())


def showDifferentParamsPerformance(X_train, y_train, min_clusters, max_clusters):
    n_init = 10
    max_iter = 300
    tol = 0.0001
    searchRange = range(min_clusters, max_clusters)

    kMeans_inertia = pd.DataFrame(data=[], index=searchRange,
                                  columns=['inertia'])
    overallAccuracy_kMeansDF = pd.DataFrame(data=[], index=searchRange, columns=['overallAccuracy'])

    best_params = pd.DataFrame(columns=['n_clusters', 'n_init', 'overall_accuracy'])

    for n_clusters in searchRange:
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init,
                        max_iter=max_iter, tol=tol,
                        random_state=random_state)
        kmeans.fit(X_train)
        kMeans_inertia.loc[n_clusters] = kmeans.inertia_
        X_train_kmeansClustered = kmeans.predict(X_train)
        X_train_kmeansClustered = pd.DataFrame(data=X_train_kmeansClustered, index=X_train.index, columns=['cluster'])

        countByCluster_kMeans, countByLabel_kMeans, countMostFreq_kMeans, accuracyDF_kMeans, overallAccuracy_kMeans, accuracyByLabel_kMeans = analyzeCluster(
            X_train_kmeansClustered, y_train)
        overallAccuracy_kMeansDF.loc[n_clusters] = overallAccuracy_kMeans

        best_params.loc[len(best_params.index)] = [n_clusters, n_init, overallAccuracy_kMeans]
        print(best_params.loc[len(best_params.index) - 1])

    print(best_params.sort_values(by=['overall_accuracy']))

    kMeans_inertia.plot()

    overallAccuracy_kMeansDF.plot()
    plt.show()
