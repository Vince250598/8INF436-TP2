from sklearn.cluster import DBSCAN
import pandas as pd

from Tools import showClusterDistribution, analyzeCluster


def showModelPerformanceDBSCAN(X_train, y_train, eps, min_samples):
    leaf_size = 30
    n_jobs = -1

    db = DBSCAN(eps=eps, min_samples=min_samples, leaf_size=leaf_size, n_jobs=n_jobs)

    X_train_dbscanClustered = db.fit_predict(X_train)
    X_train_dbscanClustered = pd.DataFrame(data=X_train_dbscanClustered, index=X_train.index, columns=['cluster'])

    showClusterDistribution(X_train, db.labels_, 0, 'Evaporation', 'Rainfall')
    showClusterDistribution(X_train, db.labels_, 17, 'Evaporation', 'Rainfall')

    print(X_train_dbscanClustered)

    countByCluster_dbscan, countByLabel_dbscan, countMostFreq_dbscan, accuracyDF_dbscan, overallAccuracy_dbscan, accuracyByLabel_dbscan = analyzeCluster(
        X_train_dbscanClustered, y_train)
    print("Accuracy by cluster from DBSCAN: \n", accuracyByLabel_dbscan)
    print("Overall accuracy from DBSCAN: ", overallAccuracy_dbscan)
    print("Standard deviation from DBSCAN: ", accuracyByLabel_dbscan.std())
    print("Cluster Results for DBSCAN")
    print(countByCluster_dbscan)
