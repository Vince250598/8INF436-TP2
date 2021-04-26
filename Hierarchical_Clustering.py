import fastcluster
from scipy.cluster.hierarchy import fcluster
import pandas as pd

from Tools import find_hierarchical_clustering_distance_threshold, showClusterDistribution, analyzeCluster


def showModelPerformanceHierarchical(X_train, y_train):
    fc = fastcluster.linkage_vector(X_train, method='ward', metric='euclidean')

    distance = find_hierarchical_clustering_distance_threshold(23, fc, X_train)  # le résultat est 174

    clusters = fcluster(fc, distance, criterion='distance')
    X_train_hierClustered = pd.DataFrame(data=clusters, index=X_train.index, columns=['cluster'])
    print(X_train_hierClustered)
    print("Number of distinct clusters: ", len(X_train_hierClustered['cluster'].unique()))

    showClusterDistribution(X_train, clusters, 6, 'Evaporation', 'Rainfall')

    countByCluster_hierClust, countByLabel_hierClust, countMostFreq_hierClust, accuracyDF_hierClust, overallAccuracy_hierClust, accuracyByLabel_hierClust = analyzeCluster(
        X_train_hierClustered, y_train)
    print("Accuracy by cluster from hierarchical clustering: \n", accuracyByLabel_hierClust)
    print("Overall accuracy from hierarchical clustering: ", overallAccuracy_hierClust)
    print("Standard deviation from hierarchical clustering: ", accuracyByLabel_hierClust.std())