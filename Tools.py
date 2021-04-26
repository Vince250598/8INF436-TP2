from scipy.cluster.hierarchy import fcluster
import pandas as pd
import matplotlib.pyplot as plt

random_state = 2018

def find_hierarchical_clustering_distance_threshold(n_clusters, fastCluster, X_train):
    distance_threshold = 0
    while True:
        print("Trying distance threshold " + str(distance_threshold))
        clusters = fcluster(fastCluster, distance_threshold, criterion='distance')
        X_train_hierClustered = pd.DataFrame(data=clusters, index=X_train.index, columns=['cluster'])
        if len(X_train_hierClustered['cluster'].unique()) <= n_clusters:
            break
        distance_threshold = distance_threshold + 1

    return distance_threshold

def showClusterDistribution(X_train, trainedModelLabels, clusterNumber, *attributes):
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = X_train.index.values
    cluster_map['cluster'] = trainedModelLabels
    chosenClusterindexes = cluster_map[cluster_map.cluster == clusterNumber]
    chosenCluster = X_train.loc[chosenClusterindexes.data_index]
    for attribute in attributes:
        chosenCluster.hist(column=attribute)

    plt.show()

# j'utilise la fonction du tutoriel 2 pour analyser la performance de chaque type de clustering
def analyzeCluster(clusterDF, labelsDF):
    countByCluster = pd.DataFrame(data=clusterDF['cluster'].value_counts())
    countByCluster.reset_index(inplace=True, drop=False)
    countByCluster.columns = ['cluster', 'clusterCount']
    preds = pd.concat([labelsDF, clusterDF], axis=1)
    preds.columns = ['trueLabel', 'cluster']
    countByLabel = pd.DataFrame(data=preds.groupby('trueLabel').count())
    countMostFreq = pd.DataFrame(data=preds.groupby('cluster').agg(lambda x: x.value_counts().iloc[0]))
    countMostFreq.reset_index(inplace=True, drop=False)
    countMostFreq.columns = ['cluster', 'countMostFrequent']
    accuracyDF = countMostFreq.merge(countByCluster, left_on="cluster", right_on="cluster")
    overallAccuracy = accuracyDF.countMostFrequent.sum() / accuracyDF.clusterCount.sum()
    accuracyByLabel = accuracyDF.countMostFrequent / accuracyDF.clusterCount
    return countByCluster, countByLabel, countMostFreq, accuracyDF, overallAccuracy, accuracyByLabel