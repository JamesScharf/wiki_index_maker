"""
Build a new index for my wiki in markdown format.
"""

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import queue


class Clusterer:
    """
    Build clusters.
    """
    def __init__(self):
        self.read_features()
        self.cluster_data()
        self.cluster_data()

    def read_features(self):
        """
        Read in the features from the file.
        """
        self.data = pd.read_csv("featurized_text.tsv", sep="\t")
        self.features = self.data.drop(self.data.columns[0], axis=1)
        self.normalized_ft = normalize(self.features)

    def cluster_data(self):
        """
        Produce clusters from the data
        """
        Z = hierarchy.linkage(self.normalized_ft, method="ward")
        # save specific heights
        self.wiki_index_df = self.data[[self.data.columns[0]]]
        self.wiki_index_df.columns = ["file_name"]
        for i, x in enumerate(range(18, 27, 1)):
            x = x / 10.0
            clusters = fcluster(Z, t=x, criterion="distance")
            self.wiki_index_df["clust_lvl_" + str(i)] = pd.Series(clusters)
            print(f"Number of clusters at level {x}: {len(set(clusters))}")
        print(self.wiki_index_df)
        # now plot the dendrogram
        plt.figure()
        hierarchy.dendrogram(Z)
        plt.show()


class Index_Creator:
    """
    Generate an index from
    the clustered data.
    """

    def __init__(self):
        self.clust = Clusterer()
        self.wiki_clusters = self.clust.wiki_index_df
        self.get_layer(3)

    
    def get_layer(self, lvl):

        labels = self.wiki_clusters["clust_lvl_" + str(lvl)].unique()
        for cluster in labels:
            subset = self.wiki_clusters[
                    self.wiki_clusters["clust_lvl_" + str(lvl)] == cluster]
            print(cluster, subset["file_name"].tolist())

if __name__ == "__main__":
    Index_Creator()
