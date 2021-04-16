"""
Build a new index for my wiki in markdown format.
"""

import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import LocalOutlierFactor


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
        self.wiki_index_df = self.data
        
        # now do top-level clustering
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.6).fit(self.normalized_ft)
        self.wiki_index_df["lvl_0"] = clustering.labels_ 

        # now let's do some post-processing to create MISC category
        #labels = LocalOutlierFactor(n_neighbors=2).fit_predict(self.normalized_ft)
        #self.wiki_index_df["outlier"] = labels
        #self.wiki_index_df.loc[self.wiki_index_df.outlier == -1, "lvl_0"] = -1
        #self.wiki_index_df = self.wiki_index_df.sort_values(by=["lvl_0"])

class Index_Creator:
    """
    Generate an index from
    the clustered data.
    """

    def __init__(self):
        self.clust = Clusterer()
        self.wiki_clusters = self.clust.wiki_index_df
        self.draw_index()

    def draw_index(self):
        labels = self.wiki_clusters["lvl_0"].unique()
        
        print("# James Scharf's Wiki")
        print("*This index has been automatically generated through clustering.*")
        for l in labels:
            print(f"## {l}")

            def print_list(file_name):
                print(f"\t-[[{file_name}]]")

            subset = self.wiki_clusters[self.wiki_clusters["lvl_0"] == l]
            subset[subset.columns[0]].apply(print_list)


if __name__ == "__main__":
    Index_Creator()
