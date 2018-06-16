import numpy as np
import pandas as pd
import community as co


class CommunityBestPartition(object):

    def __init__(self, partition=None, graph=None):
        if partition is None and graph is None:
            raise ValueError("Either partition or graph must be provided")

        self.partition = partition or co.best_partition(graph)

    def fit(self, X, y):
        pass

    def produce(self, X):
        values = [b for a, b in self.partition.items()]
        missing_community_index = np.max(values) + 10

        result = pd.Series(index=X.index)

        for i in X.index:
            node = X.loc[i][0]

            if node in self.partition:
                community = self.partition[node]

            elif str(node) in self.partition:
                community = self.partition[str(node)]

            else:
                community = missing_community_index

                # increment missing index
                missing_community_index += 1

            result.loc[i] = community

        return result.values
