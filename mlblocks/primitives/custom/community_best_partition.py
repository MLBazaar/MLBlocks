import community as co
import numpy as np
import pandas as pd


class CommunityBestPartition(object):

    def fit(self, X, y):
        pass

    def produce(self, X, partition=None, graph=None):
        partition = partition or co.best_partition(graph)
        values = [b for a, b in partition.items()]
        missing_community_index = np.max(values) + 10

        result = pd.Series(index=X.index)

        for i in X.index:
            node = X.loc[i][0]

            if node in partition:
                community = partition[node]

            elif str(node) in partition:
                community = partition[str(node)]

            else:
                community = missing_community_index

                # increment missing index
                missing_community_index += 1

            result.loc[i] = community

        return result.values
