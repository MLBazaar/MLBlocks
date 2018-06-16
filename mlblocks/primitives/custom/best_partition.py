import community as co
import numpy as np
import pandas as pd


class BestPartition(object):

    def fit(self, X, y, node_column=None, graph=None):
        self.node_column = node_column
        self.partition = co.best_partition(graph)

    def produce(self, X):
        values = [b for a, b in self.partition.items()]
        missing_community_index = np.max(values) + 10

        result = pd.Series(index=X.index)

        for i in X.index:
            node = X.loc[i, node_column]

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
