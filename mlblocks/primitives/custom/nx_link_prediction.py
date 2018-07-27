import logging

import networkx as nx
import numpy as np

LOGGER = logging.getLogger(__name__)


class LinkPrediction(object):

    def fit(self, X, y):
        pass

    def produce(self, X, node_columns, graphs=None):
        pairs = X[node_columns].values
        # X = pd.DataFrame(index=X.index)

        for i, graph in enumerate(graphs):
            def apply(function):
                try:
                    values = function(graph, pairs)
                    return np.array(list(values))[:, 2]
                except ZeroDivisionError:
                    LOGGER.warn("ZeroDivisionError captured running %s", function)
                    return np.zeros(len(pairs))

            i = str(i)
            X['jc_' + i] = apply(nx.jaccard_coefficient)
            X['rai_' + i] = apply(nx.resource_allocation_index)
            X['aai_' + i] = apply(nx.adamic_adar_index)
            X['pa_' + i] = apply(nx.preferential_attachment)

        return X
