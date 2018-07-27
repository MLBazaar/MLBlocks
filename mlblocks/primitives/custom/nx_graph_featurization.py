import networkx as nx
import numpy as np
import pandas as pd


class GraphFeaturization(object):

    def fit(self, X, y):
        pass

    def produce_graph(self, X, graph, node_column):
        index_type = type(X[node_column].values[0])

        features = pd.DataFrame(index=graph.nodes)
        features.index = features.index.astype(index_type)

        def apply(function):
            values = function(graph)
            return np.array(list(values.values()))

        features['dc_' + node_column] = apply(nx.degree_centrality)
        features['cc_' + node_column] = apply(nx.closeness_centrality)
        features['bc_' + node_column] = apply(nx.betweenness_centrality)
        features['clustering_' + node_column] = apply(nx.clustering)
        features['ec_' + node_column] = apply(nx.eigenvector_centrality)
        features['kc_' + node_column] = apply(nx.katz_centrality)
        features['comc_' + node_column] = apply(nx.communicability_betweenness_centrality)
        features['lc_' + node_column] = apply(nx.load_centrality)
        features['nr_' + node_column] = apply(nx.algorithms.bipartite.node_redundancy)
        features['lac_' + node_column] = apply(nx.algorithms.bipartite.cluster.latapy_clustering)

        merged = X.merge(features, left_on=node_column, right_index=True, how='left')

        # graph_data = pd.DataFrame(list(graph.nodes.values())).set_index('nodeID')
        graph_data = pd.DataFrame(dict(graph.nodes.items())).T
        graph_data.index = graph_data.index.astype(index_type)

        return merged.merge(graph_data, left_on=node_column, right_index=True, how='left')

    def produce(self, X, graphs):
        for node_column, graph in graphs.items():
            X = self.produce_graph(X, graph, node_column)

        return X
