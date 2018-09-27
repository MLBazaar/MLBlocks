import io
import os
import tarfile
import urllib

import networkx as nx
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array, load_img
from sklearn import datasets
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, r2_score
from sklearn.model_selection import train_test_split

INPUT_SHAPE = [224, 224, 3]

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    'data'
)
DATA_URL = 'http://dai-mlblocks.s3.amazonaws.com/{}.tar.gz'


class Dataset(object):
    def __init__(self, name, X, y, score, **kwargs):
        self.name = name
        self.__dict__.update(kwargs)

        self.data = X
        self.target = y

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        self.train_data = X_train
        self.test_data = X_test
        self.train_target = y_train
        self.test_target = y_test

        self.score = score


def _download(dataset_name, dataset_path):
    url = DATA_URL.format(dataset_name)
    response = urllib.request.urlopen(url)
    bytes_io = io.BytesIO(response.read())

    with tarfile.open(fileobj=bytes_io, mode='r:gz') as tf:
        tf.extractall(DATA_PATH)


def _load(dataset_name):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    dataset_path = os.path.join(DATA_PATH, dataset_name)
    if not os.path.exists(dataset_path):
        _download(dataset_name, dataset_path)

    return dataset_path


def _load_images(image_dir, filenames):
    images = []
    for filename in filenames:
        filename = os.path.join(image_dir, filename)

        image = load_img(filename)
        image = image.resize(tuple(INPUT_SHAPE[0:2]))
        image = img_to_array(image)
        image = image / 255.0  # Quantize images.
        images.append(image)

    return np.array(images)


def load_usps():
    dataset_path = _load('usps')

    df = pd.read_csv(os.path.join(dataset_path, 'data.csv'))
    X = _load_images(os.path.join(dataset_path, 'images'), df.image)
    y = df.label.values

    return Dataset('usps', X, y, accuracy_score)


def load_handgeometry():
    dataset_path = _load('handgeometry')

    df = pd.read_csv(os.path.join(dataset_path, 'data.csv'))
    X = _load_images(os.path.join(dataset_path, 'images'), df.image)
    y = df.target.values

    return Dataset('handgeometry', X, y, r2_score)


def load_personae():
    dataset_path = _load('personae')

    X = pd.read_csv(os.path.join(dataset_path, 'data.csv'))
    y = X.pop('label').values

    return Dataset('personae', X, y, accuracy_score)


def load_umls():
    dataset_path = _load('umls')

    X = pd.read_csv(os.path.join(dataset_path, 'data.csv'))
    y = X.pop('label').values

    node_columns = ['source', 'target']
    graph = nx.Graph(nx.read_gml(os.path.join(dataset_path, 'graph.gml')))

    return Dataset('umls', X, y, accuracy_score, graph=graph, node_columns=node_columns)


def load_dic28():
    """
    "datasetName": "DIC28 from Pajek",
    """

    dataset_path = _load('dic28')

    X = pd.read_csv(os.path.join(dataset_path, 'data.csv'))
    y = X.pop('label').values

    node_columns = ['graph1', 'graph2']
    graphs = {
        'graph1': nx.Graph(nx.read_gml(os.path.join(dataset_path, 'graph1.gml'))),
        'graph2': nx.Graph(nx.read_gml(os.path.join(dataset_path, 'graph2.gml')))
    }

    return Dataset('dic28', X, y, accuracy_score, graphs=graphs, node_columns=node_columns)


def load_nomination():
    """Sample 1 of graph vertex nomination data from MII Lincoln Lab.

    Data consists of one graph whose nodes contain two attributes, attr1 and attr2.
    Associated with each node is a label that has to be learned and predicted.
    """

    dataset_path = _load('nomination')

    X = pd.read_csv(os.path.join(dataset_path, 'data.csv'))
    y = X.pop('label').values

    graphs = {
        'node_id': nx.Graph(nx.read_gml(os.path.join(dataset_path, 'graph.gml')))
    }

    return Dataset('nomination', X, y, accuracy_score, graphs=graphs)


def load_amazon():
    """Amazon product co-purchasing network and ground-truth communities.

    Network was collected by crawling Amazon website. It is based on Customers Who Bought
    This Item Also Bought feature of the Amazon website. If a product i is frequently
    co-purchased with product j, the graph contains an undirected edge from i to j.
    Each product category provided by Amazon defines each ground-truth community.
    """

    dataset_path = _load('amazon')

    X = pd.read_csv(os.path.join(dataset_path, 'data.csv'))
    y = X.pop('label').values

    graph = nx.Graph(nx.read_gml(os.path.join(dataset_path, 'graph.gml')))

    return Dataset('amazon', X, y, normalized_mutual_info_score, graph=graph)


def load_jester():
    """Ratings from the Jester Online Joke Recommender System.

    This dataset consists of over 1.7 million instances of (user_id, item_id, rating)
    triples, which is split 50-50 into train and test data.

    source: "University of California Berkeley, CA"
    sourceURI: "http://eigentaste.berkeley.edu/dataset/"
    """

    dataset_path = _load('jester')

    X = pd.read_csv(os.path.join(dataset_path, 'data.csv'))
    y = X.pop('rating').values

    return Dataset('jester', X, y, r2_score)


def load_newsgroups():
    dataset = datasets.fetch_20newsgroups()
    return Dataset('newsgroups', dataset.data, dataset.target, accuracy_score)


def load_iris():
    dataset = datasets.load_iris()
    return Dataset('iris', dataset.data, dataset.target, accuracy_score)


def load_boston():
    dataset = datasets.load_boston()
    return Dataset('iris', dataset.data, dataset.target, r2_score)


LOADERS = {
    'graph/community_detection': load_amazon,
    'graph/graph_matching': load_dic28,
    'graph/linkPrediction': load_umls,
    'graph/vertex_nomination': load_nomination,
    'image/classification': load_usps,
    'image/regression': load_handgeometry,
    'single_table/classification': load_iris,
    'single_table/collaborative_filtering': load_jester,
    'single_table/regression': load_boston,
    'text/classification': load_personae,
}


def load_dataset(data_modality, task_type):
    problem_type = data_modality + '/' + task_type
    loader = LOADERS.get(problem_type)

    if not loader:
        raise ValueError('Unknown problem type: {}'.format(problem_type))

    return loader()
