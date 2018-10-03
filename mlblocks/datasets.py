import io
import os
import tarfile
import urllib
from functools import wraps

import networkx as nx
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array, load_img
from sklearn import datasets
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, r2_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

INPUT_SHAPE = [224, 224, 3]

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    'data'
)
DATA_URL = 'http://dai-mlblocks.s3.amazonaws.com/{}.tar.gz'


def _add_info(function):

    description = []
    for line in function.__doc__.splitlines():
        if line.startswith('    '):
            line = line[4:]

        description.append(line)

    description = '\n'.join(description)

    @wraps(function)
    def wrapper(*args, **kwargs):
        return function(description, *args, **kwargs)

    return wrapper


class Dataset(object):
    def __init__(self, description, X, y, score, splitter=KFold, shuffle=True, **kwargs):
        self.name = description.splitlines()[0]
        self.description = description

        self.data = X
        self.target = y

        self._splitter = splitter
        self._shuffle = shuffle
        self.score = score

        self.__dict__.update(kwargs)

    def __repr__(self):
        return self.name

    def describe(self):
        print(self.description)

    @staticmethod
    def _get_split(data, index):
        if hasattr(data, 'iloc'):
            return data.iloc[index]
        else:
            return data[index]

    def get_splits(self, n_splits=1, splitter=None):
        if n_splits == 1:
            return train_test_split(self.data, self.target)

        else:
            splitter = splitter or self._splitter
            splitter = splitter(n_splits=n_splits, shuffle=self._shuffle)

            splits = list()
            for train, test in splitter.split(self.data, self.target):
                X_train = self._get_split(self.data, train)
                y_train = self._get_split(self.target, train)
                X_test = self._get_split(self.data, test)
                y_test = self._get_split(self.target, test)
                splits.append((X_train, X_test, y_train, y_test))

            return splits


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


@_add_info
def load_usps(description):
    """USPs Digits Dataset.

    The data of this dataset is a 3d numpy array vector with shape (224, 224, 3)
    containing 9298 224x224 RGB photos of handwritten digits, and the target is
    a 1d numpy integer array containing the label of the digit represented in
    the image.
    """
    dataset_path = _load('usps')

    df = pd.read_csv(os.path.join(dataset_path, 'data.csv'))
    X = _load_images(os.path.join(dataset_path, 'images'), df.image)
    y = df.label.values

    return Dataset(description, X, y, accuracy_score, StratifiedKFold)


@_add_info
def load_handgeometry(description):
    """Hand Geometry Dataset.

    The data of this dataset is a 3d numpy array vector with shape (224, 224, 3)
    containing 112 224x224 RGB photos of hands, and the target is a 1d numpy
    float array containing the width of the wrist in centimeters.
    """
    dataset_path = _load('handgeometry')

    df = pd.read_csv(os.path.join(dataset_path, 'data.csv'))
    X = _load_images(os.path.join(dataset_path, 'images'), df.image)
    y = df.target.values

    return Dataset(description, X, y, r2_score)


@_add_info
def load_personae(description):
    """Personae Dataset.

    The data of this dataset is a 2d numpy array vector containing 145 entries
    that include texts written by Dutch users in Twitter, with some additional
    information about the author, and the target is a 1d numpy binary integer
    array indicating whether the author was extrovert or not.
    """
    dataset_path = _load('personae')

    X = pd.read_csv(os.path.join(dataset_path, 'data.csv'))
    y = X.pop('label').values

    return Dataset(description, X, y, accuracy_score, StratifiedKFold)


@_add_info
def load_umls(description):
    """UMLs Dataset.

    The data consists of information about a 135 Graph and the relations between
    their nodes given as a DataFrame with three columns, source, target and type,
    indicating which nodes are related and with which type of link. The target is
    a 1d numpy binary integer array indicating whether the indicated link exists
    or not.
    """
    dataset_path = _load('umls')

    X = pd.read_csv(os.path.join(dataset_path, 'data.csv'))
    y = X.pop('label').values

    graph = nx.Graph(nx.read_gml(os.path.join(dataset_path, 'graph.gml')))

    return Dataset(description, X, y, accuracy_score, StratifiedKFold, graph=graph)


@_add_info
def load_dic28(description):
    """DIC28 Dataset from Pajek.

    This network represents connections among English words in a dictionary.
    It was generated from Knuth's dictionary. Two words are connected by an
    edge if we can reach one from the other by
    - changing a single character (e. g., work - word)
    - adding / removing a single character (e. g., ever - fever).

    There exist 52,652 words (vertices in a network) having 2 up to 8 characters
    in the dictionary. The obtained network has 89038 edges.
    """

    dataset_path = _load('dic28')

    X = pd.read_csv(os.path.join(dataset_path, 'data.csv'))
    y = X.pop('label').values

    graph1 = nx.Graph(nx.read_gml(os.path.join(dataset_path, 'graph1.gml')))
    graph2 = nx.Graph(nx.read_gml(os.path.join(dataset_path, 'graph2.gml')))

    graph = graph1.copy()
    graph.add_nodes_from(graph2.nodes(data=True))
    graph.add_edges_from(graph2.edges)
    graph.add_edges_from(X[['graph1', 'graph2']].values)

    graphs = {
        'graph1': graph1,
        'graph2': graph2,
    }

    return Dataset(description, X, y, accuracy_score, StratifiedKFold, graph=graph, graphs=graphs)


@_add_info
def load_nomination(description):
    """Sample 1 of graph vertex nomination data from MII Lincoln Lab.

    Data consists of one graph whose nodes contain two attributes, attr1 and attr2.
    Associated with each node is a label that has to be learned and predicted.
    """

    dataset_path = _load('nomination')

    X = pd.read_csv(os.path.join(dataset_path, 'data.csv'))
    y = X.pop('label').values

    graph = nx.Graph(nx.read_gml(os.path.join(dataset_path, 'graph.gml')))

    return Dataset(description, X, y, accuracy_score, StratifiedKFold, graph=graph)


@_add_info
def load_amazon(description):
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

    return Dataset(description, X, y, normalized_mutual_info_score, graph=graph)


@_add_info
def load_jester(description):
    """Ratings from the Jester Online Joke Recommender System.

    This dataset consists of over 1.7 million instances of (user_id, item_id, rating)
    triples, which is split 50-50 into train and test data.

    source: "University of California Berkeley, CA"
    sourceURI: "http://eigentaste.berkeley.edu/dataset/"
    """

    dataset_path = _load('jester')

    X = pd.read_csv(os.path.join(dataset_path, 'data.csv'))
    y = X.pop('rating').values

    return Dataset(description, X, y, r2_score)


@_add_info
def load_newsgroups(description):
    """20 News Groups Dataset.

    The data of this dataset is a 1d numpy array vector containing the texts
    from 11314 newsgroups posts, and the target is a 1d numpy integer array
    containing the label of one of the 20 topics that they are about.
    """
    dataset = datasets.fetch_20newsgroups()
    return Dataset(description, dataset.data, dataset.target, accuracy_score, StratifiedKFold)


@_add_info
def load_iris(description):
    """Iris Dataset."""
    dataset = datasets.load_iris()
    return Dataset(description, dataset.data, dataset.target, accuracy_score, StratifiedKFold)


@_add_info
def load_boston(description):
    """Boston House Prices Dataset."""
    dataset = datasets.load_boston()
    return Dataset(description, dataset.data, dataset.target, r2_score)


LOADERS = {
    'graph/community_detection': load_amazon,
    'graph/graph_matching': load_dic28,
    'graph/link_prediction': load_umls,
    'graph/vertex_nomination': load_nomination,
    'image/classification': load_usps,
    'image/regression': load_handgeometry,
    'tabular/classification': load_iris,
    'tabular/collaborative_filtering': load_jester,
    'tabular/regression': load_boston,
    'text/classification': load_personae,
}


def load_dataset(data_modality, task_type):
    problem_type = data_modality + '/' + task_type
    loader = LOADERS.get(problem_type)

    if not loader:
        raise ValueError('Unknown problem type: {}'.format(problem_type))

    return loader()
