import io
import os
import tarfile
import urllib

import networkx as nx
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array, load_img
from sklearn import datasets
from sklearn.metrics import accuracy_score, r2_score
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

    df = pd.read_csv(os.path.join(dataset_path, 'data.csv'))
    y = df.pop('label')
    X = df

    return Dataset('personae', X, y, accuracy_score)


def load_umls():
    dataset_path = _load('umls')
    df = pd.read_csv(os.path.join(dataset_path, 'data.csv'))

    y = df.pop('label').values
    X = df
    node_columns = ['source', 'target']
    graph = nx.Graph(nx.read_gml(os.path.join(dataset_path, 'graph.gml')))

    return Dataset('umls', X, y, accuracy_score, graph=graph, node_columns=node_columns)


def load_newsgroups():
    dataset = datasets.fetch_20newsgroups()
    return Dataset('newsgroups', dataset.data, dataset.target, accuracy_score)


def load_iris():
    dataset = datasets.load_iris()
    return Dataset('iris', dataset.data, dataset.target, accuracy_score)


def load_boston():
    dataset = datasets.load_boston()
    return Dataset('iris', dataset.data, dataset.target, r2_score)
