import io
import os

import pandas as pd
from sklearn.metrics import accuracy_score
from mlprimitives.datasets import Dataset

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    'data'
)

DATA_URL = 'http://mlblocks.s3.amazonaws.com/{}.csv'

def _download(dataset_name, dataset_path):
    url = DATA_URL.format(dataset_name)

    data = pd.read_csv(url)
    data.to_csv(dataset_path, index=False)

def _load(dataset_name):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    dataset_path = os.path.join(DATA_PATH, dataset_name + '.csv')
    if not os.path.exists(dataset_path):
        _download(dataset_name, dataset_path)

    return dataset_path

def load_census():
    """Adult Census dataset.

    Predict whether income exceeds $50K/yr based on census data. Also known as "Adult" dataset.

    Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean
    records was extracted using the following conditions: ((AAGE>16) && (AGI>100) &&
    (AFNLWGT>1)&& (HRSWK>0))

    Prediction task is to determine whether a person makes over 50K a year.

    source: "UCI
    sourceURI: "https://archive.ics.uci.edu/ml/datasets/census+income"
    """

    dataset_path = _load('census_train')

    X = pd.read_csv(dataset_path)
    y = X.pop('label').values

    return Dataset(load_census.__doc__, X, y, accuracy_score, 'single_table',
                   'classification', 'binary', stratify=True)