# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

from mlblocks import datasets


class TestDataset(TestCase):

    def setUp(self):
        self.description = """Dataset Name.

        Some extended description.
        """
        self.score = Mock()
        self.score.return_value = 1.0

        self.dataset = datasets.Dataset(
            self.description, 'data', 'target', self.score,
            shuffle=False, stratify=True, some='kwargs')

    def test___init__(self):

        assert self.dataset.name == 'Dataset Name.'
        assert self.dataset.description == self.description
        assert self.dataset.data == 'data'
        assert self.dataset.target == 'target'
        assert self.dataset._shuffle is False
        assert self.dataset._stratify is True
        assert self.dataset._score == self.score
        assert self.dataset.some == 'kwargs'

    def test_score(self):
        returned = self.dataset.score('a', b='c')

        assert returned == 1.0
        self.score.assert_called_once_with('a', b='c')

    def test___repr__(self):
        repr_ = str(self.dataset)

        assert repr_ == "Dataset Name."


def test_dataset_describe(capsys):
    """Tested here because fixtures are not supported in TestCases."""

    description = """Dataset Name.

    Some extended description.
    """

    dataset = datasets.Dataset(description, 'data', 'target', 'score')
    dataset.describe()

    captured = capsys.readouterr()
    assert captured.out == description + '\n'
