from unittest import TestCase

from mlblocks import MLPipeline


class TestMLPipeline(TestCase):

    def test_dict(self):
        pipeline_dict = {
            'primitives': [
                'sklearn.ensemble.RandomForestClassifier'
            ],
            'init_params': {
                'sklearn.ensemble.RandomForest#1': {
                    'n_estimators': 500
                }
            },
            'input_names': {
                'sklearn.ensemble.RandomForest#1': {
                    'X': 'X1'
                }
            },
            'output_names': {
                'sklearn.ensemble.RandomForest#1': {
                    'y': 'y1'
                }
            }
        }

        pipeline = MLPipeline(pipeline_dict)

        assert pipeline.primitives == ['sklearn.ensemble.RandomForestClassifier']
        assert pipeline.init_params == {
            'sklearn.ensemble.RandomForest#1': {
                'n_estimators': 500
            }
        }
        assert pipeline.input_names == {
            'sklearn.ensemble.RandomForest#1': {
                'X': 'X1'
            }
        }
        assert pipeline.output_names == {
            'sklearn.ensemble.RandomForest#1': {
                'y': 'y1'
            }
        }

    def test_list(self):
        primitives = [
            'sklearn.ensemble.RandomForestClassifier'
        ]
        init_params = {
            'sklearn.ensemble.RandomForest#1': {
                'n_estimators': 500
            }
        }

        pipeline = MLPipeline(primitives, init_params=init_params)

        assert pipeline.primitives == ['sklearn.ensemble.RandomForestClassifier']
        assert pipeline.init_params == {
            'sklearn.ensemble.RandomForest#1': {
                'n_estimators': 500
            }
        }

    def test_none(self):
        primitives = [
            'sklearn.ensemble.RandomForestClassifier'
        ]
        init_params = {
            'sklearn.ensemble.RandomForest#1': {
                'n_estimators': 500
            }
        }

        pipeline = MLPipeline(primitives=primitives, init_params=init_params)

        assert pipeline.primitives == ['sklearn.ensemble.RandomForestClassifier']
        assert pipeline.init_params == {
            'sklearn.ensemble.RandomForest#1': {
                'n_estimators': 500
            }
        }

    def test_mlpipeline(self):
        primitives = [
            'sklearn.ensemble.RandomForestClassifier'
        ]
        init_params = {
            'sklearn.ensemble.RandomForest#1': {
                'n_estimators': 500
            }
        }

        pipeline = MLPipeline(primitives=primitives, init_params=init_params)
        pipeline2 = MLPipeline(pipeline)

        assert pipeline2.primitives == ['sklearn.ensemble.RandomForestClassifier']
        assert pipeline2.init_params == {
            'sklearn.ensemble.RandomForest#1': {
                'n_estimators': 500
            }
        }
