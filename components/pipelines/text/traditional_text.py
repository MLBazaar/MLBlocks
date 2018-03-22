"""
Traditional text pipeline
"""
from dm_pipeline.dm_pipeline import DmPipeline
from dm_pipeline.dm_step import DmStep
from dm_pipeline.dm_hyperparam import DmHyperparam, Type

# Import functions
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import FunctionTransformer

from components.functions.text.count_vectorizer import CustomCountVectorizer

from load_data import d3m_load_data

import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '....'))


class TraditionalTextPipeline(DmPipeline):
    """
    Traditional image pipeline using HOG features.
    """

    def __init__(self):

        # Auxiliary function to move to array for text pipeline
        def hist_to_array(hist):
            return hist.toarray()

        # Define pipeline variables
        vectorizer_step = DmStep('CountVectorizer', CustomCountVectorizer())
        vectorizer_step.set_tunable_hyperparam(
            DmHyperparam('max_features', Type.INT, [1000, 40000]))
        vectorizer_step.set_tunable_hyperparam(
            DmHyperparam('max_ngram', Type.INT, [1, 4]))
        vectorizer_step.set_tunable_hyperparam(
            DmHyperparam('max_df', Type.FLOAT, [0.99, 1.]))
        vectorizer_step.set_tunable_hyperparam(
            DmHyperparam('min_df', Type.FLOAT, [0., 0.01]))
        to_array_step = DmStep(
            'HistToArray',
            FunctionTransformer(func=hist_to_array,
                                validate=False))  # custom pipeline component

        tfidf_step = DmStep('TFIDF',
                            TfidfTransformer(
                                smooth_idf=True, sublinear_tf=False))
        tfidf_step.set_tunable_hyperparam(
            DmHyperparam(
                'norm', Type.STRING, ['l1', 'l2'], is_categorical=True))
        tfidf_step.set_tunable_hyperparam(
            DmHyperparam(
                'use_idf', Type.BOOL, [True, False], is_categorical=True))
        multinomial_nb_step = DmStep('MNB', MultinomialNB())
        multinomial_nb_step.set_tunable_hyperparam(
            DmHyperparam('alpha', Type.FLOAT, [0.01, 1.]))

        super(TraditionalTextPipeline, self).__init__(
            [vectorizer_step, to_array_step, tfidf_step, multinomial_nb_step])


if __name__ == "__main__":
    # Manual validation that our pipeline object is as we expect.
    text_pipeline = TraditionalTextPipeline()
    tunable_hyperparams = text_pipeline.get_tunable_hyperparams()

    # Check that the hyperparameters are correct.
    for hyperparam in tunable_hyperparams:
        print(hyperparam)

    # Check that the sklearn pipeline is correct.
    print
    print(text_pipeline.model)

    # Check that we can score properly.
    text_data_directory = '../../../data/BagsOfPopcorn_D3M'
    text_data = d3m_load_data(
        data_directory=text_data_directory, is_d3m=True, sample_size_pct=0.1)

    text_pipeline.fit(text_data.X_train, text_data.y_train)
    print
    print 'score:', text_pipeline.score(text_data.X_val, text_data.y_val)
