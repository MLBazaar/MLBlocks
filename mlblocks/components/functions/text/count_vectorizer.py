import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class CustomCountVectorizer(CountVectorizer):
    #defaults from sklearn
    def __init__(self, max_features=None, max_df=1, min_df=1, max_ngram=1):
        super(CustomCountVectorizer, self).__init__(
            max_features=max_features,
            max_df=max_df,
            min_df=min_df,
            ngram_range=(1, max_ngram),
            analyzer="word",
            tokenizer=None,
            preprocessor=None,
            stop_words=None)
