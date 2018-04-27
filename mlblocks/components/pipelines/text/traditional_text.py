from mlblocks.ml_pipeline.ml_pipeline import MLPipeline


class TraditionalTextPipeline(MLPipeline):
    """
    Traditional text pipeline.
    """

    def __new__(cls, *args, **kwargs):
        return MLPipeline.from_dm_json(['count_vectorizer', 'to_array', 'tfidf_transformer', 'multinomial_nb'])
