from mlblocks.ml_pipeline.ml_pipeline import MLPipeline


class TraditionalTextPipeline(MLPipeline):
    """Traditional text pipeline."""

    BLOCKS = ['count_vectorizer', 'to_array', 'tfidf_transformer', 'multinomial_nb']
