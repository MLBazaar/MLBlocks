from mlblocks.ml_pipeline.ml_pipeline import MLPipeline


class TraditionalImagePipeline(MLPipeline):
    """Traditional image pipeline using HOG features."""

    BLOCKS = ['HOG', 'random_forest_classifier']
