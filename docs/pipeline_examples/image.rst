Image Pipelines
===============

Here we will be showing some examples using **MLBlocks** to resolve image problems.

Image Classification
--------------------

For the image classification examples we will be using the `USPS Dataset`_, which we will
load using the ``mlblocks.dataset.load_usps`` function.

The data of this dataset is a 3d numpy array vector with shape ``(224, 224, 3)`` containing 9298
224x224 RGB photos of handwritten digits, and the target is a 1d numpy integer array containing
the label of the digit represented in the image.

OpenCV GaussianBlur + Scikit-image HOG + Scikit-Learn RandomForestClassifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this first example, we will attempt to resolve the problem using some basic preprocessing
with the `OpenCV GaussianBlur function`_, to later on calculate the Histogram of Oriented
Gradients using the corresponding `scikit-image function`_ to later on use a simple
`RandomForestClassifier from scikit-learn`_ on the generated features.

.. code-block:: python

    from mlblocks import MLPipeline
    from mlblocks.datasets import load_usps

    dataset = load_usps()
    dataset.describe()

    X_train, X_test, y_train, y_test = dataset.get_splits(1)

    primitives = [
        'cv2.GaussianBlur',
        'skimage.feature.hog',
        'sklearn.ensemble.RandomForestClassifier'
    ]
    init_params = {
        'skimage.feature.hog': {
            'multichannel': True,
            'visualize': False
        }
    }
    pipeline = MLPipeline(primitives, init_params)

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    dataset.score(y_test, predictions)


OpenCV GaussianBlur + Keras Single Layer CNN
--------------------------------------------

In this example, we will preprocess the images using the `OpenCV GaussianBlur function`_
and directly after go into a Single Layer CNN Classifier built on Keras using the corresponding
`MLPrimitives primitive`_.

.. code-block:: python

    from mlblocks import MLPipeline
    from mlblocks.datasets import load_usps

    dataset = load_usps()
    dataset.describe()

    X_train, X_test, y_train, y_test = dataset.get_splits(1)

    primitives = [
        'cv2.GaussianBlur',
        'keras.Sequential.SingleLayerCNNImageClassifier'
    ]
    init_params = {
        'keras.Sequential.SingleLayerCNNImageClassifier': {
            'dense_units': 11,
            'epochs': 5
        }
    }
    pipeline = MLPipeline(primitives, init_params)

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    dataset.score(y_test, predictions)


Image Regression
----------------

For the image regression examples we will be using the Handgeometry Dataset, which we will
load using the ``mlblocks.dataset.load_handgeometry`` function.

The data of this dataset is a 3d numpy array vector with shape ``(224, 224, 3)`` containing 112
224x224 RGB photos of hands, and the target is a 1d numpy float array containing the width of
the wrist in centimeters.

Keras MobileNet + XGBRegressor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we will introduce the usage of the `Pretrained Networks from Keras`_.
In particular, we will be using the `MobileNet`_ for feature extraction, and pass its features
to an `XGBRegressor`_ primitive.

.. code-block:: python

    from mlblocks import MLPipeline
    from mlblocks.datasets import load_handgeometry

    dataset = load_handgeometry()
    dataset.describe()

    X_train, X_test, y_train, y_test = dataset.get_splits(1)

    primitives = [
        'keras.applications.mobilenet.preprocess_input',
        'keras.applications.mobilenet.MobileNet',
        'xgboost.XGBRegressor'
    ]
    init_params = {
        'xgboost.XGBRegressor': {
            'n_estimators': 300,
            'learning_rate': 0.1
        }
    }
    pipeline = MLPipeline(primitives, init_params)

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    dataset.score(y_test, predictions)


.. _USPS Dataset: https://ieeexplore.ieee.org/document/291440/
.. _OpenCV GaussianBlur function: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=gaussianblur#gaussianblur
.. _MLPrimitives primitive: https://github.com/HDI-Project/MLPrimitives/blob/master/mlblocks_primitives/keras.Sequential.SingleLayerCNNImageClassifier.json
.. _scikit-image function: http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog
.. _RandomForestClassifier from scikit-learn: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
.. _Pretrained Networks from Keras: https://keras.io/applications/
.. _MobileNet: https://keras.io/applications/#mobilenet
.. _XGBRegressor: https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
