from dm_pipeline.dm_pipeline import DmPipeline
from dm_pipeline.dm_step import DmStep
from dm_pipeline.dm_hyperparam import DmHyperparam, Type

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier

from load_data import d3m_load_data

import copy


class CnnImagePipeline(DmPipeline):
    """
    CNN image pipeline
    """

    def __init__(self,
                 img_dims=(28, 28),
                 epochs=5,
                 batch_size=128,
                 num_classes=10):
        self.img_rows, self.img_cols = img_dims
        self.num_classes = num_classes

        cnn_step = DmStep('CNN',
                          KerasClassifier(
                              build_fn=self.create_model,
                              epochs=epochs,
                              batch_size=batch_size,
                              verbose=0))
        cnn_step.set_tunable_hyperparam(
            DmHyperparam('conv_kernel_dim', Type.INT, [3, 5]))
        cnn_step.set_tunable_hyperparam(
            DmHyperparam('pool_size', Type.INT, [2, 5]))
        cnn_step.set_tunable_hyperparam(
            DmHyperparam('dropout_percent', Type.FLOAT, [0, 0.75]))

        super(CnnImagePipeline, self).__init__([cnn_step])

    def create_model(self, conv_kernel_dim, pool_size, dropout_percent):
        # Make model
        model = Sequential()
        model.add(
            Conv2D(
                32,
                kernel_size=conv_kernel_dim,
                activation='relu',
                input_shape=(self.img_rows, self.img_cols, 1)))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Flatten())
        model.add(Dropout(dropout_percent))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])
        return model

    def preprocess_data(self, data):
        # Data processing
        data_copy = copy.deepcopy(data)

        data_copy.X_train = data_copy.X_train.astype('float32')
        data_copy.X_val = data_copy.X_val.astype('float32')
        data_copy.X_train /= 255
        data_copy.X_val /= 255

        # convert class vectors to binary class matrices
        data_copy.y_train = keras.utils.to_categorical(data_copy.y_train,
                                                       self.num_classes)
        data_copy.y_val = keras.utils.to_categorical(data_copy.y_val,
                                                     self.num_classes)

        # Reshape images
        data_copy.X_train = data_copy.X_train.reshape(
            data_copy.X_train.shape[0], self.img_rows, self.img_cols, 1)
        data_copy.X_val = data_copy.X_val.reshape(
            data_copy.X_val.shape[0], self.img_rows, self.img_cols, 1)
        return data_copy


if __name__ == "__main__":
    # Manual validation that our pipeline object is as we expect.
    image_pipeline = CnnImagePipeline(epochs=5)
    tunable_hyperparams = image_pipeline.get_tunable_hyperparams()

    # Check that the hyperparameters are correct.
    for hyperparam in tunable_hyperparams:
        print(hyperparam)

    # Check that the sklearn pipeline is correct.
    print
    print(image_pipeline.model.get_params())

    # Check that we can score properly.
    image_data_directory = '../../../data/MNIST_D3M'
    image_data = image_pipeline.preprocess_data(
        d3m_load_data(
            data_directory=image_data_directory,
            is_d3m=True,
            sample_size_pct=0.1))

    image_pipeline.fit(image_data.X_train, image_data.y_train)
    print
    print 'score:', image_pipeline.score(image_data.X_val, image_data.y_val)
