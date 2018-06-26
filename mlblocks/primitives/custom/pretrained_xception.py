from keras.applications.xception import preprocess_input, Xception
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.optimizers import SGD, RMSprop
import numpy as np

OPTIMIZERS = {
    'sgd': SGD,
    'rmsprop': RMSprop
}


class PretrainedXceptionClassifier(object):
    """Use pretrained image CNN (xception) to do "fine-tuning" classificatino.

    image width/height needs to be >71, and num channels must equal 3
    """

    def __init__(self,
                 input_shape,
                 classes=1000,
                 start_optimizer='rmsprop',
                 optimizer='sgd',
                 start_learning_rate=1e-1,
                 learning_rate=1e-3,
                 loss=None,
                 metrics=None,
                 start_training_layer=115):
        self.input_shape = input_shape
        self.classes = classes
        self.start_optimizer = start_optimizer
        self.start_learning_rate = start_learning_rate
        if self.start_learning_rate is not None:
            self.start_optimizer = OPTIMIZERS[self.start_optimizer](
                lr=self.start_learning_rate)

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        if self.learning_rate is not None:
            self.optimizer = OPTIMIZERS[self.optimizer](lr=self.learning_rate)

        self.loss = loss
        self.metrics = metrics
        self.start_training_layer = start_training_layer

        self.base_model = Xception(weights='imagenet',
                                   pooling='avg',
                                   include_top=False)
        x = self.base_model.output
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.classes, activation='sigmoid')(x)

        self.model = Model(inputs=self.base_model.input, outputs=predictions)

    def preprocess_data(self, X):
        X = preprocess_input(X)
        return X

    def fit(self, X, y=None, epochs=10, start_epochs=5,
            batch_size=16):
        X = self.preprocess_data(X)

        for layer in self.base_model.layers:
            layer.trainable = False

        self.model.compile(self.start_optimizer,
                           loss=self.loss,
                           metrics=self.metrics)
        self.model.fit(X, y,
                       epochs=start_epochs,
                       batch_size=batch_size)
                       # validation_data=(validation_data, validation_labels))

        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
        for layer in self.model.layers[:self.start_training_layer]:
            layer.trainable = False
        for layer in self.model.layers[self.start_training_layer:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss)
        self.model.fit(X, y,
                       epochs=epochs,
                       batch_size=batch_size)
                       # validation_data=(validation_data, validation_labels))

    def produce(self, X):
        X = self.preprocess_data(X)
        preds = self.model.predict(X)
        return np.argmax(preds, axis=1)
