import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import SGD, RMSprop
from .learning_utils import CyclicLR, SnapshotCallbackBuilder
import tempfile
from tqdm import tqdm

OPTIMIZERS = {
    'sgd': SGD,
    'rmsprop': RMSprop
}


class PretrainedImageBase(object):
    """Use pretrained image CNN (xception) to do "fine-tuning" classificatino.

    image width/height needs to be >71, and num channels must equal 3
    """

    _base_model_class = None
    _base_model_preprocess_func = None

    def __init__(self,
                 input_shape,
                 train_full_network=True,
                 classes=1000,
                 start_optimizer='rmsprop',
                 optimizer='sgd',
                 start_learning_rate=1e-1,
                 learning_rate=1e-3,
                 cyclical_learning_rate=False,
                 cyclical_base_lr=.001,
                 cyclical_max_lr=.006,
                 cyclical_step_size=2000,
                 cyclical_mode='triangular2',
                 cyclical_gamma=1,
                 cyclical_scale_fn=None,
                 cyclical_scale_mode='cycle',
                 snapshot_ensemble=False,
                 num_snapshots=5,
                 loss=None,
                 metrics=None,
                 start_training_layer=115):
        self.train_full_network = train_full_network
        # TODO: add clr and snapshot hyperparameters to all jsons
        self.input_shape = input_shape
        self.classes = classes
        self.start_optimizer = start_optimizer
        self.start_learning_rate = start_learning_rate
        if self.start_learning_rate is not None:
            self.start_optimizer = OPTIMIZERS[self.start_optimizer](lr=self.start_learning_rate)

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        if self.learning_rate is not None:
            self.optimizer = OPTIMIZERS[self.optimizer](lr=self.learning_rate)

        self.loss = loss
        self.metrics = metrics
        self.start_training_layer = start_training_layer

        self.start_callbacks = []
        self.callbacks = []
        if cyclical_learning_rate:
            clr = CyclicLR(base_lr=cyclical_base_lr, max_lr=cyclical_max_lr,
                           step_size=cyclical_step_size, mode=cyclical_mode,
                           gamma=cyclical_gamma, scale_fn=cyclical_scale_fn,
                           scale_mode=cyclical_scale_mode)
            self.start_callbacks.append(clr)
            self.callbacks.append(clr)
        self.snapshot_ensemble = snapshot_ensemble
        self.num_snapshots = num_snapshots

        self.base_model = self.base_model_class()(weights='imagenet', pooling='avg',
                                                  include_top=False)
        self.preprocess_data = self.base_model_preprocess_func()
        x = self.base_model.output
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        # binary, we just check True/False, don't need 2 classes
        if self.classes == 2:
            self.classes = 1
        predictions = Dense(self.classes, activation='sigmoid')(x)

        self.model = Model(inputs=self.base_model.input, outputs=predictions)

    @classmethod
    def base_model_class(cls):
        return cls._base_model_class

    @classmethod
    def base_model_preprocess_func(cls):
        return cls._base_model_preprocess_func

    def fit(self, X, y=None, epochs=10, start_epochs=5, batch_size=16,
            generator=False, steps_per_epoch=None,
            validation_data=None, validation_steps=None):

        if self.snapshot_ensemble:
            start_snapshot = SnapshotCallbackBuilder(start_epochs, self.num_snapshots, self.start_learning_rate)
            snapshot = SnapshotCallbackBuilder(epochs, self.num_snapshots, self.learning_rate)
            self.start_callbacks.extend(start_snapshot.get_callbacks(model_prefix='Model_'))
            self.callbacks.extend(snapshot.get_callbacks(model_prefix='Model_'))
        if not generator:
            X = self.preprocess_data(X)
            top_model_fit_params = {
                'X': X,
                'y': y,
                'validation_data': validation_data,
                'generator': False,
                'epochs': start_epochs,
                'batch_size': batch_size,
                'callbacks': self.start_callbacks,
            }

            fit_params = {
                'x': X,
                'y': y,
                'epochs': epochs,
                'batch_size': batch_size,
                'callbacks': self.callbacks,
            }
            fit_func = 'fit'
        else:
            top_model_fit_params = {
                'X': X,
                'validation_data': validation_data,
                'generator': True,
                'epochs': start_epochs,
                'batch_size': batch_size,
                'callbacks': self.start_callbacks,
            }
            fit_params = {
                'generator': X,
                'steps_per_epoch': steps_per_epoch,
                'epochs': epochs,
                'validation_data': validation_data,
                'validation_steps': validation_steps,
                'callbacks': self.callbacks,
            }
            fit_func = 'fit_generator'
        weights_file = self._train_topnet(**top_model_fit_params)
        if self.train_full_network:
            self._add_and_load_topnet(weights_file)

            for layer in self.model.layers[:self.start_training_layer]:
                layer.trainable = False
            for layer in self.model.layers[self.start_training_layer:]:
                layer.trainable = True

            self.model.compile(optimizer=self.optimizer, loss=self.loss)
            getattr(self.model, fit_func)(**fit_params)

    def produce(self, X, generator=False):
        if self.train_full_network:
            return self._predict_full_model(X, generator=generator)
        else:
            return self._predict_top_model(X, generator=generator)

    def _train_topnet(self, X, y=None, generator=False,
                      validation_data=None, callbacks=None,
                      epochs=50, batch_size=16):
        if generator:
            bottleneck_features_train = []
            y_train = []
            train_generator = X
            len_gen = len(train_generator)
            for i, x_y in enumerate(tqdm(train_generator,
                                         desc="generating base model train features")):
                x, y = x_y
                features = self.base_model.predict_on_batch(x)
                bottleneck_features_train.append(features)
                y_train.append(y)
                if i == len_gen - 1:
                    break
            train_generator.reset()
            bottleneck_features_train = np.concatenate(bottleneck_features_train, axis=0)
            y_train = np.concatenate(y_train)

            if validation_data is not None:
                bottleneck_features_valid = []
                y_valid = []
                valid_generator = validation_data
                len_gen = len(valid_generator)
                for i, x_y in enumerate(tqdm(valid_generator,
                                             desc="generating base model validation features")):
                    features = self.base_model.predict_on_batch(x)
                    bottleneck_features_valid.append(features)
                    y_valid.append(y)
                    if i == len_gen - 1:
                        break
                valid_generator.reset()
                bottleneck_features_valid = np.concatenate(bottleneck_features_valid, axis=0)
                y_valid = np.concatenate(y_valid)
                validation_data = (bottleneck_features_valid, y_valid)
        else:
            bottleneck_features_train = self.base_model.predict(X)
            y_train = y
            if validation_data is not None:
                validation_data = (self.base_model.predict(validation_data[0]), validation_data[1])

        self.top_model = self._build_top_model()

        self.top_model.compile(optimizer=self.start_optimizer, loss=self.loss)

        self.top_model.fit(bottleneck_features_train,
                           y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           callbacks=callbacks,
                           validation_data=validation_data)
        _, save_weights_file = tempfile.mkstemp()
        self.top_model.save_weights(save_weights_file)
        return save_weights_file

    def _build_top_model(self):
        top_model = Sequential()
        top_model.add(Dense(256, input_shape=self.base_model.output_shape[1:], activation='relu'))
        top_model.add(Dropout(0.5))
        # binary, we just check True/False, don't need 2 classes
        if self.classes == 2:
            self.classes = 1
        top_model.add(Dense(self.classes, activation='sigmoid'))
        return top_model

    def _add_and_load_topnet(self, top_model_weights_path):
        # TODO: do we need to save/load weights, or can we just
        # use reference to self.top_model?
        top_model = self._build_top_model()
        top_model.load_weights(top_model_weights_path)
        self.model = Model(inputs=self.base_model.input,
                           outputs=top_model(self.base_model.output))

    def _predict_top_model(self, X, generator=False):
        if generator:
            predictions = []
            len_gen = len(X)
            for i, x_y in enumerate(tqdm(X, desc="generating prediction inputs")):
                if isinstance(x_y, tuple):
                    x = x_y[0]
                else:
                    x = x_y
                features = self.base_model.predict_on_batch(x)
                preds = self.top_model.predict_on_batch(features)
                predictions.append(preds)
                if i == len_gen - 1:
                    break
            preds = np.concatenate(predictions, axis=0)
        else:
            X = self.preprocess_data(X)
            features = self.base_model.predict(X)
            preds = self.top_model.predict(features)
        if self.classes > 1:
            return np.argmax(preds, axis=1)
        return preds[:, 0]

    def _predict_full_model(self, X, generator=False):
        if generator:
            preds = self.model.predict_generator(X)
        else:
            X = self.preprocess_data(X)
            preds = self.model.predict(X)
        if self.classes > 1:
            return np.argmax(preds, axis=1)
        return preds[:, 0]



