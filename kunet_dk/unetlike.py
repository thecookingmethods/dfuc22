from tensorflow import keras
from keras import layers
from kunet_dk.custom_metrics import dice_coef, plus_jaccard_distance_loss, dice


class Unetlike:
    dependencies = {
        'dice': dice,
        'dice_coef': dice_coef,
        'jaccard_sth': plus_jaccard_distance_loss(keras.losses.BinaryCrossentropy())
    }

    def __init__(self, img_size, model_name):
        #  set some params
        self._initial_lr = 1e-4

        self._model_file_name = f'{model_name}.h5'

        #  create model
        self._model = self._create_model(img_size)
        self._compile()

    @property
    def model(self):
        return self._model

    def summary(self):
        net_arch = []
        self._model.summary(print_fn=lambda x: net_arch.append(x))
        summary = "\n".join(net_arch)
        return summary

    def _compile(self):
        optimizer = keras.optimizers.Adam(learning_rate=self._initial_lr)
        self._model.compile(optimizer=optimizer,
                            loss=plus_jaccard_distance_loss(keras.losses.BinaryCrossentropy()),
                            metrics=['accuracy', dice_coef, dice])

    def load(self, model_path):

        model = keras.models.load_model(model_path, custom_objects=Unetlike.dependencies)
        self._model = model

    def fit(self, train_gen, val_gen, epochs, training_verbosity,
            max_queue_size, workers, use_multiprocessing,
            initial_epoch):
        callbacks = [
            keras.callbacks.ModelCheckpoint(self._model_file_name, save_best_only=True)
        ]
        history = self._model.fit(train_gen,
                                  epochs=epochs+initial_epoch,
                                  initial_epoch=initial_epoch,
                                  validation_data=val_gen,
                                  callbacks=callbacks,
                                  verbose=training_verbosity,
                                  max_queue_size=max_queue_size,
                                  workers=workers,
                                  use_multiprocessing=use_multiprocessing
                                  )
        return history

    def _create_model(self, img_size):
        inputs = keras.Input(shape=[*img_size])

        x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x

        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.Conv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])
            previous_block_activation = x

        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)
            x = layers.add([x, residual])
            previous_block_activation = x

        outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

        model = keras.Model(inputs, outputs)
        return model