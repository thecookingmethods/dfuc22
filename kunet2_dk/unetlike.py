import os
from keras import backend as K
from tensorflow import keras
from keras import layers
from custom_metrics import hehe_loss, rdr_metric, siou_metric, r_r_metric, siou_r_pred_large_metric, \
    siou_r_true_large_metric, dist_metric, dist_rr, lr_scheduler


class Unetlike:
    dependencies = {
        'hehe_loss': hehe_loss,
        'siou_metric': siou_metric,
        'rdr_metric': rdr_metric,
        'dist_metric': dist_metric,
        'r_r_metric': r_r_metric,
        'siou_r_true_large_metric': siou_r_true_large_metric,
        'siou_r_pred_large_metric': siou_r_pred_large_metric

    }

    def __init__(self, img_size, model_name, model_save_dir):
        #  set some params
        self._initial_lr = 1e-4

        self._model_file_name = f'{model_name}'
        self._model_save_dir = model_save_dir

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
                            loss=hehe_loss,
                            metrics=[siou_metric, rdr_metric, dist_metric, r_r_metric,
                                     siou_r_pred_large_metric, siou_r_true_large_metric])

    def load(self, model_path):

        model = keras.models.load_model(model_path, custom_objects=Unetlike.dependencies)
        self._model = model
        #K.set_value(model.optimizer.learning_rate, 1000)

    def fit(self, train_gen, val_gen, epochs, training_verbosity,
            max_queue_size, workers, use_multiprocessing,
            initial_epoch):
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                os.path.join(self._model_save_dir, self._model_file_name), save_best_only=True),
            keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
        ]
        history = self._model.fit(train_gen,
                                  epochs=epochs+initial_epoch,
                                  initial_epoch=initial_epoch,
                                  validation_data=val_gen,
                                  callbacks=callbacks,
                                  verbose=training_verbosity,
                                  max_queue_size=max_queue_size,
                                  workers=workers,
                                  use_multiprocessing=use_multiprocessing)
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

        x = layers.Flatten()(x)
        output_c = layers.Dense(2, activation='sigmoid')(x)
        output_r = layers.Dense(1, activation='sigmoid')(x)

        outputs = layers.Concatenate()([output_c, output_r])

        model = keras.Model(inputs, outputs)
        return model
