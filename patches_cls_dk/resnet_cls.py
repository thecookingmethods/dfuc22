from tensorflow import keras
from keras import layers
from custom_metrics import tp, fp, tn, fn


class ResnetCls:
    dependencies = {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
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
                            loss=keras.losses.BinaryCrossentropy(),
                            metrics=['accuracy', tp, fp, tn, fn])

    def load(self, model_path):
        model = keras.models.load_model(model_path, custom_objects=ResnetCls.dependencies)
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

    def _create_model(self, img_size, start_num_filters=64, num_blocks_list=None, bottleneck=False):
        if num_blocks_list is None:
            num_blocks_list = [3, 4, 6, 3]
        inputs = keras.Input([*img_size])

        t = layers.BatchNormalization()(inputs)
        t = layers.Conv2D(kernel_size=5,
                          strides=1,
                          filters=start_num_filters,
                          padding="same")(t)
        t = ResnetCls._relu_bn(t)

        for i, num_blocks in enumerate(num_blocks_list):
            for j in range(num_blocks):
                if not bottleneck:
                    t = ResnetCls._residual_block(t, downsample=(j == 0 and i != 0), filters=start_num_filters)
                else:
                    t = ResnetCls._bottleneck_block(t, downsample=(j == 0 and i != 0), filters=start_num_filters)
            start_num_filters *= 2

        t = layers.AveragePooling2D(8)(t)
        t = layers.Flatten()(t)
        outputs = layers.Dense(1, activation='sigmoid')(t)

        model = keras.Model(inputs, outputs)

        return model

    @staticmethod
    def _relu_bn(inputs):
        relu = layers.ReLU()(inputs)
        bn = layers.BatchNormalization()(relu)
        return bn

    @staticmethod
    def _residual_block(x, downsample, filters, kernel_size=3):
        y = layers.Conv2D(kernel_size=kernel_size,
                          strides=(1 if not downsample else 2),
                          filters=filters,
                          padding="same")(x)
        y = ResnetCls._relu_bn(y)
        y = layers.Conv2D(kernel_size=kernel_size,
                          strides=1,
                          filters=filters,
                          padding="same")(y)

        if downsample:
            x = layers.Conv2D(kernel_size=1,
                              strides=2,
                              filters=filters,
                              padding="same")(x)
        out = layers.Add()([x, y])
        out = ResnetCls._relu_bn(out)
        return out

    @staticmethod
    def _bottleneck_block(x, downsample, filters, kernel_size=3):
        y = layers.Conv2D(kernel_size=1,
                          strides=(1 if not downsample else 2),
                          filters=filters,
                          padding="same")(x)
        y = ResnetCls._relu_bn(y)
        y = layers.Conv2D(kernel_size=kernel_size,
                          strides=1,
                          filters=filters,
                          padding="same")(y)
        y = ResnetCls._relu_bn(y)
        y = layers.Conv2D(kernel_size=1,
                          strides=1,
                          filters=filters,
                          padding="same")(y)

        if downsample:
            x = layers.Conv2D(kernel_size=1,
                              strides=2,
                              filters=filters,
                              padding="same")(x)
        out = layers.Add()([x, y])
        out = ResnetCls._relu_bn(out)
        return out