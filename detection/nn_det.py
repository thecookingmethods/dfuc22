#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

from tensorflow import keras
from keras import layers
from custom_metrics import tp, fp, tn, fn


class NNDet:
    dependencies = {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
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
                            loss=keras.losses.BinaryCrossentropy(),
                            metrics=['accuracy', tp, fp, tn, fn])

    def load(self, model_path):
        model = keras.models.load_model(model_path, custom_objects=NNDet.dependencies)
        self._model = model

    def fit(self, train_gen, val_gen, epochs, training_verbosity,
            max_queue_size, workers, use_multiprocessing,
            initial_epoch):
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                os.path.join(self._model_save_dir, self._model_file_name), save_best_only=True)
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
        return None