#!/usr/bin/python
# -*- coding: utf-8 -*-

import random

import numpy as np
import tensorflow as tf


class HardExampleMiner(tf.keras.utils.Sequence):
	def __init__(self, model, input_shape, batch_size, data_generator, ratio=0.8, hem_every_n_epochs=1):
		self._number_of_batches = len(data_generator)

		self.x = np.zeros([batch_size*self._number_of_batches, *input_shape], dtype=np.float32)
		self.y = np.zeros([batch_size*self._number_of_batches, input_shape[0], input_shape[1], 1], dtype=np.float32)
		self._input_shape = input_shape
		self._batch_size = batch_size

		self._hard_examples_idxs = np.arange(batch_size * self._number_of_batches)
		self._errors = np.empty(batch_size*self._number_of_batches)

		self._number_of_batches_for_training = int(self._number_of_batches * ratio)
		self._hard_examples_idxs_for_training = []

		self._model = model
		self._data_gen = data_generator

		self._epoch_counter = 0
		self._hem_every_n_epochs = hem_every_n_epochs

		self.on_epoch_end()

	def __len__(self):
		return self._number_of_batches_for_training

	def __getitem__(self, batch_id):
		start = self._batch_size * batch_id
		end = self._batch_size * (batch_id + 1)

		samples_x = np.zeros([self._batch_size, *self._input_shape], dtype=np.float32)
		samples_y = np.zeros([self._batch_size, self._input_shape[0], self._input_shape[1], 1], dtype=np.float32)
		for seq, idx in enumerate(self._hard_examples_idxs_for_training[start:end]):
			samples_x[seq, ] = self.x[idx]
			samples_y[seq, ] = self.y[idx]
		return samples_x, samples_y

	def on_epoch_end(self):
		if self._epoch_counter % (self._hem_every_n_epochs*2) == 0:
			print('resetting ohem examples...')
			self._hard_examples_idxs_for_training = []
			for batch_id in range(self._number_of_batches):
				samples_x, samples_y = self._data_gen[batch_id]

				from_id = batch_id * self._batch_size
				to_id = (batch_id + 1) * self._batch_size

				self.x[from_id:to_id] = samples_x
				self.y[from_id:to_id] = samples_y

				outputs = self._model.model.predict_on_batch(samples_x)
				losses = self._jaccard_distance_loss_per_example(samples_y, outputs)
				self._errors[from_id:to_id] = losses
			self._hard_examples_idxs = np.argsort(-self._errors)
			lst = self._hard_examples_idxs[:self._number_of_batches_for_training*self._batch_size]
			self._hard_examples_idxs_for_training.extend(lst)
			random.shuffle(self._hard_examples_idxs_for_training)
		self._epoch_counter += 1

	def _jaccard_distance_loss_per_example(self, y_true, y_pred, smooth=1e-4):
		intersection = np.sum(y_true * y_pred, axis=(1, 2, 3))
		sum_ = np.sum(np.abs(y_true) + np.abs(y_pred), axis=(1, 2, 3))
		jac = intersection / (sum_ - intersection + smooth)
		return 1 - jac
