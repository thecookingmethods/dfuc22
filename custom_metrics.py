from tensorflow.keras import backend as K
from tensorflow import math as tfmath
import tensorflow as tf
import math

def dice(y_true, y_pred):
    y_pred = K.round(y_pred)
    c = K.sum(y_true * y_pred)
    a = K.sum(y_true)
    b = K.sum(y_pred)

    dice = 2 * c / (a + b)
    return dice


def plus_jaccard_distance_loss(build_in_loss):
    def jaccard_sth(y_true, y_pred):
        jaccard_loss = jaccard_distance_loss(y_true, y_pred)
        sth_loss = build_in_loss(y_true, y_pred)
        return 0.8*jaccard_loss + 0.2*sth_loss
    return jaccard_sth


def jaccard_distance_loss(y_true, y_pred, smooth=1e-4):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
    jac = intersection / (sum_ - intersection + smooth)
    return 1 - jac


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)
    return lr
