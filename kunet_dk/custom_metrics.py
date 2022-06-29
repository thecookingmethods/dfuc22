from tensorflow.keras import backend
import tensorflow as tf
import math


def iou(y_true, y_pred):
    y_pred = backend.round(y_pred)
    intersection = y_true * y_pred
    union = 1 - ((1 - y_true) * (1 - y_pred))

    sum_of_intersection = backend.sum(intersection)
    sum_of_union = backend.sum(union)

    iou = sum_of_intersection / sum_of_union
    return iou


def dice(y_true, y_pred):
    y_pred = backend.round(y_pred)
    c = backend.sum(y_true * y_pred)
    a = backend.sum(y_true)
    b = backend.sum(y_pred)

    dice = 2 * c / (a + b)
    return dice


def tp(y_true, y_pred):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    y_pred_f = backend.round(y_pred_f)
    count = backend.sum(y_true_f * y_pred_f)
    return count


def fp(y_true, y_pred):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    y_pred_f = backend.round(y_pred_f)
    count = backend.sum((1-y_true_f) * y_pred_f)
    return count


def tn(y_true, y_pred):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    y_pred_f = backend.round(y_pred_f)
    count = backend.sum((1-y_true_f) * (1-y_pred_f))
    return count


def fn(y_true, y_pred):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    y_pred_f = backend.round(y_pred_f)
    count = backend.sum(y_true_f * (1-y_pred_f))
    return count


def plus_jaccard_distance_loss(build_in_loss):
    def jaccard_sth(y_true, y_pred):
        jaccard_loss = jaccard_distance_loss(y_true, y_pred)
        sth_loss = build_in_loss(y_true, y_pred)
        return 0.2*jaccard_loss + 0.8*sth_loss
    return jaccard_sth


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = backend.sum(backend.abs(y_true * y_pred), axis=-1)
    sum_ = backend.sum(backend.abs(y_true) + backend.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def dice_coefx(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = backend.sum(backend.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (backend.sum(backend.square(y_true),-1) + backend.sum(backend.square(y_pred),-1) + smooth)


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def plus_dice_distance_loss(build_in_loss):
    def dice_sth(y_true, y_pred):
        dice_loss = dice_coef_loss(y_true, y_pred)
        sth_loss = build_in_loss(y_true, y_pred)
        return dice_loss + sth_loss
    return dice_sth


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)
    return lr


def lr_decay_fun(start_lr, drop_by_percetage, drop_every_n_epoch):
    def lr_decay(epoch):
        lr = start_lr * math.pow(drop_by_percetage, math.floor((1+epoch)/drop_every_n_epoch))
        return lr
    return lr_decay