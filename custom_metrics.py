from tensorflow.keras import backend as K
from tensorflow import math as tfmath
import tensorflow as tf
import math

EPS = 1e-4
PI = 3.1415926


def lr_scheduler(epoch, lr):
    if epoch <= 20:
        lr = 1e-4
    elif epoch < 80:
        lr = 1e-2
    elif epoch < 150:
        lr = 1e-3
    else:
        lr = 1e-4
    return lr


def dist_rr(y_true, y_pred):
    r_r = r_r_metric(y_true, y_pred)
    dist = dist_metric(y_true, y_pred)
    loss = dist + r_r
    return loss


def hehe_loss(y_true, y_pred):
    rdr = rdr_metric(y_true, y_pred)
    siou = siou_metric(y_true, y_pred)
    #r_r = r_r_metric(y_true, y_pred)
    #d_rr = dist_rr(y_true, y_pred)
    #dist = dist_metric(y_true, y_pred)
    loss = 1.0 - siou + rdr #- siou_r_pred_large_metric(y_true, y_pred) - siou_r_true_large_metric(y_true, y_pred)

    return loss


def siou_vec(y_true, y_pred, eps=EPS, pi=PI):
    true_h, true_w, true_r = get_h_w_r(y_true)
    pred_h, pred_w, pred_r = get_h_w_r(y_pred)
    dist = get_dist(true_h, true_w, pred_h, pred_w)

    r_true_r_pred_gt_dist = K.cast(true_r + pred_r > dist, dtype='float32')
    r_true_plus_dist_lt_r_pred = K.cast(true_r + dist < pred_r, dtype='float32')
    r_pred_plus_dist_lt_r_true = K.cast(pred_r + dist < true_r, dtype='float32')

    siou_r_pred_large = r_true_r_pred_gt_dist * r_true_plus_dist_lt_r_pred * (K.pow(true_r / (pred_r + eps), 2))
    siou_r_true_large = r_true_r_pred_gt_dist * r_pred_plus_dist_lt_r_true * (K.pow(pred_r / (true_r + eps), 2))

    cos_a = (K.pow(true_r, 2) + K.pow(dist, 2) - K.pow(pred_r, 2)) / (2 * true_r * dist + eps)
    cos_b = (K.pow(pred_r, 2) + K.pow(dist, 2) - K.pow(true_r, 2)) / (2 * pred_r * dist + eps)
    h1 = true_r * (1 - cos_a)
    h2 = pred_r * (1 - cos_b)

    p = 0.5 * (true_r + pred_r + dist)
    H = 2 * K.sqrt(p * (p - true_r) * (p - pred_r) * (p - dist) + eps) / (dist + eps)

    intersection = H * h1 + H * h2
    sum_ = pi * (K.pow(true_r, 2) + K.pow(pred_r, 2)) - intersection

    cos_ab = (K.pow(pred_r, 2) + K.pow(true_r, 2) - K.pow(dist, 2)) / (2 * pred_r * true_r + eps)
    #theta_ab = tf.acos(cos_ab)
    eta = 1.0 - cos_ab  # theta_ab / pi

    siou3 = intersection / (sum_ + eps) - eta
    siou3 = siou3 * r_true_r_pred_gt_dist * (1.0 - r_true_plus_dist_lt_r_pred) * (1.0 - r_pred_plus_dist_lt_r_true)

    siou = siou3 + siou_r_pred_large + siou_r_true_large

    return siou


def siou_r_pred_large_metric(y_true, y_pred, eps=EPS):
    true_h, true_w, true_r = get_h_w_r(y_true)
    pred_h, pred_w, pred_r = get_h_w_r(y_pred)
    dist = get_dist(true_h, true_w, pred_h, pred_w)
    r_true_r_pred_gt_dist = K.cast(true_r + pred_r > dist, dtype='float32')
    r_true_plus_dist_lt_r_pred = K.cast(true_r + dist <= pred_r, dtype='float32')

    #siou1 = r_true_r_pred_gt_dist * r_true_plus_dist_lt_r_pred * K.pow(pred_r - true_r, 2)
    siou1 = r_true_r_pred_gt_dist * r_true_plus_dist_lt_r_pred * (K.pow(true_r / (pred_r + eps), 2))

    return K.mean(siou1)


def siou_r_true_large_metric(y_true, y_pred, eps=EPS):
    true_h, true_w, true_r = get_h_w_r(y_true)
    pred_h, pred_w, pred_r = get_h_w_r(y_pred)
    dist = get_dist(true_h, true_w, pred_h, pred_w)
    r_true_r_pred_gt_dist = K.cast(true_r + pred_r > dist, dtype='float32')
    r_pred_plus_dist_lt_r_true = K.cast(pred_r + dist <= true_r, dtype='float32')

    #siou2 = r_true_r_pred_gt_dist * r_pred_plus_dist_lt_r_true * K.pow(true_r - pred_r, 2)
    siou2 = r_true_r_pred_gt_dist * r_pred_plus_dist_lt_r_true * (K.pow(pred_r / (true_r + eps), 2))

    return K.mean(siou2)


def siou_metric(y_true, y_pred):
    siou = siou_vec(y_true, y_pred)
    return K.mean(siou)


def r_r_vec(y_true, y_pred):
    true_h, true_w, true_r = get_h_w_r(y_true)
    pred_h, pred_w, pred_r = get_h_w_r(y_pred)
    r_r = K.pow(true_r - pred_r, 2)
    return r_r


def r_r_metric(y_true, y_pred):
    r_r = r_r_vec(y_true, y_pred)
    return K.mean(r_r)


def rdr_vec(y_true, y_pred, eps=EPS):
    true_h, true_w, true_r = get_h_w_r(y_true)
    pred_h, pred_w, pred_r = get_h_w_r(y_pred)
    d = get_dist(true_h, true_w, pred_h, pred_w)
    rdr = d / (d + pred_r + true_r + eps)
    #r_true_r_pred_let_dist = K.cast(true_r + pred_r <= d, dtype='float32')

    return rdr# * r_true_r_pred_let_dist


def rdr_metric(y_true, y_pred):
    rdr = rdr_vec(y_true, y_pred)
    return K.mean(rdr)


def get_h_w_r(y):
    return y[:, 0], y[:, 1], y[:, 2]


def get_dist_vec(y_true, y_pred):
    true_h, true_w, _ = get_h_w_r(y_true)
    pred_h, pred_w, _ = get_h_w_r(y_pred)
    d = get_dist(true_h, true_w, pred_h, pred_w)
    return d


def dist_metric(y_true, y_pred):
    vec = get_dist_vec(y_true, y_pred)
    return K.mean(vec)


def get_dist(true_h, true_w, pred_h, pred_w, eps=EPS):
    d = K.sqrt(K.pow(true_h - pred_h, 2) + K.pow(true_w - pred_w, 2) + eps)
    return d


def iou(y_true, y_pred):
    y_pred = K.round(y_pred)
    intersection = y_true * y_pred
    union = 1 - ((1 - y_true) * (1 - y_pred))

    sum_of_intersection = K.sum(intersection)
    sum_of_union = K.sum(union)

    iou = sum_of_intersection / sum_of_union
    return iou


def dice(y_true, y_pred):
    y_pred = K.round(y_pred)
    c = K.sum(y_true * y_pred)
    a = K.sum(y_true)
    b = K.sum(y_pred)

    dice = 2 * c / (a + b)
    return dice


def tp(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f = K.round(y_pred_f)
    count = K.sum(y_true_f * y_pred_f)
    return count


def fp(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f = K.round(y_pred_f)
    count = K.sum((1-y_true_f) * y_pred_f)
    return count


def tn(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f = K.round(y_pred_f)
    count = K.sum((1-y_true_f) * (1-y_pred_f))
    return count


def fn(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f = K.round(y_pred_f)
    count = K.sum(y_true_f * (1-y_pred_f))
    return count


def plus_jaccard_distance_loss(build_in_loss):
    def jaccard_sth(y_true, y_pred):
        jaccard_loss = jaccard_distance_loss(y_true, y_pred)
        sth_loss = build_in_loss(y_true, y_pred)
        return 0.9*jaccard_loss + 0.1*sth_loss
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
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def dice_coefx(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


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