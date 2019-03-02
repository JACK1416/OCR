# -*- coding: utf-8 -*-
import sys
sys.path.append('..')

import tensorflow as tf

from nets.vgg16 import Vgg16
from nets.rpn_layer import rpn_layer
from utils.anchor_target_layer_tf import anchor_target_layer

def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def ctpn(image):
    vgg = Vgg16('../pre-weights/vgg16.npy')
    conv5_3 = vgg.build(image)

    bbox_pred, cls_pred = rpn_layer(conv5_3)

    # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
    cls_pred_shape = tf.shape(cls_pred)
    cls_pred_reshape = tf.reshape(cls_pred, [cls_pred_shape[0], cls_pred_shape[1], -1, 2])

    return bbox_pred, cls_pred
