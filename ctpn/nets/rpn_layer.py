# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
import numpy as np
import tensorflow as tf

from nets.lstm_layer import brnn_layer

def rpn_layer(conv_feature, in_channel=512, k=10, scope_name='rpn'):
    '''
    for any sliding windows, feeding into 256D convolution
    '''
    with tf.variable_scope(scope_name) as scope:
        #intermediate layer
        kernel = tf.get_variable(
                'DW',
                [3, 3, in_channel, 512],
                tf.float32,
                initializer=tf.random_normal_initializer())
        strides = [1, 1, 1, 1]
        intermediate_layer = tf.nn.conv2d(conv_feature, kernel, strides, padding='SAME')
        intermediate_layer = tf.nn.relu(intermediate_layer)

        brnn_output = brnn_layer(intermediate_layer, 512, 128, 512)

        # output layer 
        cls_pred = _lstm_fc(brnn_output, 512, 2*k, 'score')
        bbox_pred = _lstm_fc(brnn_output, 512, 4*k, 'bbox')

        cls_pred= tf.nn.relu(cls_pred)
        bbox_pred = tf.nn.relu(bbox_pred)

        return bbox_pred, cls_pred

def _lstm_fc(bottom, in_channel, out_channel, scope):
    with tf.variable_scope(scope):
        strides = [1, 1, 1, 1]
        kernel = tf.get_variable(
                'kernel',
                [1, 1, in_channel, out_channel],
                initializer=tf.random_normal_initializer())
        output = tf.nn.conv2d(bottom, kernel, strides, padding='SAME')

        return output


if __name__ == '__main__':
    input_data = tf.placeholder(tf.float32, [None, None, None, 512])
    bbox_pred, cls_pred = rpn_layer(input_data)
    print(bbox_pred, cls_pred)
