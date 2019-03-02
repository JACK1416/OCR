# -*- coding: utf-8 -*-
import numpy as py
import tensorflow as tf

def brnn_layer(bottom, in_channel, rnn_size, out_channel, scope_name='brnn'):
    '''
    bottom is intermediate layer feature map (?, 14, 14, 512)
    '''
    with tf.variable_scope(scope_name) as scope:
        shape = tf.shape(bottom)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        bottom = tf.reshape(bottom, [N * H, W, C])
        bottom.set_shape([None, None, in_channel])
        bottom = _gru_layer(bottom, rnn_size)
        bottom = _fc_layer(bottom, 2*rnn_size, out_channel)
        bottom = tf.reshape(bottom, [N, H, W, out_channel])

        return bottom

def _fc_layer(bottom, in_size, out_size):
    initial_value = tf.random_normal([in_size, out_size])
    W = tf.Variable(initial_value)
    b = tf.Variable(tf.zeros([out_size], dtype=tf.float32))
    x = tf.reshape(bottom, [-1, in_size])
    fc = tf.nn.bias_add(tf.matmul(x, W), b)
    return fc

def _gru_layer(input_sequence, rnn_size):
    cell_fw = tf.contrib.rnn.GRUCell(rnn_size)
    cell_bw = tf.contrib.rnn.GRUCell(rnn_size)
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                    cell_bw,
                                                    input_sequence,
                                                    dtype=tf.float32)
    rnn_output_stack = tf.concat(rnn_output, 2, name='output_stack')

    return rnn_output_stack


if __name__ == '__main__':
    import numpy as np
    input_data = tf.placeholder(tf.float32, [None, None, None, 512])
    output_data = brnn_layer(input_data, 512, 128, 512)
    print(output_data)
    with tf.Session() as sess:
        data = np.ones([1,3,3,512])
        sess.run(tf.global_variables_initializer())
        print(sess.run(output_data, feed_dict={input_data: data}))
