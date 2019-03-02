# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
import numpy as np
import tensorflow as tf

from utils.anchor_util import generate_anchors
from utils.bbox_transform import bbox_transform
from utils.bbox import bbox_overlaps, bbox_intersections
from utils.config import Config as cfg

def anchor_target_layer(cls_pred, bbox, im_info, scope_name):
    with tf.variable_scope(scope_name) as scope:
        # 'rpn_cls_score', 'gt_boxes', 'im_info'
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            tf.py_func(anchor_target_layer_py,
                       [cls_pred, bbox, im_info],
                       [tf.float32, tf.float32, tf.float32, tf.float32])

        rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32),
                                          name='rpn_labels')
        rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets,
                                                name='rpn_bbox_targets')
        rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights,
                                                       name='rpn_bbox_inside_weights')
        rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights,
                                                        name='rpn_bbox_outside_weights')

        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

def anchor_target_layer_py(cls_pred, gt_boxes, im_info, feat_stride=16, scale=None):
    '''
    Task: 1. Assign anchor to ground-truth targets
          2. Produce anchor classification labels
          3. Produce bounding-box regression targets
    '''
    # generate 10 basal anchors, can run file <anchor_util.py> to look the example
    if scale is not None:
        _anchor = generate_anchors(scale)
    else:
        _anchor = generate_anchors()
    _allowed_border = 0

    # height and width of feature map
    height, width = cls_pred.shape[1:3]
    # shift from feature map to ground-truth
    shift_x = np.arange(width) * feat_stride
    shift_y = np.arange(height) * feat_stride
    shift_x, shift_y = map(np.ravel, np.meshgrid(shift_x, shift_y))
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()

    # generate all anchors for a feature map
    all_anchors = (_anchor.reshape((1, -1, 4)) + shifts.reshape((-1, 1, 4))).reshape(-1, 4)
    total_anchors = all_anchors.shape[0]

    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &
        (all_anchors[:, 3] < im_info[0] + _allowed_border)
    )[0]
    anchors = all_anchors[inds_inside, :]
    num_inside_anchors = len(anchors)

    #=============================================================================
    # label : 1 is positive, 0 is negative, -1 is don't care
    labels = -np.ones(num_inside_anchors, dtype=np.float32)

    '''
    calculate overlaps between the anchors and the ground-truth boxes
    for labelling anchors
    '''
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    # for all inside anchors , finding ground-truth box which has max overlap
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(num_inside_anchors), argmax_overlaps]
    # for all ground-trurh boxes, finding anchors which has max overlap
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    #gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    
    # label anchors
    if not cfg.RPN_CLOBBER_POSITIVES:
        labels[max_overlaps < cfg.RPN_NEGATIVE_OVERLAP] = 0
    labels[gt_argmax_overlaps] = 1
    labels[max_overlaps >= cfg.RPN_POSITIVE_OVERLAP] = 1
    if cfg.RPN_CLOBBER_POSITIVES:
        labels[max_overlaps < cfg.RPN_NEGATIVE_OVERLAP] = 0

    # pass doncare areas part
    pass

    # pass hard samples part
    pass

    # subsample positive labels if we have too many
    num_fg = int(cfg.RPN_FG_FRACTION * cfg.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = np.random.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = cfg.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = np.random.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    #=============================================================================
    '''
    [TODO] need modify
    '''
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    bbox_inside_weights = np.zeros((num_inside_anchors, 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array(cfg.RPN_BBOX_INSIDE_WEIGHTS)
    
    bbox_outside_weights = np.zeros((num_inside_anchors, 4), dtype=np.float32)
    bbox_outside_weights[labels == 1, :] = 1.0

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside)
    
    A = _anchor.shape[0]
    rpn_labels = labels.reshape((1, height, width, A))
    shape = (1, height, width, A * 4)
    rpn_bbox_targets = bbox_targets.reshape(shape)
    rpn_bbox_inside_weights = bbox_inside_weights.reshape(shape)
    rpn_bbox_outside_weights= bbox_outside_weights.reshape(shape)

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

if __name__ == '__main__':
    from nets.rpn_layer import rpn_layer
    from nets.data_layer import DataLayer

    path = '../../data/mlt/'

    input_data = tf.placeholder(tf.float32, [None, None, None, 512], name='input_data')
    bbox_pred, cls_pred = rpn_layer(input_data)
    data_layer = DataLayer(path)
    gen = data_layer.generator()
    _, bbox, im_info = next(gen)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = anchor_target_layer(cls_pred, bbox, im_info, 'anchor_target_layer')
        sess.run(result, feed_dict={input_data:np.ones([1,14,14,512])})
        print('=' * 30)
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = result[0], result[1], result[2], result[3]

        print('rpn_labels : ', rpn_labels)
        print('rpn_bbox_targets : ', rpn_bbox_targets)
        print('rpn_bbox_inside_weights : ', rpn_bbox_inside_weights)
        print('rpn_bbox_outside_weights : ', rpn_bbox_outside_weights)
