# -*- coding: utf-8 -*-
import numpy as np


def generate_basic_anchors(sizes, base_size=16):
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1], np.int32)
    anchors = np.zeros((len(sizes), 4), np.int32)
    index = 0
    for h, w in sizes:
        anchors[index] = scale_anchor(base_anchor, h, w)
        index += 1
    return anchors


def scale_anchor(anchor, h, w):
    x_ctr = (anchor[0] + anchor[2]) * 0.5
    y_ctr = (anchor[1] + anchor[3]) * 0.5
    scaled_anchor = anchor.copy()
    scaled_anchor[0] = x_ctr - w / 2  # xmin
    scaled_anchor[2] = x_ctr + w / 2  # xmax
    scaled_anchor[1] = y_ctr - h / 2  # ymin
    scaled_anchor[3] = y_ctr + h / 2  # ymax
    return scaled_anchor


def generate_anchors(base_size=16,
                     scales=1.4285714285714286 ** np.arange(-1, 9)): # 1 / 0.7
    heights = base_size * scales
    widths = [base_size] # widths fixed
    sizes = []
    for h in heights:
        for w in widths:
            sizes.append((h, w))
    return generate_basic_anchors(sizes, base_size)


if __name__ == '__main__':
    import time
    import cv2 as cv

    t = time.time()
    a = generate_anchors()
    print('use time: ', time.time() - t)
    print(a)
    img = np.ones([300, 48, 3])
    x_bias = 16
    y_bias = 140
    for p in a:
        cv.rectangle(img,
                     (p[0] + x_bias, p[1] + y_bias),
                     (p[2] + x_bias, p[3] + y_bias),
                     color=(0, 0, 0),
                     thickness=1)
    cv.imshow('example', img)
    cv.waitKey(0)

