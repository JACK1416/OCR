# -*- coding: utf-8 -*-
import os
import sys
sys.path.append('..')
import time

import numpy as np
import cv2 as cv

from utils.data_util import GeneratorEnqueuer


class DataLayer:
    def __init__(self, path):
        self.path = path

    def get_batch(self, num_works, **kwargs):
        try:
            enqueuer = GeneratorEnqueuer(self.generator(**kwargs), use_multiprocessing=True)
            enqueuer.start(num_works, 24)
            generator_output = None
            while True:
                while enqueuer.is_running():
                    if not enqueuer.queue.empty():
                        generator_output = enqueuer.queue.get()
                        break
                    else:
                        time.sleep(0.01)
                yield generator_output
                generator_output = None
        finally:
            if enqueuer is not None:
                enqueuer.stop()

    def generator(self, vis=False):
        '''
        '''
        image_list = np.array(self._get_data())
        index = np.arange(image_list.shape[0])
        np.random.shuffle(index)
        while True:
            for i in index:
                try:
                    im_fn = image_list[i]
                    im = cv.imread(im_fn)
                    im = cv.resize(im, (224, 224))
                    h, w, c = im.shape
                    im_info = np.array([h, w, c])

                    _, fn = os.path.split(im_fn)
                    fn, _ = os.path.splitext(fn)
                    txt_fn = os.path.join(self.path, 'label', fn + '.txt')
                    if not os.path.exists(txt_fn):
                        print('Ground truth for image ', im_fn, ' not exist!')
                        continue
                    bbox = self._get_annotation(txt_fn)
                    if (len(bbox) == 0):
                        print('Ground truth for image ', im_fn, ' empty!')
                        continue

                    if vis:
                        for p in bbox:
                            cv.rectangle(im, (p[0], p[1]), (p[2], p[3]), color=(0, 0, 255), thickness=1)
                        cv.imshow('iamge and annotation', im)
                        cv.waitKey(0)
                    yield [im], bbox, im_info

                except Exception as e:
                    print(e)
                    continue

    def _get_data(self):
        img_files = []
        exts = ['jpg', 'png', 'jpeg']
        img_path = os.path.join(self.path, 'image')
        for fname in os.listdir(img_path):
            fname_lower = fname.lower()
            for ext in exts:
                if (fname_lower.endswith(ext)):
                    img_files.append(os.path.join(img_path, fname))
                    break
        print('Find ', len(img_files), ' images in ', img_path)
        return img_files

    def _get_annotation(self, fname):
        bbox = []
        with open(fname, 'r') as f:
            for line in f:
                line = line.strip().split(',')
                x_min, y_min, x_max, y_max = map(int, line)
                bbox.append([x_min, y_min, x_max, y_max, 1])
            bbox = np.array(bbox)
        return bbox

if __name__ == '__main__':
    data_layer = DataLayer('../../data/mlt/')
    '''
    [TODO] why is get_batch wrong
    '''
    #gen = data_layer.get_batch(num_works=2, vis=True)
    gen = data_layer.generator(vis=True)

    for image, bbox, im_info in gen:
        print(im_info)
    print('done')
