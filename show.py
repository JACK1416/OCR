# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np

corners = []

with open('data/mlt/label/213_icdar13.txt', 'r') as f:
    for line in f:
        line = line.strip().split(',')
        a = list(map(int, line))
        corners.append(a)

img = mpimg.imread('data/mlt/image/213_icdar13.png')

ax1 = plt.subplot(2, 1, 1)
plt.imshow(img)
plt.xticks([])
plt.yticks([])

ax2 = plt.subplot(2, 1, 2)
plt.imshow(img)
ax = plt.gca()
for corner in corners:
    width = corner[2] - corner[0]
    height = corner[3] - corner[1]
    rect = patches.Rectangle(corner[:2], width, height)
    ax.add_patch(rect)
plt.xticks([])
plt.yticks([])
plt.show()
