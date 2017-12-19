# 2017.12.18. Seokju Lee
# Debugging code (pycaffe)

import numpy as np
import os
import sys
import setproctitle
import scipy
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import argrelextrema
from scipy import signal
from matplotlib import pylab as pl
from sklearn import linear_model, datasets
from scipy.spatial.distance import pdist, squareform
import random
import time
import pdb

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# init
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')

# solver.net.copy_from(weights)
# solver.net.set_mode_gpu()


for _ in range(100):
	# pdb.set_trace()
	n = 0

	# solver.solve()
	solver.net.forward()
	# solver.test_nets[0].forward()

	# pdb.set_trace()

	data = solver.net.blobs['data'].data[n]
	label = solver.net.blobs['label'].data[n]
	type = solver.net.blobs['type'].data[n]
	# vp = solver.net.blobs['vp'].data[n]
	# mask = solver.net.blobs['mask'].data[n]

	# plt.imshow(data[0]), plt.colorbar(), plt.ion(), plt.show()
	# plt.imshow(type[0]), plt.colorbar(), plt.ion(), plt.show()
	# plt.imshow(label[0]), plt.colorbar(), plt.ion(), plt.show()

	pdb.set_trace()
	# print [(k, v.data.shape) for k, v in solver.net.blobs.items()]

	# pdb.set_trace()
