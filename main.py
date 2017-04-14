"""
GAN on MNIST
JP Chen
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import edward as ed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import tensorflow as tf

from edward.models import Uniform
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data

def plot(samples):
  fig = plt.figure(figsize=(4, 4))
  gs = gridspec.GridSpec(4, 4)
  gs.update(wspace=0.05, hspace=0.05)

  for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

  return fig


ed.set_seed(100) # random #

M = 128  # batch size during training
d = 100  # latent dimension

DATA_DIR = "data/mnist"
IMG_DIR = "img"

if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)
if not os.path.exists(IMG_DIR):
  os.makedirs(IMG_DIR)

mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
x_ph = tf.placeholder(tf.float32, [M, 784])

