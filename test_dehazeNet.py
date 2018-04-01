from scipy.io import loadmat
from dehazeNet import dehazeNet
from cv2 import imread, resize, imwrite, INTER_LINEAR
import cv2
import tensorflow as tf
import numpy as np
import time

MAT_PATH = '/home/sar/SarKerson/dehaze/DnCNN/DeCNN/data/model_15_12b_64/model_15_12b_64-epoch-300.mat'
TR_PATH = 'pre_t.png'

# build the graph
graph = tf.Graph()
with graph.as_default():
    data = loadmat(MAT_PATH)
    data = data['net']

    # read meta info
    meta = data['meta']
    learning_rate = meta[0][0][0][0][1]
    inputSize = meta[0][0][0][0][2]
    image_size = np.squeeze(inputSize)
    input_maps = tf.placeholder(tf.float32, [None, 300, 400, 1])
    net = dehazeNet(data, input_maps)
    output = net['layer24'][0]

img = imread(TR_PATH, 0)
img = resize(img, (400, 300), interpolation=INTER_LINEAR)
# print(img.shape)
img = np.expand_dims(img, 0)
img = np.expand_dims(img, 3)
#
# # run the graph
with tf.Session(graph=graph) as sess:
    time1 = time.time()
    output_trm = sess.run(output, feed_dict={input_maps: img})
    output_trm = np.reshape(output_trm, [300, 400])
    output_trm = output_trm * 255.0
    print('time:' + str(time.time() - time1)[0:5])
# imshow('out', output_trm)
imwrite('out.png', output_trm)