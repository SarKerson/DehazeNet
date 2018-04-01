import tensorflow as tf
from cv2 import imwrite, imshow, imread, resize
from ImageUtils import imgworker
from ImageUtils import fuse

import time
import numpy as np
import os

sess = tf.Session()

file_name = 'sea2.png'

source = imread(file_name)/255.0
source = source.astype('float32')
tune_out = imgworker.auto_tune(source)
haze_free = imgworker.haze_free(sess,tune_out,15)

time_t = time.time()
time1 = time.time()
tune_out = imgworker.auto_tune(source)
time_auto_tune = time.time()-time1
time1 = time.time()
haze_free = imgworker.haze_free(sess,tune_out,15)
time_haze_free = time.time()-time1
time1 = time.time()
h_eq = imgworker.equlize_hist(tune_out)
time_equlize_hist = time.time()-time1
time1 = time.time()
final = fuse.laplace_fuse(haze_free, h_eq)
time_laplace_fuse = time.time()-time1
time_total = time.time()-time_t
file = open("fuck.txt",mode='a')
file.write(file_name + '\t' + str(time_auto_tune)[0:5]+'\t\t'+str(time_haze_free)[0:5]+'\t\t'+str(time_equlize_hist)[0:5]+'\t\t'+str(time_laplace_fuse)[0:5]+'\t\t'+str(time_total)[0:5]+'\r\n')
file.close()
imwrite('ylq_new'+file_name,final)
#imshow('final', final)
