import tensorflow as tf
from scipy.io import loadmat
import numpy as np
from dehazeNet import create_dehazeNet


from ImageUtils.imgworker import cal_dark_channel_fast, dehazeFun, guidedfilter, em_A_color, auto_tune, boundcon
MAT_PATH = './model/imdb2.mat'
IMG_PATH = './input/d.jpg'

IMG_WIDTH = 180
IMG_HEIGHT = 180
IMG_CHANNEL = 1
NUM_TRAIN=800
NUM_TEST=200
LEARNING_RATE_BASE = 0.01
MOVING_AVG_DEC = 0.99
LEARNING_RATE_DEC = 0.99
BATCH_SIZE = 200

data = loadmat(MAT_PATH)
data_train = np.zeros([NUM_TRAIN, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL])
label_train = np.zeros([NUM_TRAIN, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL])
data_test = np.zeros([NUM_TEST, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL])
label_test = np.zeros([NUM_TEST, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL])

for i in range(800):
    data_train[i,:,:,0]=data['data'][:,:,0,i]
    label_train[i,:,:,0]=data['label'][:,:,0,i]
for i in range(200):
    data_test[i,:,:,0]=data['data'][:,:,0,800+i]
    label_test[i,:,:,0]=data['label'][:,:,0,800+i]

BATCH_SIZE = 5

def init_graph(sess):
    for i in range(1, 13):
        with tf.variable_scope('conv'+str(i), reuse=True):
            weights = tf.get_variable("weights", trainable=True)
            biases = tf.get_variable("biases", trainable=True)
            if i is 1:
                sess.run(weights.assign(tf.truncated_normal(shape=[3, 3, 1, 64],
                                                            mean=0,
                                                            stddev=0.1)))
                sess.run(biases.assign(tf.constant(0.1, shape=[64])))
            elif i is 12:
                sess.run(weights.assign(tf.truncated_normal(shape=[3, 3, 64, 1],
                                                           mean=0,
                                                           stddev=0.1)))
                sess.run(biases.assign(tf.constant(0.1, shape=[1])))
            else:
                sess.run(weights.assign(tf.truncated_normal(shape=[3, 3, 64, 64],
                                                            mean=0,
                                                            stddev=0.1)))
                sess.run(biases.assign(tf.constant(0.1, shape=[64])))


def train():
    X = tf.placeholder(tf.float32, shape=[None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL])
    Y = tf.placeholder(tf.float32, shape=[None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL])

    Y_pre = create_dehazeNet(X)
    Y_pre = tf.reshape(Y_pre, shape=[-1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL])

    global_step = tf.Variable(0, trainable=False)
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVG_DEC, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())
    learning_rate = tf.train.exponential_decay(
        learning_rate=LEARNING_RATE_BASE,
        global_step=global_step,
        decay_steps=NUM_TRAIN / BATCH_SIZE,
        decay_rate=LEARNING_RATE_DEC)

    loss = tf.reduce_mean(tf.pow(tf.subtract(Y_pre, Y), 2.0))

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)

    with tf.control_dependencies([optimizer, variable_average_op]):
        train_op = tf.no_op(name="train")

    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer());
        init_graph(sess)
        for epoch in range(1000):
            for i in range(int(NUM_TRAIN / BATCH_SIZE)):
                batch = []
                for j in range(BATCH_SIZE):
                    batch.append(np.random.random_integers(0, NUM_TRAIN - 1, 1))

                batch_X = data_train[batch, :, :, :]
                batch_Y = label_train[batch, :, :, :]

                _, loss_value, step = sess.run([train_op, loss, global_step],
                                               feed_dict={X: batch_X, Y: batch_Y})

            if epoch % 10 == 0:
                print("After %d training steps, loss on training batch is %g."
                      % (step, loss_value))

                saver.save(sess, "./model/new-model/new-model",
                           global_step=global_step)
        graphdef = tf.get_default_graph().as_graph_def()
        frozen_graph = tf.graph_util.convert_variables_to_constants(sess,
                                                                    graphdef,
                                                                    ['conv12/output'])
        with tf.gfile.GFile('./model/new-model/new-model.pb', "wb") as f:
            f.write(frozen_graph.SerializeToString())

if __name__ == '__main__':
    train()