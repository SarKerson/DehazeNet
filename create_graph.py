import tensorflow as tf
from scipy.io import loadmat
import numpy as np
from dehazeNet import create_dehazeNet
MAT_PATH = './model/model_15_12b_64-epoch-300.mat'


def loadDehazeNet(data):

    # read layer info
    layers = data['layers']
    layers = layers[0][0][0][0:-1]    # from the first to the last second layer
    network = {}
    data_dict = {}
    layer_num = 1
    for layer in layers:
        name = layer['name'][0][0][0]
        layer_type = layer['type'][0][0][0]
        if layer_type == 'conv':
            data_dict['conv' + str(layer_num)] = {}
            if name[:2] == 'fc':
                padding = 'VALID'
            else:
                padding = 'SAME'
            stride = layer['stride'][0][0][0]
            kernel, bias = layer['weights'][0][0][0]
            if len(kernel.shape) == 3:
                kernel = np.expand_dims(kernel, 3)
            bias = np.squeeze(bias).reshape(-1)
            data_dict['conv' + str(layer_num)]['weights'] = kernel
            data_dict['conv' + str(layer_num)]['biases'] = bias
            print name, 'stride:', stride, 'kernel size:', np.shape(kernel)
        elif layer_type == 'relu':
            layer_num += 1
            print name + " " + layer_type
        elif layer_type == 'pool':
            stride = layer['stride'][0][0][0]
            pool = layer['pool'][0][0][0]
            print name, 'stride:', stride
        elif layer_type == 'bnorm':
            epsilon = layer['epsilon'][0][0][0]
            scale, offset, _ = layer['weights'][0][0][0]
            scale = np.transpose(scale)[0]
            offset = np.transpose(offset)[0]
            data_dict['conv' + str(layer_num)]['scale'] = scale
            data_dict['conv' + str(layer_num)]['offset'] = offset
            print name + " " + layer_type

    return data_dict

data = loadmat(MAT_PATH)
data = data['net']
data_dict = loadDehazeNet(data=data)

for layer in data_dict:
    print(layer)
    for opt in data_dict[layer]:
        if opt == 'epsilon':
            print(data_dict[layer][opt])
        elif opt != 'stride':
            print(opt + ":" + str(data_dict[layer][opt].shape))
        else:
            print(opt + ":" + str(data_dict[layer][opt]))

def init_graph(sess):
    for layer in data_dict:
        with tf.variable_scope(layer, reuse=True):
            for param_name, data in data_dict[layer].iteritems():
                var = tf.get_variable(param_name, trainable=False)
                sess.run(var.assign(data))

def main():
    image_batch = tf.constant(0,
                              dtype=tf.float32,
                              shape=[1, 300, 400, 1])
    net = create_dehazeNet(image_batch)
    var_list = tf.global_variables()

    # config = tf.ConfigProto()
    # config.gpu_op

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        init_graph(sess)

        saver = tf.train.Saver(var_list=var_list,
                               write_version=1)
        saver.save(sess,
                   "/tmp/dehazeNet/dehazeNet-model")

if __name__ == '__main__':
    main()