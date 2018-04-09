import tensorflow as tf
from scipy.io import loadmat
import numpy as np
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
            data_dict['conv' + str(layer_num)]['weight'] = kernel
            data_dict['conv' + str(layer_num)]['bias'] = bias
            data_dict['conv' + str(layer_num)]['stride'] = np.array([1, stride[0], stride[0], 1])
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
            data_dict['conv' + str(layer_num)]['epsilon'] = epsilon
            print name + " " + layer_type

    return data_dict

data = loadmat(MAT_PATH)
data = data['net']
data_dict = loadDehazeNet(data=data)

# for layer in data_dict:
#     for opt in data_dict[layer]:
#         if opt == "weight":
#
def init_graph(sess)
    for layer in data_dict:
        with tf.variable_scope(layer, reuse=True):
            for param_name, data in data_dict[layer].iteritems():
                try:
                    var = tf.get_variable(param_name)
                    sess.run(var.assign(data))
                except ValueError:
                    print("VALUE ERROR!")

def main:


if __name__ == '__main__':
    main()