from mnist import MNIST
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import os.path

from multilayer_perceptron import *

# Padrão X -> imagens ; Y -> labels

# Extração de dados (biblioteca python-mnist)
mnist_data = MNIST(os.path.abspath('data'))
X, Y = mnist_data.load_training()
X_training = np.reshape(np.asarray(X, dtype=np.uint8), (60000, 28 * 28))
Y_training = np.reshape(np.asarray(Y, dtype=np.uint8), (60000,))
X, Y = mnist_data.load_testing()
X_test = np.reshape(np.asarray(X, dtype=np.uint8), (10000, 28 * 28))
Y_test = np.reshape(np.asarray(Y, dtype=np.uint8), (10000,))

# A partir daqui temos imagens em X_training e X_test e labels em Y_training e Y_test
x = T.ivector(name='x')
teste = MultilayerPerceptron((10,9,8,7,6,5,4,3,2), x, np.random.RandomState(1234))

# Build model #

# Counts each batch
index = T.lscalar()
# Each line is an example on a batch
x = T.matrix('x')
y = T.ivector('y')
