from mnist import MNIST
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import os.path

from multilayer_perceptron_classifier import *

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
teste = MultilayerPerceptronClassifier((28*28,500,10), n_epochs=1)

X_valid = X_training[0:5000,:]
Y_valid = Y_training[0:5000]
X_training = X_training[5000:,:]
Y_training = Y_training[5000:]

teste.fit((X_training,Y_training), (X_valid, Y_valid), (X_test, Y_test))
teste.save_model('teste.json')
teste.load_model('teste.json')
