from mnist import MNIST
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import os.path

from multilayer_perceptron import *

# Padrão X -> imagens ; Y -> labels

# Extração de dados (biblioteca python-mnist)
mndata = MNIST(os.path.abspath('data'))
X, Y = mndata.load_training()
X_training = np.reshape(np.asarray(X, dtype=np.uint8), (60000, 28, 28))
Y_training = np.reshape(np.asarray(Y, dtype=np.uint8), (60000,))
X, Y = mndata.load_testing()
X_test = np.reshape(np.asarray(X, dtype=np.uint8), (10000, 28, 28))
Y_test = np.reshape(np.asarray(Y, dtype=np.uint8), (10000,))

# Teste básico de carregamento
print(Y_training[0:1])
print(Y_test[0:1])

plt.matshow(np.squeeze(X_training[0:1,:,:], axis=0))
plt.matshow(np.squeeze(X_test[0:1,:,:], axis=0))
plt.show()

# A partir daqui temos imagens em X_training e X_test e labels em Y_training e Y_test
