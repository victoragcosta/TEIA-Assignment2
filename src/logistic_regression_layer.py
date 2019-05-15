import numpy as np
from functools import reduce
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

class LogisticRegressionLayer():

    def __init__(self, layer_input, n_in, n_out, W=None, b=None, name='out'):
        """ Cria uma camada de saída com probabilidades de ser cada classe

        :type layer_input: theano.tensor.TensorType
        :param layer_input: variável simbólica que representa a entrada da camada

        :type n_in: int
        :param n_in: quantidade de entradas para cada neurônio

        :type n_out: int
        :param n_out: quantidade de neurônios da camada

        """

        # Keep track of model input
        self.layer_input = layer_input

        # Create weights if none is given
        if W is None:
            # Allocates weight matrix in theano
            W = theano.shared(
                value=np.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),
                name='W' if name is None else 'W{}'.format(name),
                borrow=True
            )

        # Create biases if none is given
        if b is None:
            # Allocates biases vector in theano
            b = theano.shared(
                value=np.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b' if name is None else 'b{}'.format(name),
                borrow=True
            )

        self.W = W
        self.b = b

        # Symbolic expression for computing the matrix of class-membership
        # probabilities
        self.output = T.nnet.softmax(T.dot(layer_input, self.W) + self.b)
        self.params = [self.W, self.b]
