import numpy as np
from functools import reduce
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

class HiddenLayer:

    def __init__(self, layer_input, n_in, n_out, random_generator, W=None, b=None, activation=T.tanh, name=None):
        """ Cria uma camada escondida genérica

        :type layer_input: theano.tensor.TensorType
        :param layer_input: variável simbólica que representa a entrada da camada

        :type n_in: int
        :param n_in: quantidade de entradas da camada

        :type n_out: int
        :param n_out: quantidade de neurônios e saídas da camada

        :type random_generator: numpy.random.RandomState
        :param random_generator: um gerador de números aleatórios para criar
                                 a matriz de pesos iniciais

        :type activation: theano.Op or function
        :param activation: função de ativação não linear

        """

        self.layer_input = layer_input

        # Initialize with random values the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            W_values = np.asarray(
                random_generator.uniform(
                    low=-6/(n_in+n_out),
                    high=6/(n_in+n_out),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            # Allocates weight matrix in theano
            W = theano.shared(
                value=W_values,
                name='W' if name is None else 'W{}'.format(name),
                borrow=True
            )

        # Initialize the biases b as a vector of n_out 0s
        if b is None:
            b_values = np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            )

            # Allocates biases vector in theano
            b = theano.shared(
                value=b_values,
                name='b' if name is None else 'b{}'.format(name),
                borrow=True
            )

        self.W = W
        self.b = b

        # Tells theano how to calculate the linear output (before activation function)
        lin_output = T.dot(layer_input, self.W) + self.b
        # Tells theano how to calculate output (activation function applied
        # on linear output)
        self.output = lin_output if activation is None else activation(lin_output)
        self.params = [self.W, self.b]
