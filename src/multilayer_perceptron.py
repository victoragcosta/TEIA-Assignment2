import numpy as np
from functools import reduce
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from logistic_regression_layer import *
from hidden_layer import *

class MultilayerPerceptron:

    def __init__(self, structure, classifier_input, random_generator, last_logistic=True):
        """ Cria uma rede neural completa baseada nos parâmetros de estrutura

        :type structure: tuple
        :param structure: tupla contendo quantos neurônios por camada, incluindo a entrada

        :type classifier_input: theano.tensor.TensorType
        :param classifier_input: variável simbólica que representa a entrada do
                                 classificador

        :type random_generator: numpy.random.RandomState
        :param random_generator: um gerador de números aleatórios para criar
                                 a matriz de pesos iniciais

        """

        # Start list of layers
        self.layers = []
        # The first input is the classifier input
        layer_input = classifier_input
        # Creates pairs of before and after layers neurons counts
        if last_logistic:
            pairs = zip(structure[:-2],structure[1:-1])
        else:
            pairs = zip(structure[:-1],structure[1:])
        # Creates all but the last layer
        for count, (layer_count, next_layer_count) in enumerate(pairs):
            #print(count, '(', layer_count, ',', next_layer_count, ')')
            # Creates a layer with the input as the output of the last layer
            new_layer = HiddenLayer(
                layer_input=layer_input,
                n_in=layer_count,
                n_out=next_layer_count,
                random_generator=random_generator,
                name=count
            )
            # Save in the layers list
            self.layers.append(new_layer)
            # Sets the new input to be the last output
            layer_input = new_layer.output

        # Add the last layer, that is logistical
        if last_logistic:
            new_layer = LogisticRegressionLayer(
                layer_input=layer_input,
                n_in=structure[-2],
                n_out=structure[-1]
            )
            self.layers.append(new_layer)
        self.output = new_layer.output
        self.y_prediction = T.argmax(self.output, axis=1)

        self.L1 = reduce(
            lambda a, b: a+b,
            map(
                lambda layer: abs(layer.W).sum(),
                self.layers
            )
        )

        self.L2_sqr = reduce(
            lambda a, b: a+b,
            map(
                lambda layer: (layer.W ** 2).sum(),
                self.layers
            )
        )

        self.params = reduce(lambda a,b: a+b, map(lambda l: l.params, self.layers))

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.output)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """ Retorna um float representando a porcentagem de erros da minibatela

        :type y: theano.tensor.TensorType
        :param y: saídas esperadas para cada minibatela
        """

        # check if y has same dimension of y_prediction
        if y.ndim != self.y_prediction.ndim:
            raise TypeError(
                'y deve ter a mesma forma que self.y_prediction',
                ('y', y.type, 'y_prediction', self.y_prediction.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_prediction, y))
        else:
            raise NotImplementedError()
