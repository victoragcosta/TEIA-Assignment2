import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt


class HiddenLayers:

    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):

        self.input = input

        # initialize with random values the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-1,
                    high=1,
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            W = theano.shared(value=W_values, name='W', borrow=True)

        # initialize the biases b as a vector of n_out 0s
        if b is None:
            b = theano.shared(
                value=np.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]



class MultilayerPerceptron:
    def __init__(self):
        pass
