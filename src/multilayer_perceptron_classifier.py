import numpy as np
from functools import reduce
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import timeit

from multilayer_perceptron import *

class MultilayerPerceptronClassifier:

    def __init__(self, structure, seed=1234, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, batch_size=20):
        """ Constroi modelo e treina o modelo

        :type structure: tuple
        :param structure: tupla contendo quantos neurônios por camada, incluindo a entrada

        :type seed: int
        :param seed: seed a ser usada para criar os pesos da rede aleatóriamente

        :type learning_rate: float
        :param learning_rate: taxa de aprendizado do modelo, evita overshoot do mínimo da
                              função erro. Deve ser menor que 1 para convergir.

        :type L1_reg: float
        :param L1_reg: alguma coisa

        :type L2_reg: float
        :param L2_reg: alguma coisa

        :type n_epochs: int
        :param n_epochs: quantas vezes o conjunto de treinamento completo deve ser treinado

        :type batch_size: int
        :param batch_size: quantos exemplos devem ser mostrados à rede antes de ser
                           realizada a backpropagation

        """
        # Save parameters
        self.structure = structure
        self.seed = seed
        self.learning_rate = learning_rate
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Each line is an example on a batch
        self.x = T.matrix('x')
        # A vector with all the outputs for the batch
        self.y = T.ivector('y')

        self.classifier = MultilayerPerceptron(
            structure=structure,
            classifier_input=x,
            random_generator=np.random.RandomState(seed)
        )

        self.cost = (
            self.classifier.negative_log_likelihood(y)
            + L1_reg * self.classifier.L1
            + L2_reg * self.classifier.L2_sqr
        )

        self.test_model = theano.function(
            inputs=[x, y],
            outputs=self.classifier.errors(y)
        )

        self.validate_model = theano.function(
            inputs=[x, y],
            outputs=classifier.errors(y)
        )

        # Calculates the gradient for the parameters
        self.gparams = [T.grad(self.cost, param) for param in self.classifier.params]

        # Tells theano how to update the weights
        self.updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.classifier.params, self.gparams)
        ]

        # Compiles function for both feedforward and backpropagation
        self.train_model = theano.function(
            inputs=[x, y],
            outputs=self.cost,
            updates=self.updates
        )


    def fit(self, X_train, Y_train, X_validation, Y_validation):
        """ Treina o modelo com os dados fornecidos

        :type X_train: numpy.ndarray
        :param X_train: entradas de treinamento
        :type Y_train: numpy.ndarray
        :param Y_train: saídas desejadas a partir de entrada

        """
        n_train_batches = Y_train.shape[0] // self.batch_size
        n_valid_batches = Y_validation.shape[0] // self.batch_size
        validation_frequency = n_train_batches

        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):

                minibatch_avg_cost = self.train_model(
                    X_train[minibatch_index * self.batch_size:(minibatch_index+1) * self.batch_size],
                    Y_train[minibatch_index * self.batch_size:(minibatch_index+1) * self.batch_size]
                )
                # iteration number
                iteration = (epoch - 1) * n_train_batches + minibatch_index

                if (iteration + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [self.validate_model(
                        X_validation[minibatch_index * self.batch_size:(minibatch_index+1) * self.batch_size],
                        Y_validation[minibatch_index * self.batch_size:(minibatch_index+1) * self.batch_size]
                    ) for i in range(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)

                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                        ):
                            patience = max(patience, iteration * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iteration = iteration

                        # test it on the test set
                        test_losses = [test_model(i) for i
                                    in range(n_test_batches)]
                        test_score = np.mean(test_losses)

                        print(('     epoch %i, minibatch %i/%i, test error of '
                            'best model %f %%') %
                            (epoch, minibatch_index + 1, n_train_batches,
                            test_score * 100.))

                if patience <= iteration:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print(('Optimization complete. Best validation score of %f %% '
            'obtained at iteration %i, with test performance %f %%') %
            (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print(('The code for file ' +
            os.path.split(__file__)[1] +
            ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
