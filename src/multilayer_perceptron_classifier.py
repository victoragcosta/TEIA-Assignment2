import numpy as np
from functools import reduce
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import timeit
import json
from decimal import Decimal
from sklearn.metrics import classification_report, precision_score

from multilayer_perceptron import *

class MultilayerPerceptronClassifier:

    def __init__(self, structure, seed=1234, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, batch_size=20, last_logistic=False, cost_function=None, early_stopping=False):
        """ Constrói modelo e treina o modelo

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

        :type last_logistic: bool
        :param last_logistic: se verdadeiro, a última camada soltará probabilidades de ser
                              cada classe

        :type cost_function: str or None
        :param cost_function: indica que função de custo usar para o backpropagation e
                              métricas. {'negative_log_likelihood', None}

        :type early_stopping: bool
        :param early_stopping: determina se deve parar o treinamento antes de completar
                               todas as épocas caso não haja melhora significativa

        """
        # Save parameters
        self.structure = structure
        self.seed = seed
        self.learning_rate = learning_rate
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.last_logistic = last_logistic
        self.cost_function = cost_function
        self.early_stopping = early_stopping

        # To help create statistics
        self.valid_loss_graph = []
        self.test_loss_graph = []
        self.valid_precision_graph = []
        self.test_precision_graph = []

        # Creates an index for easy control
        self.index = T.lscalar()
        # Each line is an example on a batch
        self.x = T.matrix('x')
        # A vector with all the outputs for the batch
        self.y = T.ivector('y')

        self.classifier = MultilayerPerceptron(
            structure=structure,
            classifier_input=self.x,
            random_generator=np.random.RandomState(seed),
            last_logistic=last_logistic
        )

        if cost_function is None:
            self.cost = (
                self.classifier.mean_squared_error(self.y)
                + L1_reg * self.classifier.L1
                + L2_reg * self.classifier.L2_sqr
            )
        elif cost_function == 'negative_log_likelihood':
            self.cost = (
                self.classifier.negative_log_likelihood(self.y)
                + L1_reg * self.classifier.L1
                + L2_reg * self.classifier.L2_sqr
            )

        # Calculates the gradient for the parameters
        self.gparams = [T.grad(self.cost, param) for param in self.classifier.params]

        # Tells theano how to update the weights
        self.updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.classifier.params, self.gparams)
        ]


    # Helps to copy less times the data to the GPU and increase efficiency
    @staticmethod
    def share_data(X, Y, borrow=True):
        shared_X = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=borrow)
        shared_Y = theano.shared(np.asarray(Y, dtype=theano.config.floatX), borrow=borrow)
        return shared_X, T.cast(shared_Y, 'int32')

    def fit(self, train, valid, test):
        """ Treina o modelo com os dados fornecidos

        :type train: tuple(numpy.ndarray, numpy.ndarray)
        :param train: tupla de entradas e saídas de treinamento

        :type valid: tuple(numpy.ndarray, numpy.ndarray)
        :param valid: tupla de entradas e saídas de validação

        :type valid: tuple(numpy.ndarray, numpy.ndarray)
        :param valid: tupla de entradas e saídas de teste

        """
        # Convert from numpy arrays to theano shared variables
        X_train, Y_train = MultilayerPerceptronClassifier.share_data(train[0], train[1])
        X_valid, Y_valid = MultilayerPerceptronClassifier.share_data(valid[0], valid[1])
        X_test, Y_test   = MultilayerPerceptronClassifier.share_data(test[0],  test[1])

        n_train_batches = X_train.get_value(borrow=True).shape[0] // self.batch_size
        n_valid_batches = X_valid.get_value(borrow=True).shape[0] // self.batch_size
        n_test_batches  = X_test.get_value(borrow=True).shape[0] // self.batch_size

        # Compiles a function to validate the model
        self.validate_model = theano.function(
            inputs=[self.index],
            outputs=self.classifier.errors(self.y),
            givens= {
                self.x: X_valid[self.index * self.batch_size:(self.index+1) * self.batch_size],
                self.y: Y_valid[self.index * self.batch_size:(self.index+1) * self.batch_size]
            }
        )

        # Compiles a function to test the model
        self.test_model = theano.function(
            inputs=[self.index],
            outputs=self.classifier.errors(self.y),
            givens= {
                self.x: X_test[self.index * self.batch_size:(self.index+1) * self.batch_size],
                self.y: Y_test[self.index * self.batch_size:(self.index+1) * self.batch_size]
            }
        )

        # Compiles function for both feedforward and backpropagation
        self.train_model = theano.function(
            inputs=[self.index],
            outputs=self.cost,
            updates=self.updates,
            givens= {
                self.x: X_train[self.index * self.batch_size:(self.index+1) * self.batch_size],
                self.y: Y_train[self.index * self.batch_size:(self.index+1) * self.batch_size]
            }
        )

        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is found
        improvement_threshold = 0.995 # a relative improvement of this much is considered significant
        validation_frequency = min(n_train_batches, patience // 2) # Check this many before validating

        best_validation_loss = np.inf
        best_iteration = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0 # Counts which epoch i am at
        done_looping = False # Parameter for early stopping

        print('Starting Training')

        # Runs all epochs or stops early
        while (epoch < self.n_epochs) and (not done_looping):
            epoch = epoch + 1
            # Train for every minibatch
            for minibatch_index in range(n_train_batches):

                minibatch_avg_cost = self.train_model(minibatch_index)
                iteration = (epoch - 1) * n_train_batches + minibatch_index # total number of minibatches iterated

                # Tests if it is time to validate
                if (iteration + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [
                        self.validate_model(i)
                        for i in range(n_valid_batches)
                    ]
                    this_validation_loss = np.mean(validation_losses)
                    self.valid_loss_graph.append((epoch, this_validation_loss))

                    this_validation_precision = self.precision_avg(valid[0],valid[1])
                    self.valid_precision_graph.append((
                        epoch,
                        this_validation_precision
                    ))

                    print(
                        'epoch {0}, minibatch {1}/{2}, validation error {3:.2f}%, validation precision {4:.2f}'.format(
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.,
                            this_validation_precision * 100.
                        )
                    )


                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        # improve patience if loss improvement is good enough
                        if (
                            (this_validation_loss < best_validation_loss *
                            improvement_threshold) and self.early_stopping
                        ):
                            patience = max(patience, iteration * patience_increase)

                        self.save_snapshot()

                        best_validation_loss = this_validation_loss
                        best_iteration = iteration

                        # test it on the test set
                        test_losses = [
                            self.test_model(i)
                            for i in range(n_test_batches)
                        ]
                        test_score = np.mean(test_losses)
                        self.test_loss_graph.append((epoch, test_score))

                        this_test_precision = self.precision_avg(test[0],test[1])
                        self.test_precision_graph.append((
                            epoch,
                            this_test_precision
                        ))

                        print(
                            '     epoch {0}, minibatch {1}/{2}, test error of best model {3}%, precision {4}'.format(
                                epoch,
                                minibatch_index + 1,
                                n_train_batches,
                                test_score * 100.,
                                this_test_precision * 100.
                            )
                        )

                if patience <= iteration and self.early_stopping:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print(
            ('Optimization complete. Best validation score of {0:f}% ' +
            'obtained at iteration {1}, with test performance {2} %').format(
                best_validation_loss * 100.,
                best_iteration + 1,
                test_score * 100.
            )
        )
        print('The model ran for {0:.2f} minutes'.format((end_time - start_time) / 60.))

        test_losses = [
            self.test_model(i)
            for i in range(n_test_batches)
        ]
        test_score = np.mean(test_losses)

        self.best_validation_loss = best_validation_loss
        self.load_snapshot() # Reloads the best classifier
        self.test_loss_best = test_score

    def predict(self, x):
        predict_theano = theano.function(
            inputs=[self.x],
            outputs=self.classifier.y_prediction
        )
        return predict_theano(x)

    def save_snapshot(self):
        self.snapshot = []
        for param in self.classifier.params:
            self.snapshot.append(param.eval())

    def load_snapshot(self):
        for new, old in zip(self.snapshot, self.classifier.params):
            old.set_value(new)

    def save_model(self, filename):
        """ Salva os pesos da rede em um arquivo .json

        :type filename: str
        :param filename: nome do arquivo para salvar os pesos. Deve terminar com ".json"

        """

        assert filename.split('.')[-1].lower() == 'json'

        data = {
            "structure": self.structure,
            "last_logistic": self.last_logistic,
            "cost_function": self.cost_function,
            "early_stopping":self.early_stopping,
            "layers": []
        }
        pairs = zip(self.classifier.params[0::2], self.classifier.params[1::2])
        for count, (W, b) in enumerate(pairs):
            data['layers'].append(
                {
                'layer{0:02}{1:02}'.format(count,count+1):  {
                        'W':W.eval().tolist(),
                        'b':b.eval().tolist()
                    }
                }
            )

        f = open(filename, 'w')
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.close()

    def load_model(self, filename):
        """ Carrega os pesos de um arquivo .json

        :type filename: str
        :param filename: nome do arquivo json que contém os pesos. Deve terminar com ".json"

        """

        assert filename.split('.')[-1].lower() == 'json'

        f = open(filename, 'r')
        data = json.load(f, parse_float=Decimal)
        f.close()

        assert self.structure == tuple(data['structure'])
        assert self.last_logistic == data['last_logistic']
        assert self.cost_function == data['cost_function']
        assert self.early_stopping == data['early_stopping']

        self.snapshot = []
        for layer in data['layers']:
            weights = list(layer.values())[0]
            self.snapshot.append(np.asarray(weights['W'], dtype=theano.config.floatX))
            self.snapshot.append(np.asarray(weights['b'], dtype=theano.config.floatX))

        self.load_snapshot()

    def save_graphs(self, filename):
        data = {
            "structure":self.structure,
            "last_logistic":self.last_logistic,
            "cost_function":self.cost_function,
            "early_stopping":self.early_stopping,
            "test_loss_best":self.test_loss_best,
            "valid_loss_graph": self.valid_loss_graph,
            "test_loss_graph": self.test_loss_graph,
            "valid_precision_graph":self.valid_precision_graph,
            "test_precision_graph":self.test_precision_graph,
        }
        f = open(filename, "w")
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.close()

    def precision_avg(self, x, y):
        precision = precision_score(
            y_true=y,
            y_pred=self.predict(x),
            labels=list(range(10)),
            average='macro'
        )
        return precision

    def precision_per_label(self, x, y):
        aux = classification_report(
            # Use the numpy array of the labels
            y_true=y,
            # Use the numpy array of the pixels
            y_pred=self.predict(x),
            labels=list(range(10)),
            output_dict=True
        )
        precision = {}
        for label, data in aux.items():
            if not (label in ['micro avg', 'macro avg', 'weighted avg']):
                precision[label] = data['precision']
        return precision
