from mnist import MNIST
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import os.path

from multilayer_perceptron_classifier import *

# Padrão X -> imagens ; Y -> labels

interactive = False

# Extração de dados (biblioteca python-mnist)
mnist_data = MNIST(os.path.abspath('data'))
X, Y = mnist_data.load_training()
X_training = np.reshape(np.asarray(X, dtype=np.uint8), (60000, 28 * 28))
Y_training = np.reshape(np.asarray(Y, dtype=np.uint8), (60000,))
X, Y = mnist_data.load_testing()
X_test = np.reshape(np.asarray(X, dtype=np.uint8), (10000, 28 * 28))
Y_test = np.reshape(np.asarray(Y, dtype=np.uint8), (10000,))

# Seleciona parte de treinamento como validação
size_validation = 1000
X_valid = X_training[0:size_validation,:]
Y_valid = Y_training[0:size_validation]
X_training = X_training[size_validation:,:]
Y_training = Y_training[size_validation:]

a_testar = [
    (
        'logistic-negative_log_likelihood-L1L2',
        {
            "structure":(28*28, 500, 100, 10),
            "n_epochs":200,
            "batch_size":50,
            "last_logistic":True,
            "cost_function":'negative_log_likelihood',
            "early_stopping":True,
            "L1_reg": 0.001,
            "L2_reg": 0.0001
        }
    ),
    (
        'logistic-negative_log_likelihood-L1',
        {
            "structure":(28*28, 500, 100, 10),
            "n_epochs":200,
            "batch_size":50,
            "last_logistic":True,
            "cost_function":'negative_log_likelihood',
            "early_stopping":True,
            "L1_reg": 0.0,
            "L2_reg": 0.0001
        }
    ),
    (
        'logistic-negative_log_likelihood',
        {
            "structure":(28*28, 500, 100, 10),
            "n_epochs":200,
            "batch_size":50,
            "last_logistic":True,
            "cost_function":'negative_log_likelihood',
            "early_stopping":True,
            "L1_reg": 0.0,
            "L2_reg": 0.0001
        }
    ),
    (
        'logistic-negative_log_likelihood-noL1L2-2layer',
        {
            "structure":(28*28, 500, 100, 10),
            "n_epochs":200,
            "batch_size":50,
            "last_logistic":True,
            "cost_function":'negative_log_likelihood',
            "early_stopping":True,
            "L1_reg": 0.0,
            "L2_reg": 0.0
        }
    ),
    (
        'logistic-negative_log_likelihood-noL1L2',
        {
            "structure":(28*28, 500, 10),
            "n_epochs":200,
            "batch_size":50,
            "last_logistic":True,
            "cost_function":'negative_log_likelihood',
            "early_stopping":True,
            "L1_reg": 0.0,
            "L2_reg": 0.0
        }
    ),
]

print('Instanciando classificador')

# A partir daqui temos imagens em X_training e X_test e labels em Y_training e Y_test
for nome, teste in a_testar:
    classificador = MultilayerPerceptronClassifier(**teste)

    try:
        print('Carregando modelo {0}'.format(nome))
        classificador.load_model("data/models/{0}.json".format(nome))
    except:
        print('Modelo {0} ainda não foi treinado'.format(nome))
        print('Iniciando treinamento')
        classificador.fit((X_training,Y_training), (X_valid, Y_valid), (X_test, Y_test))
        print('Salvando modelo')
        classificador.save_model("data/models/{0}.json".format(nome))
        print('Salvando estatísticas')
        classificador.save_graphs("data/graphs/{0}.json".format(nome))

    # for i in range(10):
    #     random_example = np.random.randint(low=0, high=X_test.shape[0])
    #     print('Testando exemplo aleatório número {0} do teste'.format(random_example))
    #     X = X_test[random_example:random_example+1]
    #     print('Previsto:', classificador.predict(X))
    #     print('Esperado:', Y_test.item((random_example)))
    #     if interactive:
    #         plt.matshow(np.reshape(X,(28,28)))
    #         plt.show()
