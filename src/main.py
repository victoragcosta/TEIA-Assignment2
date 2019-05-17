from mnist import MNIST
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import os.path

from multilayer_perceptron_classifier import *

# Padrão X -> imagens ; Y -> labels

load_models = True

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

print('Instanciando classificador')

# A partir daqui temos imagens em X_training e X_test e labels em Y_training e Y_test
classificador = MultilayerPerceptronClassifier(
    structure=(28*28, 500, 100, 10),
    n_epochs=1,
    batch_size=50,
    last_logistic=True,
    cost_function='negative_log_likelihood',
    early_stopping=True
)

if not load_models:
    print('Iniciando treinamento')
    classificador.fit((X_training,Y_training), (X_valid, Y_valid), (X_test, Y_test))
    print('Salvando modelo')
    classificador.save_model("data/models/logistic-negative_log_likelihood.json")
    print('Salvando estatísticas')
    classificador.save_graphs("data/graphs/logistic-negative_log_likelihood.json")
else:
    print('Carregando modelo já treinado')
    classificador.load_model("data/models/logistic-negative_log_likelihood.json")

for i in range(10):
    random_example = np.random.randint(low=0, high=X_test.shape[0])
    print('Testando exemplo aleatório número {0} do teste'.format(random_example))
    X = X_test[random_example:random_example+1]
    print('Previsto:', classificador.predict(X))
    print('Esperado:', Y_test.item((random_example)))
    plt.matshow(np.reshape(X,(28,28)))
    plt.show()
