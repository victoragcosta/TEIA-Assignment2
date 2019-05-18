# libraries
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rnd


def plot_graph(files_name, metric_name='valid_precision_graph'):

    list_jsons = []
    graph_obj = {}

    for file_name in files_name:
        with open('../data/graphs/{file_name}.json'.format(file_name=file_name), 'r') as myfile:
            data = myfile.read()
            json_obj = json.loads(data)
            list_jsons.append(json_obj)

            list_y = json_obj[metric_name]
            list_y = list(map(lambda y: y[1], list_y))
            graph_obj[file_name] = list_y

    list_x = list_jsons[0][metric_name]
    graph_obj['x'] = list(map(lambda x: x[0], list_x))


    # Data
    df = pd.DataFrame(graph_obj)

    # multiple line plot
    for count, file_name in enumerate(files_name):
        rnd.seed(count)
        color = (rnd.uniform(0, 1), rnd.uniform(0, 1), rnd.uniform(0, 1))
        plt.plot('x', file_name, data=df, linewidth=2, color=color, label=file_name.replace('_',' ').replace('-',' ').title())

    plt.legend()
    plt.title(metric_name.replace('_',' ').replace('-',' ').title())
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')

    plt.savefig("../data/graphs/{name}.png".format(name=metric_name))

    plt.show()


def plot_graphs(files_name, metrics_name):
    for metric_name in metrics_name:
        plot_graph(files_name, metric_name)


# Plot all graphs
graphs_metrics = ["valid_loss_graph", "test_loss_graph", "valid_precision_graph", "test_precision_graph"]
files = ["logistic-negative_log_likelihood",
         ]

plot_graphs(files, graphs_metrics)
