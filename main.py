import networkx as nx
import numpy as np
from algo import SCML
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import *


def init_graph():
    path = './data/aucs_nodelist.txt'
    g = nx.Graph()
    with open(path) as f:
        for line in f:
            line = line.strip().split(',')
            g.add_node(line[0])
    return g


def plot_elbow_k(k, sse):
    plt.figure(figsize=(8, 8))
    plt.plot(k, sse, '-o')
    plt.xlabel('Number of cluster')
    plt.ylabel('Sum of Square Error')
    plt.savefig("./results/kmeans elbow method.png")


def main():
    path = './data/aucs_edgelist.txt'

    # Declare each layer's graph
    lunch = init_graph()
    facebook = init_graph()
    leisure = init_graph()
    work = init_graph()
    coauthor = init_graph()
    table = {
        'lunch': lunch,
        'facebook': facebook,
        'leisure': leisure,
        'work': work,
        'coauthor': coauthor,
    }

    # Load data into graph
    print("--------------------------------------------------Load multilayers graph--------------------------------------------------")
    with open(path) as f:
        for line in f:
            line = line.strip().split(',')
            name = line[2]
            table[name].add_edge(line[0], line[1])
    for name, graph in table.items():
        print("\nGraph: {}".format(name))
        print("\tNumber of nodes: {}".format(nx.number_of_nodes(graph)))
        print("\tNumber of edges: {}".format(nx.number_of_edges(graph)))

    graph_list = [lunch, facebook, leisure, work, coauthor]

    # Tunning k and alpha
    print("--------------------------------------------------Perform k clusters selection--------------------------------------------------")
    sse_list = []
    range_k = np.arange(2, 15)
    for k in range_k:
        labels, matrix, sse = SCML(graph_list, k, 0.5)
        score = silhouette_score(matrix, labels, random_state=42)
        print("Clusters = {}".format(k),
              ",Silhouette Score = {}".format(round(score, 5)))
        sse_list.append(sse)

    # Plot elbow method
    plot_elbow_k(range_k, sse_list)

    # Tunning alpha
    range_a = np.arange(0.2,1.1,0.1)
    for alpha in range_a:
        label = SCML(graph_list, 8, alpha)
        print(label)
        print(Counter(label))

    # Select the best model
    labels, matrix, sse = SCML(graph_list, 8, 0.5)

    # Evaluation of clustring
    print("--------------------------------------------------Clustering evaluation--------------------------------------------------")
    db_index = round(davies_bouldin_score(matrix, labels), 5)
    ch_index = round(calinski_harabasz_score(matrix, labels), 5)
    s_coef = round(silhouette_score(matrix, labels, metric='euclidean'), 5)
    print("Silhouette Score: {}".format(s_coef))
    print("Davies-Bouldin Score: {}".format(db_index))
    print("Calinski-Harabaz Score: {}".format(ch_index))


if __name__ == "__main__":
    main()
