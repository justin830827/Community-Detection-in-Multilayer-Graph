import networkx as nx
import numpy as np
from algo import SCML
from algo import onelayer
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import v_measure_score as nmi
from sklearn.metrics import silhouette_score
from sklearn import metrics


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def init_graph():
    path = './data/aucs_nodelist.txt'
    g = nx.Graph()
    with open(path) as f:
        for line in f:
            line = line.strip().split(',')
            if line[1] == 'NA':
                continue
            else:
                g.add_node(line[0])
    return g


def get_truth():
    truth = []
    na_list = []
    path = './data/aucs_nodelist.txt'
    with open(path) as f:
        for line in f:
            line = line.strip().split(',')
            t = line[1]
            if t == 'NA':
                na_list.append(line[0])
            else:
                truth.append(int(t[-1])-1)
    return truth, na_list


def plot_elbow(x, y, name):
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, '-o')
    plt.xlabel('Number of cluster' if name == "Selection of k" else "alpha")
    plt.ylabel('Sum of Square Error' if name ==
               "Selection of k" else "Density")
    plt.title(name)
    plt.savefig("./results/{}.png".format(name))


def compute_density(g, partition):
    density = [nx.density(g.subgraph(p)) for p in partition]
    return np.mean(density)


def get_partition(labels, nodes):
    num_part = len(set(labels))
    part = [[] for i in range(num_part)]
    for i in range(len(labels)):
        part[labels[i]].append(nodes[i])
    return part


def get_score(graph_list, partitions):
    density = np.zeros((len(graph_list), len(partitions)))
    # conductance = np.zeros((len(graph_list), len(partitions)))
    for i, g in enumerate(graph_list):
        for k in range(len(partitions)):
            g_sub = g.subgraph(partitions[k])
            density[i, k] = nx.density(g_sub)
    # print("Density found for each cluster across all layers:")
    # print(np.mean(density, axis=0))
    # print("Conductance found for each cluster across all layers:")
    # print(np.mean(conductance, axis=0))
    mean_density = np.mean(np.mean(density, axis=0))
    # mean_conductance = np.mean(np.mean(conductance, axis=0))
    return mean_density


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
    truth, na = get_truth()

    # Load data into graph
    print("--------------------------------------------------Load multilayers graph--------------------------------------------------")
    with open(path) as f:
        for line in f:
            line = line.strip().split(',')
            name = line[2]
            if line[0] in na or line[1] in na:
                continue
            else:
                table[name].add_edge(line[0], line[1])
    for name, graph in table.items():
        print("\nGraph: {}".format(name))
        print("\tNumber of nodes: {}".format(nx.number_of_nodes(graph)))
        print("\tNumber of edges: {}".format(nx.number_of_edges(graph)))

    graph_list = [lunch, work,coauthor,leisure]
    node_list = list(lunch.nodes)

    # # Tunning k
    # print("--------------------------------------------------Perform k clusters selection--------------------------------------------------")
    # sse_list = []
    # range_k = np.arange(2, 15)
    # for k in range_k:
    #     labels, sse = SCML(graph_list, k, 0.5)
    #     score = silhouette_score(matrix, labels, random_state=42)
    #     print("Number of clusters k = {}".format(k),
    #           ",Silhouette Score = {}".format(round(score, 5)))
    #     sse_list.append(sse)

    # # Plot elbow method for k
    # plot_elbow(range_k, sse_list, "Selection of k")

    # Tunning alpha
    print("--------------------------------------------------Perform alpha selection--------------------------------------------------")
    range_a = np.arange(0.2, 1.1, 0.1)
    den = []
    for alpha in range_a:
        labels, sse = SCML(graph_list, 8, alpha)
        partitions = get_partition(labels, node_list)
        density = get_score(graph_list, partitions)
        den.append(density)
        print("Alpha = {}".format(round(alpha, 1)),
              ", Density = {}".format(density))

    # Plot elbow method for alpha
    plot_elbow(range_a, den, "Selection of alpha")

    # Select the best model
    labels, sse = SCML(graph_list, 8, 0.6)
    partitions = get_partition(labels, node_list)
    # print(labels)
    # print(np.array(truth))

    print("NMI: {}".format(nmi(truth, labels)))
    purity = purity_score(truth, labels)
    print("Purity: {}".format(purity))
    print('====================================one layer===================================')
    print('lunch')
    labels,sse=onelayer(lunch,8)
    print(labels)
    print("NMI: {}".format(nmi(truth, labels)))
    purity = purity_score(truth, labels)
    print("Purity: {}".format(purity))

    print('facebook')
    labels,sse=onelayer(facebook,8)
    print(labels)
    print("NMI: {}".format(nmi(truth, labels)))
    purity = purity_score(truth, labels)
    print("Purity: {}".format(purity))

    print('work')
    labels,sse=onelayer(work,8)
    print(labels)
    print("NMI: {}".format(nmi(truth, labels)))
    purity = purity_score(truth, labels)
    print("Purity: {}".format(purity))

    print('coauthor')
    labels,sse=onelayer(coauthor,8)
    print(labels)
    print("NMI: {}".format(nmi(truth, labels)))
    purity = purity_score(truth, labels)
    print("Purity: {}".format(purity))

    print('leisure')
    labels,sse=onelayer(leisure,8)
    print(labels)
    print("NMI: {}".format(nmi(truth, labels)))
    purity = purity_score(truth, labels)
    print("Purity: {}".format(purity))


if __name__ == "__main__":
    main()
