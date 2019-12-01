import networkx as nx
import numpy as np
from clustering import SCML
import matplotlib.pyplot as plt
from collections import Counter
def init_graph():
    path = './data/aucs_nodelist.txt'
    g = nx.Graph()
    with open(path) as f:
        for line in f:
            line = line.strip().split(',')
            g.add_node(line[0])
    return g


def main():
    path = './data/aucs_edgelist.txt'

    # declare each layer's graph
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

    with open(path) as f:
        for line in f:
            line = line.strip().split(',')
            name = line[2]
            table[name].add_edge(line[0], line[1])
    for name, graph in table.items():
        print("\nGraph: {}:".format(name))
        print("\tNumber of nodes: {}".format(nx.number_of_nodes(graph)))
        print("\tNumber of edges: {}".format(nx.number_of_edges(graph)))
    graph_list=[lunch,facebook,leisure,work,coauthor]

    #for i in range(2,11):
    #    label=SCML(graph_list,i,0.5)
        
    #    print (label)
    label=SCML(graph_list,7,0.5)
    print(label)
    print(Counter(label))
    labels, values = zip(*Counter(label).items())

    indexes = np.arange(len(labels))
    width = 1

    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.show()

if __name__ == "__main__":
    main()
