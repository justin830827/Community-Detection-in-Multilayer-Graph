import networkx as nx
import numpy as np
from clustering import SCML



def main():
    path = './data/aucs_edgelist.txt'

    # declare each layer's graph
    lunch = nx.Graph()
    facebook = nx.Graph()
    leisure = nx.Graph()
    work = nx.Graph()
    coauthor = nx.Graph()
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
        print("nodes in Graph {}: {}".format(name, nx.number_of_nodes(graph)))
    graph_list=[lunch,facebook,leisure,work,coauthor]
    label=SCML(graph_list,3,0.5)
    print (label)

if __name__ == "__main__":
    main()
