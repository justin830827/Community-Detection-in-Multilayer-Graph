# Graph Data Mining

This Capstone Project aims to solve community detection in multilayers graph.

## Environment to test the script

This project has been tested on `CSC591_ADBI_v3` VCL environment.

## Packages requirments

Please ensure the packages have been installed beforehand, or run the following command to install:

```
pip3 install -r requirements.txt
```

## How to run the script

After download the zip, first unzip the zip file and get into the folder.

```
unzip capstone.zip
cd capstone
```

Once the path is under `capstone`, please run the command in following format.

```
python3 main.py
```

After the program complete, the results of program will display on terminal and plots will be saved in `./results`.

## Description of Dataset

This dataset is downloaded from [link](http://multilayer.it.uu.se/datasets.html). In this graph, the multiple layers represent relationships between 61 employees of a University department in five different aspects: (i) coworking, (ii) having lunch together, (iii) Facebook friendship, (iv) o✏ine friendship (having fun together), and (v) coauthor-ship.

#### Dataset Name: AUCS

#### Type: Multi-layers Graph

#### Layers

1. Facebook,UNDIRECTED
2. Lunch,UNDIRECTED
3. Coauthor,UNDIRECTED
4. Leisure,UNDIRECTED
5. Work,UNDIRECTED

#### ACTOR ATTRIBUTES

1. ResearchGroup,STRING
2. Role,STRING

## Results

```console
--------------------Load multilayers graph--------------------

Graph: lunch
        Number of nodes: 55
        Number of edges: 176

Graph: facebook
        Number of nodes: 55
        Number of edges: 116

Graph: leisure
        Number of nodes: 55
        Number of edges: 88

Graph: work
        Number of nodes: 55
        Number of edges: 155

Graph: coauthor
        Number of nodes: 55
        Number of edges: 21
--------------------Perform alpha selection-------------------

Alpha = 0.2
        Density = 0.06324630230880231
        NMI = 0.28437039334841613

Alpha = 0.3
        Density = 0.05426587301587302
        NMI = 0.22851191671984766

Alpha = 0.4
        Density = 0.074259768009768
        NMI = 0.2724979205931001

Alpha = 0.5
        Density = 0.0838045634920635
        NMI = 0.24702647831111396

Alpha = 0.6
        Density = 0.05803571428571429
        NMI = 0.24631830263834753

Alpha = 0.7
        Density = 0.057311958874458876
        NMI = 0.26013012383823736

Alpha = 0.8
        Density = 0.059573412698412695
        NMI = 0.27394942614468387

Alpha = 0.9
        Density = 0.06098935786435787
        NMI = 0.2531115624498492

Alpha = 1.0
        Density = 0.08007756132756133
        NMI = 0.2515334583668016
--------------------Multilayer Result--------------------
NMI: 0.28437039334841613
Purity: 0.34545454545454546

--------------------Single layer Result--------------------

Layer: lunch
        NMI: 0.4078232316115382
        Purity: 0.4909090909090909

Layer: facebook
        NMI: 0.23710067652109873
        Purity: 0.2909090909090909

Layer: leisure
        NMI: 0.36515019997995346
        Purity: 0.45454545454545453

Layer: work
        NMI: 0.3601914640378153
        Purity: 0.4727272727272727

Layer: coauthor
        NMI: 0.30730821666442587
        Purity: 0.4
```

## Project Member:

1. Wen-Han Hu (whu24)
2. Yang-Kai Chou (ychou3)

## Reference

1. Dong, Xiaowen, et al. "Clustering on multi-layer graphs via subspace analysis on Grassmann manifolds." IEEE Transactions on signal processing 62.4 (2013): 905-918.
2. Kim, Jungeun, and Jae-Gil Lee. "Community detection in multi-layer graphs: A survey." ACM SIGMOD Record 44.3 (2015): 37-48.
3. Zhang, Pan. "Evaluating accuracy of community detection using the relative normalized mutual information." Journal of Statistical Mechanics: Theory and Experiment 2015.11 (2015): P11006.
