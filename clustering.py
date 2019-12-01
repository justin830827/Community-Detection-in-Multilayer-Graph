#!/usr/bin/env python
# coding: utf-8

# In[4]:


import networkx as nx
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.sparse import csr_matrix
from sklearn import preprocessing

# In[5]:


def getU(lap,k):
    #print(lap)
    eig_val=eig_vec=None
    if type(lap).__module__!='numpy':
#        print(type(lap))
        eig_val, eig_vec = np.linalg.eig(lap.toarray())
    else:
        eig_val, eig_vec = np.linalg.eig(lap)
#    print(eig_val,eig_vec)
    top_indices = np.argsort(eig_val)[-k:]
    top_vecs = [eig_vec[:,i].transpose() for i in top_indices]
    
    return csr_matrix(np.vstack(top_vecs).T)


# In[6]:


def modifyLap(lap_list, U_list, alpha):
    n = lap_list[0].shape[0]
    uu_dash = [u.dot(u.T) for u in U_list]
    lap_sum = np.zeros((n,n))
    uu_sum = np.zeros((n,n))
    for L in lap_list:
        lap_sum = np.add(lap_sum,L.todense())
    for u in uu_dash:
        uu_sum = np.add(uu_sum,u.todense())
    return np.subtract(lap_sum, alpha * uu_sum) 


# In[7]:


def SCML(G,k,alpha):
    
    lap_list = [nx.normalized_laplacian_matrix(g, nodelist = sorted(g.nodes()))for g in G]

    #print(lap_list[1].toarray())
    U_list = [getU(l, k) for l in lap_list]
    Lmod = modifyLap(lap_list,U_list, alpha)

    U = getU(Lmod, k).real.todense()
       
    U = preprocessing.normalize(U, axis=1, norm='l1')
     

    centroids, labels = kmeans2(U,k,iter=20)
    print (centroids)
    print(labels)
    print(len(labels))
    return labels





