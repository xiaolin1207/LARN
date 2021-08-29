import numpy as np
import random
import scipy.sparse as sp
import torch
import gc
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx
from sklearn import metrics
rng = np.random.RandomState(seed=72)



def load_data():
    print('Loading dataset...')
    labels = sp.load_npz("data/label.npz").toarray()
    graph = nx.read_edgelist("data/dblp_multi.edgelist", delimiter="\t")
    features = torch.FloatTensor(sp.load_npz("data/lsi.npz").toarray())

    label_emb = np.load("data/label_embed_4_50.npy")
    label_emb = torch.from_numpy(label_emb.astype(np.float32))

    adj_lists = graph._adj
    h = []
    dict1 = defaultdict()
    for i in range(len(adj_lists)):
        h.append(list(adj_lists[str(i)]))
    for i in range(len(adj_lists)):
        for j in range(len(h[i])):
            h[i][j] = int(h[i][j])
    z = 0
    neighbor = []
    for i in adj_lists.keys():
        h[z] = np.insert(h[z], 0, z)
        neighbor.append(h[z])
        dict1.setdefault(int(i), list(h[z]))
        z = z + 1
    print("done")
    max_l = 10
    for i in range(len(adj_lists)):
        if len(neighbor[i]) >= max_l:
            neighbor[i] = random.sample(list(neighbor[i]), max_l)
        else:

            pad = neighbor[i][0] * np.ones(max_l - len(neighbor[i]))
            neighbor[i] = np.concatenate((neighbor[i], pad), axis=0)
    neighbors = np.int64(np.array(neighbor))
    neighbors = torch.from_numpy(neighbors)


    return features, labels, label_emb, neighbors


def get_metrics(y, y_pre):
    hamming_loss = metrics.hamming_loss(y, y_pre)
    micro_precision = metrics.precision_score(y, y_pre, average='micro')
    micro_f1 = metrics.f1_score(y, y_pre, average='micro')
    micro_recall = metrics.recall_score(y, y_pre, average='micro')
    return hamming_loss, micro_f1, micro_precision, micro_recall
