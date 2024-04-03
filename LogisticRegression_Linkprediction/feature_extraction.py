from sys import path
import networkx as nx
path.append(r"../node2vec/src/")
import node2vec
from gensim.models import Word2Vec
import numpy as np


def feature_extraction(fb_df, fb_df_ghost, data):
    print("Extract node features from the graph...")
    fb_df_partial = fb_df.drop(index=fb_df_ghost.index.values)
    G_data = nx.Graph()
    G_data.add_weighted_edges_from(np.concatenate([fb_df_partial, np.ones(shape=[fb_df_partial.shape[0], 1], dtype=np.int8)], axis=1))
    G = node2vec.Graph(G_data, is_directed=False, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=50, walk_length=16)
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=100, window=10, min_count=0, sg=1, workers=8, iter=1)
    x = []
    for i, j in zip(data['node_1'], data['node_2']):
        node1_embedding = model.wv[str(i)]
        node2_embedding = model.wv[str(j)]
        combined_embedding = node1_embedding + node2_embedding
        x.append(combined_embedding)

    return x