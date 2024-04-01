from sys import path
import networkx as nx
#path.append(r"./node2vec_model/")
#from  node2vec_model import node2vec
from node2vec import Node2Vec


def feature_extraction(fb_df, fb_df_ghost, data):
    print("Extract node features from the graph...")
    # drop removable edges
    fb_df_partial = fb_df.drop(index=fb_df_ghost.index.values)

    # build graph
    G_data = nx.from_pandas_edgelist(fb_df_partial, "node_1", "node_2", create_using=nx.Graph())

    # Generate walks
    #node2vec1 = node2vec.Node2Vec(G_data, dimensions=100 ,  num_walks=50 , walk_length=16) #dimensions=100 ,  num_walks=50 , walk_length=16
    node2vec = Node2Vec(G_data, dimensions=100 ,  num_walks=50 , walk_length=16) 

    # train node2vec model
    n2w_model = node2vec.fit(window=10, min_count=1)

    x = [n2w_model.wv[str(i)] + n2w_model.wv[str(j)] for i, j in zip(data['node_1'], data['node_2'])]

    return x