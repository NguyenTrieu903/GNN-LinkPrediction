import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    # load nodes details
    print("Load data...")
    with open("raw_data/fb-pages-food.nodes", "rb") as f:
        fb_nodes = f.read().splitlines() 

    # load edges (or links)
    with open("raw_data/fb-pages-food.edges") as f:
        fb_links = f.read().splitlines() 

    print("Capture nodes in 2 separate lists...")
    node_list_1 = []
    node_list_2 = []

    for i in fb_links:
        node_list_1.append(i.split(',')[0])
        node_list_2.append(i.split(',')[1])

    fb_df = pd.DataFrame({'node_1': node_list_1, 'node_2': node_list_2})

    return fb_df, node_list_1, node_list_2

def plot_graph(fb_df):
  
  # create graph
  G = nx.from_pandas_edgelist(fb_df, "node_1", "node_2", create_using=nx.Graph())

  # plot graph
  plt.figure(figsize=(10,10))
  
  pos = nx.random_layout(G, np.random.seed(23))
  nx.draw(G, with_labels=False,  pos = pos, node_size = 40, alpha = 0.6, width = 0.7)

  plt.show()
  return G