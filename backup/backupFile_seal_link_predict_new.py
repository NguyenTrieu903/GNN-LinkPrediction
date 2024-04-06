def learning_embedding_at_node(positive, negative, network_size, test_ratio, dimension, network_type, index,
                               negative_injection=True):
    """
    :param positive: ndarray, from 'load_data', all positive edges
    :param negative: ndarray, from 'load_data', all negative edges
    :param network_size: scalar, nodes size in the network
    :param test_ratio: proportion of the test set
    :param dimension: size of the node2vec
    :param network_type: directed or undirected
    :param negative_injection: add negative edges to learn word embedding
    :return:
    """
    print("learning embedding...")
    # used training data only
    test_size = int(test_ratio * positive.shape[0])
    train_posi, train_nega = positive[:-test_size], negative[:-test_size]
    # negative injection
    A = nx.Graph() if network_type == 0 else nx.DiGraph()
    # So if train_posi is an array containing the edges of the graph,
    # the line of code will add these edges to graph A and assign a weight of 1 to each edge.
    A.add_weighted_edges_from(
        np.concatenate([train_posi, np.ones(shape=[train_posi.shape[0], 1], dtype=np.int8)], axis=1))
    if negative_injection:
        A.add_weighted_edges_from(
            np.concatenate([train_nega, np.ones(shape=[train_nega.shape[0], 1], dtype=np.int8)], axis=1))
    # node2vec
    G = node2vec.graph(A, is_directed=False if network_type == 0 else True, p=1, q=1)
    # G = Graph(A, is_directed=False if network_type == 0 else True, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=dimension, window=10, min_count=0, sg=1, workers=8, iter=1)
    wv = model.wv
    embedding_feature, empty_indices, avg_feature = np.zeros([network_size, dimension]), [], 0
    for i in range(network_size):
        print("i: ", i)
        if str(i) in wv:
            embedding_feature[i] = wv.word_vec(str(i))
            print("embedding_feature[{}]: {}".format(i, embedding_feature[i]))
            avg_feature += wv.word_vec(str(i))
        else:
            empty_indices.append(i)
    embedding_feature[empty_indices] = avg_feature / (network_size - len(empty_indices))
    print("embedding feature shape: ", embedding_feature.shape)
    return embedding_feature

def link2subgraph_at_one_node(node_pair, positive, negative, nodes_size, test_ratio, hop, network_type,
                              max_neighbors=100):
    """
    :param positive: ndarray, from 'load_data', all positive edges
    :param negative: ndarray, from 'load_data', all negative edges
    :param nodes_size: int, scalar, nodes size in the network
    :param test_ratio: float, scalar, proportion of the test set
    :param hop: option: 0, 1, 2, ..., or 'auto'
    :param network_type: directed or undirected
    :param max_neighbors:
    :return:
    """
    print("extract enclosing subgraph...")
    test_size = int(len(positive) * test_ratio)
    train_pos, test_pos = positive[:-test_size], positive[-test_size:]
    train_neg, test_neg = negative[:-test_size], negative[-test_size:]

    A = np.zeros([nodes_size, nodes_size], dtype=np.uint8)
    # create adjacency matrix of graph
    A[train_pos[:, 0], train_pos[:, 1]] = 1
    if network_type == 0:
        A[train_pos[:, 1], train_pos[:, 0]] = 1

    def calculate_auc(scores, test_pos, test_neg):
        pos_scores = scores[test_pos[:, 0], test_pos[:, 1]]
        neg_scores = scores[test_neg[:, 0], test_neg[:, 1]]
        s = np.concatenate([pos_scores, neg_scores])
        y = np.concatenate([np.ones(len(test_pos), dtype=np.int8), np.zeros(len(test_neg), dtype=np.int8)])
        assert len(s) == len(y)
        auc = metrics.roc_auc_score(y_true=y, y_score=s)
        return auc

    # determine the h value
    # doan code dung de tinh toan do tuong dong giua cac nut trong do thi dua tren thong tin cau truc.
    # 2 do tuong dong duoc tinh o day la: cn(Common Neighbors) va aa(Adamic-Adar).... Sau do so sanh xe auc cua phuong phap nao lon hon thi su dung phuong phap do
    if hop == "auto":
        def cn():
            return np.matmul(A, A)

        def aa():
            A_ = A / np.log(A.sum(axis=1))
            A_[np.isnan(A_)] = 0
            A_[np.isinf(A_)] = 0
            return A.dot(A_)

        cn_scores, aa_scores = cn(), aa()
        cn_auc = calculate_auc(cn_scores, test_pos, test_neg)
        aa_auc = calculate_auc(aa_scores, test_pos, test_neg)
        if cn_auc > aa_auc:
            print("cn(first order heuristic): %f > aa(second order heuristic) %f." % (cn_auc, aa_auc))
            hop = 1
        else:
            print("aa(second order heuristic): %f > cn(first order heuristic) %f. " % (aa_auc, cn_auc))
            hop = 2

    print("hop = %s." % hop)

    # tao mot do thi moi G de tinh toan, di tim cac subgraph
    # extract the subgraph for (positive, negative)
    G = nx.Graph() if network_type == 0 else nx.DiGraph()
    G.add_nodes_from(set(positive[:, 0]) | set(positive[:, 1]) | set(negative[:, 0]) | set(negative[:, 1]))
    # G.add_nodes_from(set(sum(positive.tolist(), [])) | set(sum(negative.tolist(), [])))
    G.add_edges_from(train_pos)

    graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes = [], [], [], [], []
    sub_nodes, sub_adj, vertex_tag = extract_subgraph(node_pair, G, A, hop, network_type, max_neighbors)
    graphs_adj.append(sub_adj)
    vertex_tags.append(vertex_tag)
    node_size_list.append(len(vertex_tag))
    sub_graphs_nodes.append(sub_nodes)
    # for graph_label, data in enumerate([negative, positive]):
    #     print("for %s. " % "negative" if graph_label == 0 else "positive")
    # for node_pair in tqdm(data):

    # assert len(graphs_adj) == len(vertex_tags) == len(node_size_list)
    # dung de tao ma tran nhan cho cac canh trong do thi. tao thanh ma tran co 1 cot va so hang la tong so cac canh trong do thi.
    # Ket qua la mot ma tran nhan, trong do moi hang dai dien cho 1 canh trong do thi va mot nhan tuong ung.
    labels = np.concatenate([np.zeros(len(negative), dtype=np.uint8), np.ones(len(positive), dtype=np.uint8)]).reshape(
        -1, 1)

    # vertex_tags_set = list(set(sum(vertex_tags, [])))
    vertex_tags_set = set()
    for tags in vertex_tags:
        vertex_tags_set = vertex_tags_set.union(set(tags))
    vertex_tags_set = list(vertex_tags_set)
    tags_size = len(vertex_tags_set)
    print("tight the vertices tags.")
    # kiem tra xem tat ca cac phan tu trong vertex_tags_set co tao thanh mot chuoi so nguyen lien tuc tu 0 den len(vertex_tags_set-1) hay khong.
    # Dieu nay dung de dam bao tinh day du va dung dan cua cac nhan duoc gan cho cac nut trong do thi.
    if set(range(len(vertex_tags_set))) != set(vertex_tags_set):
        vertex_map = dict([(x, vertex_tags_set.index(x)) for x in vertex_tags_set])
        for index, graph_tag in tqdm(enumerate(vertex_tags)):
            vertex_tags[index] = list(itemgetter(*graph_tag)(vertex_map))
    return graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes, tags_size

def extract_subgraph_at_node(node, G, A, hop, network_type, max_neighbors):
    """
    :param node:  vertex to extract subgraph around
    :param G:  nx object from the positive edges
    :param A:  equivalent to the G, adj matrix of G
    :param hop: number of hops to extend the subgraph
    :param network_type: type of the network (directed or undirected)
    :param max_neighbors: maximum number of neighbors to include in the subgraph
    :return:
        sub_graph_nodes: nodes in the extracted subgraph
        sub_graph_adj: adjacency matrix of the subgraph
        vertex_tag: node type information from the labeling algorithm
    """
    sub_graph_nodes = set([node])
    for i in range(int(hop)):
        neighbors = list(nx.neighbors(G, node))
        if len(sub_graph_nodes) + len(neighbors) < max_neighbors:
            sub_graph_nodes = sub_graph_nodes.union(neighbors)
        else:
            np.random.shuffle(neighbors)
            sub_graph_nodes = sub_graph_nodes.union(neighbors[:max_neighbors - len(sub_graph_nodes)])
            break

    sub_graph_adj = A[list(sub_graph_nodes), :][:, list(sub_graph_nodes)]

    # labeling (coloring/tagging)
    vertex_tag = node_labeling(sub_graph_adj, network_type)
    return list(sub_graph_nodes), sub_graph_adj, vertex_tag

def create_input_for_gnn_fly_node_pair(graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes,
                                       embedding_feature, explicit_feature, tags_size):
    print("create input for gnn on fly, (skipping I/O operation)")
    # graphs, nodes_size_list, labels = data["graphs"], data["nodes_size_list"], data["labels"]

    # 1 - prepare Y
    # dung de tao ma tran nhan Y trong do moi phan tu co gia tri 1 neu tuong ung voi mot canh co nhan 1, va co gia tri 0 neu tuong ung voi 1 canh co nhan 0
    Y = np.where(np.reshape(labels, [-1, 1]) == 1, 1, 0)
    print("positive examples: %d, negative examples: %d." % (np.sum(Y == 0), np.sum(Y == 1)))
    # 2 - prepare A_title
    # graphs_adj is A_title in the formular of Graph Convolution layer
    # add eye to A_title
    # code dung de them ma tran don vi vao moi ma tran ke. np.eye dung de tao ma tran don vi
    for index, x in enumerate(graphs_adj):
        graphs_adj[index] = x + np.eye(x.shape[0], dtype=np.uint8)
    # 3 - prepare D_inverse
    D_inverse = []
    for x in graphs_adj:
        # su dung de tinh ma tran nghich dao cua ma tran duong cheo. np.sum(x, axis=1) -> tinh tong moi hang cua ma tran
        # np.diag -> tao mot ma tran duong cheo tu mang bang cach su dung np.diag()
        # np.linalg.inv -> tinh ma tran nghich dao cua ma tran duong cheo nay.
        D_inverse.append(np.linalg.inv(np.diag(np.sum(x, axis=1))))
    # 4 - prepare X
    X, initial_feature_channels = [], 0

    # Target: chuyen doi mot mang cac nhan lop thanh mot dang ma hoa one-hot. Một ma trận one-hot với mỗi hàng biểu diễn một nhãn lớp,
    # trong đó chỉ có một phần tử bằng 1 và tất cả các phần tử khác bằng 0
    def convert_to_one_hot(y, C):
        return np.eye(C, dtype=np.uint8)[y.reshape(-1)]

    # vertex_tags la mot list cac nhan cac dinh trong do thi. Neu khac None thi chay qua cac vertex_tag va one_hot chung.
    if vertex_tags is not None:
        initial_feature_channels = tags_size
        print("X: one-hot vertex tag, tag size %d." % initial_feature_channels)
        for tag in vertex_tags:
            x = convert_to_one_hot(np.array(tag), initial_feature_channels)
            X.append(x)
    # Nguoc lai se chuan hoa roi them vao mang X.
    else:
        print("X: normalized node degree.")
        for graph in graphs_adj:
            degree_total = np.sum(graph, axis=1)
            X.append(np.divide(degree_total, np.sum(degree_total)).reshape(-1, 1))
        initial_feature_channels = 1
    X = np.array(X)
    # doan code xay dung cac embedding features cho cac dinh trong do thi bang cach ket hop cac dac trung hien co voi cac dac trung nhung neu chung
    # co san.
    if embedding_feature is not None:
        # print("embedding feature has considered")
        # # build embedding for enclosing sub-graph
        # sub_graph_emb = []
        # for sub_nodes in sub_graphs_nodes:
        #     sub_graph_emb.append(embedding_feature[sub_nodes])
        # for i in range(len(X)):
        #     X[i] = np.concatenate([X[i], sub_graph_emb[i]], axis=1)
        # # so luong kenh dac trung ban dau duoc cap nhat thanh so luong kenh dac trung dau tien trong X.
        # initial_feature_channels = len(X[0][0])
        print("embedding feature has been considered")
        print(len(sub_graphs_nodes))
        # Build embedding for the node pair
        node1_emb = embedding_feature[sub_graphs_nodes[0]]
        node2_emb = embedding_feature[sub_graphs_nodes[1]]

        # Concatenate node embeddings with node features
        X_node1 = np.concatenate([X[0], node1_emb], axis=1)
        X_node2 = np.concatenate([X[1], node2_emb], axis=1)
        # Update initial feature channels
        initial_feature_channels = X_node1.shape[1]
        # Update X with the concatenated embeddings
        X = [X_node1, X_node2]
    if explicit_feature is not None:
        initial_feature_channels = len(X[0][0])
        pass
    print("so, initial feature channels: ", initial_feature_channels)
    return np.array(D_inverse), graphs_adj, Y, X, node_size_list, initial_feature_channels  # ps, graph_adj is A_title

def create_input(data, directed):
    print("create input...")
    offset = 1 if data["index_from"] == 1 else 0
    graphs, nodes_size_list, labels = data["graphs"], data["nodes_size_list"], data["labels"]

    A_tilde, count = [], 0
    for index, graph in enumerate(graphs):
        A_tilde.append(np.zeros([nodes_size_list[index], nodes_size_list[index]], dtype=np.uint8))
        for edge in graph:
            A_tilde[count][edge[0] - offset][edge[1] - offset] = 1
            if directed == 0:
                A_tilde[count][edge[1] - offset][edge[0] - offset] = 1
        count += 1
    Y = np.where(np.reshape(labels, [-1, 1]) == 1, 1, 0)
    print("positive examples: %d, negative examples: %d." % (np.sum(Y == 0), np.sum(Y == 1)))
    A_tilde = np.array(A_tilde)

    # get A_title
    for index, x in enumerate(A_tilde):
        A_tilde[index] = x + np.eye(x.shape[0])
    # get D_inverse
    D_inverse = []
    for x in A_tilde:
        D_inverse.append(np.linalg.inv(np.diag(np.sum(x, axis=1))))
    # get X
    X, initial_feature_channels = [], 0
    def convert_to_one_hot(y, C):
        return np.eye(C)[y.reshape(-1)]
    if data["vertex_tag"]:
        vertex_tag = data["vertex_tag"]
        initial_feature_channels = len(set(sum(vertex_tag, [])))
        print("X: one-hot vertex tag, tag size %d." % (initial_feature_channels))
        for tag in vertex_tag:
            x = convert_to_one_hot(np.array(tag) - offset, initial_feature_channels)
            X.append(x)
    else:
        print("X: normalized node degree.")
        for graph in A_tilde:
            degree_total = np.sum(graph, axis=1)
            X.append(np.divide(degree_total, np.sum(degree_total)).reshape(-1, 1))
        initial_feature_channels = 1
    X = np.array(X)
    if data["feature"] is not None:
        print("embedding or explicit feature.")
        feature = data["feature"]
        for i in range(len(X)):
            X[i] = np.concatenate([X[i], feature[i]], axis=1)
        initial_feature_channels = len(X[0][0])
    print("initial feature channels: ", initial_feature_channels)
    return np.array(D_inverse), A_tilde, Y, X, nodes_size_list, initial_feature_channels

def excursion_twitter_id():
    # pre-process the twitter data
    # the original data split by comma, what `seal` need is "\t".
    data = np.array(pd.read_table("./raw_data/twitter_raw.txt", delimiter=",", header=None))
    # vertices_set = list(set(sum(data, []))) # time and memory consumption
    vertices_set = set()
    for line in data:
        vertices_set.add(line[0])
        vertices_set.add(line[1])
    vertices_set = list(vertices_set)
    vertex_map = dict([(x, vertices_set.index(x)) for x in vertices_set])

    for new_index, old_index in tqdm(enumerate(data)):
        data[new_index] = list(itemgetter(*old_index)(vertex_map))
    np.savetxt("./raw_data/twitter.txt", data, delimiter="\t", fmt="%d")

    # from seal_link_predict_new import *

    from SEAL.operators.seal_link_predict import classifier

    def seal_for_link_predict():
        classifier(0, 0.1, 128, "auto", 0.00001)

    if __name__ == "__main__":
        seal_for_link_predict()