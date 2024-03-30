import argparse
from seal_link_predict_new import *
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def seal_for_link_predict():
    classifier("fb-pages", 0, 0.1, 128, "auto",0.00001)
    # classifier_for_node_pair((2,182),"fb-pages", 0, 0.1, 128, "auto",0.00001)
    # positive, negative, nodes_size = load_data(0)
    # embedding_feature = learning_embedding(positive, negative, nodes_size, 0.2, 128, 0)
    # print(embedding_feature)
if __name__ == "__main__":
    seal_for_link_predict()
    # positive, negative, nodes_size = load_data("abc", 0)
    
    # pos_train_edge, pos_test_edge = train_test_split(positive, test_size=0.2, random_state=1)
    # pos_train_edge, pos_valid_edge  = train_test_split(positive, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

    # pos_train_edge = tf.convert_to_tensor(pos_train_edge)
    # pos_test_edge = tf.convert_to_tensor(pos_test_edge)
    # pos_valid_edge = tf.convert_to_tensor(pos_valid_edge)

    # neg_train_edge, neg_test_edge = train_test_split(negative, test_size=0.2, random_state=1)
    # neg_train_edge, neg_valid_edge  = train_test_split(negative, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

    # neg_train_edge = tf.convert_to_tensor(neg_train_edge)
    # neg_test_edge = tf.convert_to_tensor(neg_test_edge)
    # neg_valid_edge = tf.convert_to_tensor(neg_valid_edge)

    # print(neg_valid_edge)
    # print("load data...")
    # file_path = "./raw_data/fb-pages-food/fb-pages-food.csv"
    # positive_df = pd.read_csv(file_path, delimiter=',', dtype=int)
    # positive = positive_df.to_numpy()

    # print(X_test)

    # # Đọc dữ liệu từ tệp CSV
    # data = np.loadtxt("./raw_data/fb-pages-food/fb-pages-food.csv", delimiter=',', dtype=int)
    # print(type(data))
    # # Hiển thị dữ liệu
    # print(data)