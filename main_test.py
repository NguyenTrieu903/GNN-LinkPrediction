import argparse
from seal_link_predict_new import *
import numpy as np


def seal_for_link_predict():
    # classifier("fb-pages", 0, 0.1, 128, "auto",0.00001)
    classifier_for_node_pair((2,182),"fb-pages", 0, 0.1, 128, "auto",0.00001)

if __name__ == "__main__":
    seal_for_link_predict()
    

    # # Đọc dữ liệu từ tệp CSV
    # data = np.loadtxt("./raw_data/fb-pages-food/fb-pages-food.csv", delimiter=',', dtype=int)
    # print(type(data))
    # # Hiển thị dữ liệu
    # print(data)