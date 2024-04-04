import argparse
# from seal_link_predict_new import *
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from SEAL.seal_link_predict_new import classifier


def seal_for_link_predict():
    classifier("fb-pages", 0, 0.1, 128, "auto",0.00001)
if __name__ == "__main__":
    seal_for_link_predict()