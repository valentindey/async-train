# -*- coding: utf-8 -*-

import os
import gzip
import pickle
import logging


def load_data(flatten=False):

    datasets_dir = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(datasets_dir, "mnist.pkl.gz")
    if not os.path.isfile(filename):
        logging.info("Downloading mnist data...")
        import urllib.request
        urllib.request.urlretrieve("https://s3.amazonaws.com/img-datasets/mnist.pkl.gz",
                                   filename)

    with gzip.open(filename, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    (x_train, y_train), (x_test, y_test) = data

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    return (x_train, y_train), (x_test, y_test)
