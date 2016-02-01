import logging
import sys

import numpy as np
import pandas as pd
import plac
np.random.seed(1337)  # for reproducibility

import nettie.backend as net

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from keras.utils import np_utils

# mxnet logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# nettie logger
logging.getLogger('nettie').addHandler(logging.StreamHandler(stream=sys.stdout))

def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X)  # https://youtu.be/uyUXoap67N8
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids


def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def main(backend):
    print('Loading data...')
    X, labels = load_data('train.csv', train=True)
    X, scaler = preprocess_data(X)

    if backend == 'keras':
        y, encoder = preprocess_labels(labels)
        nb_classes = y.shape[1]
        print(nb_classes, 'classes')
    else:
        y, encoder = preprocess_labels(labels, categorical=False)
        nb_classes = len(np.unique(y))
        print(nb_classes, 'classes')

    dims = X.shape[1]
    print(dims, 'dims')

    nn = net.set_backend(backend)

    model = nn.Sequential()
    model.add(nn.Dense(512, input_shape=(dims,)))
    model.add(nn.PReLU())
    model.add(nn.BatchNormalization())
    model.add(nn.Dropout(p=0.5))

    model.add(nn.Dense(512))
    model.add(nn.PReLU())
    model.add(nn.BatchNormalization())
    model.add(nn.Dropout(p=0.5))

    model.add(nn.Dense(512))
    model.add(nn.PReLU())
    model.add(nn.BatchNormalization())
    model.add(nn.Dropout(p=0.5))

    model.add(nn.Dense(nb_classes))
    model.add(nn.Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.fit(X, y, nb_epoch=20, batch_size=128, validation_split=0.15)

    return model

if __name__ == '__main__':
    model = plac.call(main)
