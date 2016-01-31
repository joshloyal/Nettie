import logging
import sys

import numpy as np
import plac
from keras.datasets import mnist
from keras.utils import np_utils

import nettie.backend as net

# mxnet logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# nettie logger
logging.getLogger('nettie').addHandler(logging.StreamHandler(stream=sys.stdout))

np.random.seed(1337)  # for reproducibility

def get_mnist(backend='keras'):
    batch_size = 128
    nb_classes = 10
    nb_epoch = 20

    # the data, shuffled and split between tran and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    if backend == 'keras':
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)

    return (X_train, y_train), (X_test, y_test)

def main(backend):
    (X_train, y_train), (X_test, y_test) = get_mnist(backend)

    # set the backend
    nn = net.set_backend(backend)

    model = nn.Sequential()
    model.add(nn.Dense(512, input_shape=(784,)))
    model.add(nn.Activation('relu'))
    model.add(nn.BatchNormalization())
    model.add(nn.Dropout(p=0.2))
    model.add(nn.Dense(512))
    model.add(nn.Activation('relu'))
    model.add(nn.BatchNormalization())
    model.add(nn.Dropout(p=0.2))
    model.add(nn.Dense(10))
    model.add(nn.Activation('softmax'))

    rms = nn.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=rms)

    model.fit(X_train, y_train, nb_epoch=20)#, optimizer=rms)
    proba = model.predict(X_test)
    pred = np.argmax(proba, axis=1)

    if backend == 'keras':
        print 'Test error: {:.05f}'.format(np.mean(pred == np.argmax(y_test, axis=1)))
    else:
        print 'Test error: {:.05f}'.format(np.mean(pred == y_test))

    return model

if __name__ == '__main__':
    model = plac.call(main)

