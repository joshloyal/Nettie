import six
import numbers
import logging

import numpy as np

from sklearn.metrics import log_loss

import mxnet as mx
from mxnet.metric import EvalMetric

__all__ = ['Activation', 'Dense', 'SoftmaxOutput', 'Variable',
           'BatchNormalization', 'Dropout', 'Sequential', 'Adam']

def logloss(y_true, y_pred):
    try:
        return log_loss(y_true, y_pred)
    except ValueError:
        return 1

class LogLoss(mx.metric.EvalMetric):
    def __init__(self):
        super(LogLoss, self).__init__('log_loss')

    def update(self, labels, preds):
        for i in range(len(labels)):
            pred = preds[i].asnumpy()
            label = labels[i].asnumpy().astype('int32')

            self.sum_metric += log_loss(label, pred)
            self.num_inst += pred_label.shape[0]

LOSS_MAP = {'categorical_crossentropy': mx.metric.np(logloss)}

class MXNetSymbol(object):
    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.args = args
        self.kwargs = kwargs

    @property
    def symbol(self):
        pass

    def __call__(self, prev_symbol=None):
        if prev_symbol:
            return self.symbol(prev_symbol, *self.args, **self.kwargs)
        return self.symbol(*self.args, **self.kwargs)


class Activation(MXNetSymbol):
    @property
    def symbol(self):
        return mx.symbol.Activation

    def __call__(self, prev_symbol=None):
        """ Overwrite to allow for passing of the name of the activation
            directly. In addition, alllow for detection of output layers,
            i.e. SoftmaxOutput
        """
        assert len(self.args) == 1  # this should be the name of the activaiton

        # pop off the activation
        activation = self.args[0]
        self.args = self.args[1:]

        if not isinstance(activation, six.string_types):
            raise ValueError('activation type must be a string')

        if activation == 'softmax':
            self.logger.debug('Detected SoftmaxOutput in activation.')
            return mx.symbol.SoftmaxOutput(
                prev_symbol, name='softmax', *self.args, **self.kwargs)
        elif prev_symbol:
            return self.symbol(
                prev_symbol, *self.args, act_type=activation, **self.kwargs)

        return self.symbol(*self.args, act_type=activation, **self.kwargs)

class LeakyReLU(MXNetSymbol):
    @property
    def symbol(self):
        return mx.symbol.LeakyReLU

    @property
    def act_type(self):
        pass

    def __call__(self, prev_symbol=None):
        if prev_symbol:
            return self.symbol(
                prev_symbol, *self.args, act_type=self.act_type, **self.kwargs)
        return self.symbol(*self.args, **self.kwargs)


class PReLU(LeakyReLU):
    @property
    def act_type(self):
        return 'prelu'


class Dense(MXNetSymbol):
    """ We are going to use the Keras naming convention. We need a base
        layer class eventually.
    """
    @property
    def symbol(self):
        return mx.symbol.FullyConnected

    def __call__(self, prev_symbol=None):
        """ Overwrite to allow for passing num_hidden directly. """
        assert len(self.args) == 1  # this should be the number of hidden units

        # pop off the activation
        num_hidden = self.args[0]
        self.args = self.args[1:]
        if not isinstance(num_hidden, numbers.Integral) or num_hidden < 0:
            raise ValueError('number of hidden units must be a '
                             'positive integer.')

        if prev_symbol:
            # HACK: input_shape is used in keras and not mxnet. Lets pop
            #       it off for now and figure out a better inference later.
            if 'input_shape' in self.kwargs:
                del self.kwargs['input_shape']
            return self.symbol(
                prev_symbol, *self.args, num_hidden=num_hidden, **self.kwargs)
        return self.symbol(*self.args, num_hidden=num_hidden, **self.kwargs)



class SoftmaxOutput(MXNetSymbol):
    @property
    def symbol(self):
        return mx.symbol.SoftmaxOutput


class Variable(MXNetSymbol):
    @property
    def symbol(self):
        return mx.symbol.Variable


class BatchNormalization(MXNetSymbol):
    @property
    def symbol(self):
        return mx.symbol.BatchNorm


class Dropout(MXNetSymbol):
    @property
    def symbol(self):
        return mx.symbol.Dropout


class Sequential(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.prev_symbol = Variable('data')()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def compile(self, optimizer, loss, class_mode='categorical'):
        """ for mxnet this is not necessary, but we have it here for
            convenience.
        """
        try:
            self.loss = LOSS_MAP[loss]
        except KeyError:
            self.logger.debug('Loss function not found.')
            self.loss = 'acc'

        self.optimizer = optimizer

    def visualize(self):
        return mx.viz.plot_network(self.prev_symbol)

    def add(self, symbol):
            self.prev_symbol = symbol(self.prev_symbol)

    def fit(self, X, y, nb_epoch=10, learning_rate=0.01, batch_size=128, validation_split=0.15):
        self.model = mx.model.FeedForward(self.prev_symbol,
                                          num_epoch=nb_epoch,
                                          optimizer=self.optimizer,
                                          numpy_batch_size=batch_size,
                                          learning_rate=learning_rate)

        self.model.fit(X, y, eval_metric=self.loss)

    def predict(self, X):
        return self.model.predict(X)

# direct imports
Adam = mx.optimizer.Adam
