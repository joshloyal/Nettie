import logging

def set_backend(name):
    logger = logging.getLogger(__name__)
    if name == 'mxnet':
        logger.debug('Using mxnet backend.')
        import mxnet_backend as backend
        return backend
    elif name == 'keras':
        logger.debug('Using keras backend.')
        import keras_backend as backend
        return backend
    else:
        raise NotImplementedError('Backend {} not implemented.'.format(name))
