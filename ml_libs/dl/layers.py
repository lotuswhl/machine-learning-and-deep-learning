import numpy as np
import copy


class Layer(object):

    def layer_name(self):
        return self.__class__.__name__

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    def forward(self, input, trainable):
        raise NotImplementedError()

    def backward(self, accum_grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()

    def num_parameters(self):
        return 0


class Linear(Layer):
    def __init__(self, num_units, input_shape=None):
        self.num_units = num_units
        self.input_shape = input_shape

        self.layer_input = None
        self.W = None
        self.W0 = None
        self.trainable = True

    def init(self, optimizer):

        limit = np.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit,
                                   [self.input_shape[0], self.num_units])
        self.W0 = np.zeros(1, self.num_units)

        self.W_optmizer = copy.copy(optimizer)
        self.W0_optimizer = copy.copy(optimizer)

    def num_parameters(self):
        return np.prod(self.W.shape)+np.prod(self.W0.shape)

    def forward(self, input):
        self.layer_input = input
        return input.dot(self.W)+self.W0

    def backward(self, accum_grad):
        W = self.W

        if self.trainable:
            grad_W = self.layer_input.T.dot(accum_grad)
            grad_W0 = np.sum(accum_grad, axis=0, keepdims=True)

            self.W = self.W_optmizer.update(self.W, grad_W)
            self.W0 = self.W0_optimizer.update(self.W0, grad_W0)

        return accum_grad.dot(W.T)

    def output_shape(self):
        return (self.num_units,)
