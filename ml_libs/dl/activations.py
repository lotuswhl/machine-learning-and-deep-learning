import numpy as np


class Sigmoid(object):
    def __call__(self, x):
        return 1.0/(1.0+np.exp(-x))

    def gradient(self, x):
        sig = self.__call__(x)
        return sig*(1.0-sig)


class Relu(object):
    def __call__(self, x):
        return np.where(x < 0, 0, x)

    def gradient(self, x):
        return np.where(x < 0, 0, 1)


class Tanh(object):
    def __call__(self, x):
        return (1-np.exp(-x))/(1+np.exp(-x))

    def gradient(self, x):
        return 1-self.__call__(x)**2


class Softmax(object):
    def __call__(self, x):
        expx = np.exp(x-np.max(x, axis=-1, keepdims=True))
        return expx/np.sum(expx, axis=-1, keepdims=True)

    def gradient(self, x):
        softx = self.__call__(x)
        return softx*(1-softx)


class LeakyRelu(object):
    def __init__(self, leak=0.2):
        self.leak = leak

    def __call__(self, x):
        return np.where(x < 0, self.leak*x, x)

    def gradient(self, x):
        return np.where(x < 0, self.leak, 1)


class Elu(object):
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x < 0.0, self.alpha*(np.exp(x)-1), x)

    def gradient(self, x):
        return np.where(x < 0.0, self.alpha*np.exp(x), 1)


class Softplus(object):
    def __call__(self, x):
        return np.log(1+np.exp(x))

    def gradient(self, x):
        return 1.0/(1+np.exp(-x))
