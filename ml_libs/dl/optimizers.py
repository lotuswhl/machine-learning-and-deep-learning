import numpy as np


class SGD(object):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = None

    def update(self, w, grad_w):
        if self.v is None:
            self.v = np.zeros_like(grad_w)
        self.v = self.momentum*self.v + (1-self.momentum)*grad_w
        return w-self.learning_rate*self.v


class Adagrad(object):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.G = None
        self.epsilo = 1e-9

    def update(self, w, grad_w):
        if self.G is None:
            self.G = np.zeros_like(grad_w)

        self.G += grad_w**2

        return w-self.learning_rate*grad_w/(self.G+self.epsilo)


class Adadelta(object):
    def __init__(self, lamd=0.95, epsilon=1e-7):
        self.lamd = lamd
        self.epsilo = epsilon

        # 梯度的平方和 均值
        self.grad_avg = None
        # 权重更新梯度的平方和 均值
        self.w_update_avg = None
        # 权重更新值
        self.w_update = None

    def update(self, w, grad_w):
        if self.w_update is None:
            self.w_update = np.zeros_like(grad_w)
            self.grad_avg = np.zeros_like(grad_w)
            self.w_update_avg = np.zeros_like(grad_w)

        self.grad_avg = self.lamd*self.grad_avg+(1-self.lamd)*grad_w**2

        rms_w_up = np.sqrt(self.w_update_avg+self.epsilo)
        rms_grad = np.sqrt(self.grad_avg+self.epsilo)

        adaptive_lr = rms_w_up/rms_grad

        self.w_update = adaptive_lr*grad_w

        self.w_update_avg = self.lamd*self.w_update_avg + \
            (1-self.lamd)*self.w_update**2
        return w - self.w_update


class RMSprop(object):
    def __init__(self, learning_rate=0.01, lamd=0.9, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.lamd = lamd

        self.grad_avg = None

    def update(self, w, grad_w):

        if self.grad_avg is None:
            self.grad_avg = np.zeros_like(grad_w)

        self.grad_avg = self.lamd * self.grad_avg + (1-self.lamd)*grad_w**2

        return w - self.learning_rate*grad_w/(self.grad_avg+self.epsilon)


class Adam(object):
    def __init__(self, learning_rate=0.001, alpha=0.9, beta=0.999, epsilon=1e-7):

        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

        self.m = None
        self.v = None

    def update(self, w, grad_w):
        if self.m is None:
            self.m = np.zeros_like(grad_w)
            self.v = np.zeros_like(grad_w)

        self.m = self.alpha*self.m + (1-self.alpha)*grad_w
        self.v = self.beta*self.v+(1-self.beta)*grad_w**2

        m_hat = self.m/(1-self.alpha)
        v_hat = self.v/(1-self.beta)

        return w - self.learning_rate*grad_w*m_hat/(v_hat+self.epsilon)
