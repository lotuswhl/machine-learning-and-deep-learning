# --*-- coding:utf-8 --*--
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(2018)

"""
a navive implementation of neural networks with an example 
"""

# activation functions

class relu(object):
    """
    elementwise operation on numpy ndarrays (or like numpy arrays)
    relu activation function: max(0,x)
    """

    def __call__(self, x):
        return np.where(x > 0, x, 0)

    def gradient(self, x):
        return np.where(x > 0, 1, 0)

        
class tanh(object):
    """
    tanh activation function: tanh(x)=(1-exp(-x))/(1+exp(-x))
    """

    def __call__(self, x):
        # return (1 - np.exp(-x)) / (1 + np.exp(-x))
        return np.tanh(x)

    def gradient(self, x):
        # return (1 - self.__call__(x) ** 2)
        return 1 - np.tanh(x) ** 2


class sigmoid(object):
    """
    sigmoid(x)=exp(-x)/(1-exp(-x))
    """

    def __call__(self, x):
        return np.exp(-x) / (1 - np.exp(-x))
    
    def gradient(self, x):
        sig = self.__call__(x)
        return sig * (1 - sig)

# helper functions


def get_weights(node_in, node_out, initializer="random"):
    """
    currently for simplicity ,provide initializer:random,uniform
    """
    if initializer == "random":
        return np.random.randn(node_in, node_out) / np.sqrt(node_in)
    elif initializer == "uniform":
        return np.random.uniform(low=-1, high=1, size=[node_in, node_out])
    else:
        raise NotImplementedError("not supported initializer")
    

def get_biases(n_hnode):
    return np.zeros(shape=[1, n_hnode])


def softmax_score(output):
    exps = np.exp(output)
    return exps / exps.sum(axis=1, keepdims=True)


def softmax_loss(scores, y_true):
    num_samples = scores.shape[0]
    nlls = -np.log(scores[range(num_samples), y_true])
    return np.sum(nlls)

# neural nets begins


class LinearLayer(object):
    """
    basic linear layer for nn
    Parameters:
    ----------------------
    in_dim: input dimention
    out_dim: output dimention
    activation_fn: optional activation function for the layer
    name: optional name for the layer 
    """

    def __init__(self, in_dim, out_dim, activation_fn=None, name="default"):
        super(LinearLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation_fn = activation_fn
        self.name = name

        # initialize weights and biases
        self.weights = get_weights(self.in_dim, self.out_dim)
        self.biases = get_biases(self.out_dim)

    def __call__(self, X):
        """
        coputer forward pass over the layer
        -------------------
        Z=XW+b
        A=f(Z)
        Parameters:
        X: input data X
        """
        self.X = X
        self.Z = X.dot(self.weights) + self.biases
        self.A = self.activation_fn(self.Z) if self.activation_fn else self.Z
        return self.A

    def gradient(self, g_out):
        """
        compute the gradients over the layer used for backpropagation
        Parameters:
        ------------------
        g_out: gradient caused by the output of the layer

        """
        self.g_A = g_out
        self.g_Z = self.g_A * self.activation_fn.gradient(self.Z) if self.activation_fn else self.g_A
        self.g_w = self.X.T.dot(self.g_Z)
        self.g_b = self.g_Z
        self.g_X = self.g_Z.dot(self.weights.T)
        return (self.g_X, self.g_w, self.g_b)


class VanilaNN(object):
    """
    a very naive implementation of NN,just to get familiar with it
    """

    def __init__(self, X, y, lr=0.01, reg=0.01):
        super(VanilaNN, self).__init__()
        self.X = X 
        self.y = y
        self.num_samples, self.num_features = self.X.shape
        self.out_nodes = np.max(y) + 1
        self.lr = lr
        self.reg = reg

        self.build_model()

    def build_model(self):
        """
        just use one hidden layer and one output layer for simplicity
        """
        self.layer1 = LinearLayer(self.num_features, 4, activation_fn=tanh())
        self.layer2 = LinearLayer(4, self.out_nodes)

    def calculate_loss(self):
        """
        calcualte the total loss for the batch
        """
        softloss = softmax_loss(self.scores, self.y)
        regloss = 0.0
        regloss += np.sum(np.square(self.layer1.weights))
        regloss += np.sum(np.square(self.layer2.weights))
        return (softloss + self.reg / 2 * regloss) / self.num_samples
    
    def forward(self):
        """
        computer forward propagation for the whole batch
        """
        self.A1 = self.layer1(self.X)
        self.A2 = self.layer2(self.A1)
        self.scores = softmax_score(self.A2)

    def backward(self):
        """
        compute the correspond gradient through backpropagation
        """
        self.g_out = self.scores
        self.g_out[range(self.num_samples), self.y] -= 1
        self.g_l2, self.g_l2_w, self.g_l2_b = self.layer2.gradient(self.g_out)
        self.g_l1, self.g_l1_w, self.g_l1_b = self.layer1.gradient(self.g_l2)

    def optimize(self):
        """
        update weights and biases ,using batch gradient descent
        """

        self.g_l1_b = self.g_l1_b.sum(axis=0, keepdims=True)
        self.g_l2_b = self.g_l2_b.sum(axis=0, keepdims=True)

        # calculate regularization
        self.g_l2_w += self.reg * self.layer2.weights
        self.g_l1_w += self.reg * self.layer1.weights

        self.layer2.weights += -self.lr * self.g_l2_w
        self.layer1.weights += -self.lr * self.g_l1_w
        self.layer2.biases += -self.lr * self.g_l2_b
        self.layer1.biases += -self.lr * self.g_l1_b

    def train(self, niters=2000):
    
        for i in range(niters):
            # forward
            self.forward()
            if  i % 100 == 0:
                print("after {0} training iterations, loss is:{1}".format(i, self.calculate_loss()))
            self.backward()
            self.optimize()
            # print(self.layer1.weights,self.layer2.weights)

    def predict(self, X):
        self.X = X 
        self.forward()
        return np.argmax(self.scores, axis=1)


def plot_decision_boundary(X, y, pred_func):
    """
    plot decition boundary through model's prediction function
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


def main(): 
    from sklearn.datasets import make_moons
    
    datax, datay = make_moons(300, noise=0.2)
    # print(datax[:10,:])
    # plt.scatter(datax[:, 0], datax[:, 1], c=datay, cmap=plt.cm.Spectral,s=40)
    # plt.show()
    model = VanilaNN(datax, datay)
    model.train()
    plot_decision_boundary(datax, datay, model.predict)


if __name__ == '__main__':
    main()
        
        
