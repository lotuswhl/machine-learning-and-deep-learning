import numpy as np 
import math



"""
实现基础的回归模型,线性模型,多项式模型以及各种正则化的版本;
"""


class FundmentalRegression(object):
    def __init__(self,n_iterations,learning_rate):
        super(FundmentalRegression,self).__init__()
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initialize_weights(self,n_features):
        N=math.sqrt(n_features)
        self.w = np.random.uniform(-1/N, 1/N, n_features)

    def fit(self,X,y):
        # 首先给X插入bias项
        X=np.insert(arr=X, obj=0, values=1,axis=1)
        # 然后初始化权重
        self.initialize_weights(X.shape[1])
        # 初始化一个损失列表,保存每次迭代的损失,这样就可以在后续查看优化过程
        self.loss_list=[]
        # 接下来就可以使用gradient descent来优化权重
        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)
            loss = np.mean(0.5*(y-y_pred)**2+self.regularization(self.w))
            self.loss_list.append(loss)
            # loss 对 权重w的梯度
            grad_w = -(y-y_pred).dot(X)+self.regularization.grad(self.w)
            # 更新权重
            self.w -= self.learning_rate*grad_w
    def predict(self,X):
        X=np.insert(X, 0, 1,axis=1)
        pred=X.dot(self.w)
        return pred




class LinearRegression(FundmentalRegression):
    """
    实现线性回归
    """
    def __init__(self,n_iterations=100,learning_rate=0.001,use_gradient_descent=True):
        super(LinearRegression,self).__init__(n_iterations,learning_rate)
        self.use_gradient_descent=use_gradient_descent
        self.regularization=lambda x:0
        self.regularization.grad=lambda x:0

    def fit(self,X,y):
        # 不使用gradient descent,则直接计算
        if not self.use_gradient_descent:
            # 同样需要插入bias项
            X=np.insert(X,0,1,axis=1)
            # 根据:xw=y
            # xTxw=xTy
            # w=((xTx)^(-1))xTy
            # 首先对xTx 奇异值分解
            U,S,V=np.linalg.svd(X.T.dot(X))
            # 将奇异值向量转为对角矩阵
            S=np.diag(S)
            # ((xTx)^(-1))
            x_trans_x_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w=x_trans_x_inv.dot(X.T).dot(y)
        else:
            super(LinearRegression,self).fit(X, y)


class L1Regularization(object):
    def __init__(self,alpha):
        super(L1Regularization,self).__init__()
        self.alpha=alpha
    def __call__(self,w):
        return self.alpha*np.linalg.norm(w)
    def grad(self,w):
        return self.alpha*np.sign(w)

class L2Regularization(object):
    def __init__(self,alpha):
        super(L2Regularization,self).__init__()
        self.alpha=alpha
    def __call__(self,w):
        return 0.5*self.alpha*w.T.dot(w)
    def grad(self,w):
        return self.alpha*w    

class L1AndL2Regularization(object):
    def __init__(self,alpha,beta):
        self.alpha=alpha
        self.beta=beta
    def __call__(self,w):
        l1_w = self.beta*np.linalg.norm(w)
        l2_w = (1-self.beta)*0.5*w.T.dot(w)
        return self.alpha*(l1_w+l2_w)

    def grad(self,w):
        l1_g = self.beta*np.sign(w)
        l2_g = (1-self.beta)*w
        return self.alpha*(l1_g+l2_g)



                                                                            





if __name__=="__main__":
    import pandas as pd 
    data = pd.read_csv("../datasets/tmp.csv",header=0,delimiter="\\s*")
    y=data.values[:,-1]
    X=data.values[:,0]
    import matplotlib.pyplot as plt 
    fig = plt.figure(figsize=(20,10))
    ax = plt.subplot(2,1,1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim((-20,20))
    ax.set_xlim((0,2))
    plt.plot(X,y)

    # plt.show()
    lr = LinearRegression()
    lr.fit(X[:,np.newaxis], y)
    y_pred=lr.predict(X[:,np.newaxis])
    print(y_pred[:10])

    ax = plt.subplot(2,1,2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim((-20,20))
    ax.set_xlim((0,2))
    plt.plot(X,y_pred)
    plt.show()

