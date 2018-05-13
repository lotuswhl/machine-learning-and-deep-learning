import numpy as np 

np.random.seed(47)

class SimplePCA(object):
    """
    naive pac implementation

    Parameters:
    num_components: 需要保留的主成分数量
    """

    def __init__(self,num_components):
        super(SimplePCA,self).__init__()
        self.num_components  = num_components

    def fit(self,x):
        # make sure we are dealing with numpy array
        x=np.asarray(x)
        x=x.T
        # # make sure the input data take the format N(num_samples)xM(feature dimention)
        # assert(len(x.shape)==2)
        n_features,n_samples=x.shape
        assert(self.num_components <= min(n_samples,n_features))
        
        # zero mean for all dimensions
        means=np.mean(x,axis=1)
        means=means.reshape(-1,1)
        x-=means

        # now computer covariance matrix for x
        # and store it for future use
        self.cov = np.cov(x)
        # then computer eigen decomposition for the cov
        eigenvalues,eigenvectors=np.linalg.eig(self.cov)
        # sort the eigen pairs by eigenvalues,since roughly speaking,
        # the lower eigenvalue is ,the least information it carrys for that
        # eigenvetcor direction,and should be droped
        pairs=[(eigenvalues[i],eigenvectors[:,i]) for i in range(len(eigenvalues))]

        # sort
        pairs.sort(key=lambda p:p[0], reverse=True)

        self.explained_variance_,self.components_=(np.asarray([k[0] for k in pairs]),
                np.hstack((p[1].reshape((-1,1)) for p in pairs)))

    def transform(self,nx):
        # n_features*k
        trans_matrix=self.components_[:,:self.num_components]
        # nx: num_samples*n_features
        trans_nx=np.matmul(nx,trans_matrix)

        return trans_nx



if __name__=="__main__":
    x=np.arange(0,20,0.5)
    y=2*x+3
    noise=3*np.random.normal(size=[len(x)])
    y+=noise

    import matplotlib.pyplot as plt 
    fig,axes=plt.subplots(2,1)
    ax=axes[0]

    ax.scatter(x, y)
    pca=SimplePCA(num_components=1)
    x=np.array(x).reshape((-1,1))
    y=np.array(y).reshape((-1,1))
    data=np.hstack((x,y))

    # print(data)

    pca.fit(data)

    trans_nx=pca.transform(data)
    x=trans_nx[:,0]
    y=np.zeros_like(x)
    ax=axes[1]
    ax.scatter(x, y)
    plt.show()    








