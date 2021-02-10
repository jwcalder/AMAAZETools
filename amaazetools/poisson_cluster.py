import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

#Version of Poisson learning to compute class medians
def poisson_median(W,g,min_iter=50):

    n = W.shape[0]
    
    #Poisson source term
    Kg,_ = gl.LabelsToVec(g)
    b = Kg.T - np.mean(Kg,axis=1)

    #Setup matrices
    D = gl.degree_matrix(W,p=-1)
    P = D*W.transpose()
    Db = D*b

    v = np.max(Kg,axis=0)
    v = v/np.sum(v)
    vinf = gl.degrees(W)/np.sum(gl.degrees(W))
    RW = W.transpose()*D
    u = np.zeros_like(b)

    #Number of iterations
    T = 0
    while (T < min_iter or np.max(np.absolute(v-vinf)) > 1/n) and (T < 1000):
        u = Db + P*u
        v = RW*v
        T = T + 1

    print("Grad Desc: %d iter" % T)

    u = u - np.mean(u,axis=0)
    return u

def poisson_kmeans(W, num_classes, ind=None):
    n = W.shape[0]
    #Randomly choose num_classes labeled points
    centroids = []
    if ind is None:
        ind = np.random.choice(n, size=num_classes, replace=False)
    num_changed = 1
    while num_changed > 0:
        centroids.append(ind)
        #Semi-supervised learning 
        #l = gl.graph_ssl(W,ind,np.arange(num_classes),method='poisson2')
        u,_ = gl.poisson2(W,ind,np.arange(num_classes),min_iter=1000)
        u = u.T
        u = u - np.mean(u,axis=0)
        l = np.argmax(u,axis=1)
        u = poisson_median(W,l)
        ind_old = ind.copy()
        ind = np.argmax(u,axis=0)
        num_changed = np.sum(ind_old != ind)
    #l = gl.graph_ssl(W,ind,np.arange(num_classes),method='poisson')
    return (u, centroids)


