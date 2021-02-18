import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import scipy.sparse as sparse
import scipy.spatial as spatial
from sklearn.neighbors import NearestNeighbors

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



def canonical_labels(u):
    n = len(u)
    k = len(np.unique(u))
    label_set = np.zeros((k,1))
    label = 0
    for i in range(n):
        if u[i] > label:
            label += 1
            l = u[i]
            I = u == label
            J = u == l
            u[I] = l
            u[J] = label
    return u

def graph_setup(x,y,z,faces,n,r,p, edgeSep=0):
    
    Pts = np.column_stack((x,y,z))
    normals = np.zeros(Pts.shape)
    
    tri = Pts[faces]
    triVectors = np.cross(tri[::,1 ] - tri[::,0], tri[::,2 ] - tri[::,0])
    triVectorsLens = np.sqrt(triVectors[:,0]**2 + triVectors[:,1]**2 + triVectors[:,2]**2)
   
    triVectors[:,0] /= triVectorsLens
    triVectors[:,1] /= triVectorsLens
    triVectors[:,2] /= triVectorsLens
    
    normTriVectors = triVectors
    
    normals[faces[:,0]] += normTriVectors
    normals[faces[:,1]] += normTriVectors
    normals[faces[:,2]] += normTriVectors
    
    normalsLens = np.sqrt(normals[:,0]**2 + normals[:,1]**2 + normals[:,2]**2)
    normals[:,0] /= normalsLens
    normals[:,1] /= normalsLens
    normals[:,2] /= normalsLens

    v = normals #vertex unit normals   
    N = len(Pts)
    
    #Random subsample
    # if edgeSep > 0:
    # else:
    ss_idx = np.matrix(np.random.choice(Pts.shape[0],n,False))
    y = np.squeeze(Pts[ss_idx,:])
    w = np.squeeze(v[ss_idx,:])

    xTree = spatial.cKDTree(Pts)
    nn_idx = xTree.query_ball_point(y, r)
    yTree = spatial.cKDTree(y)
    nodes_idx = yTree.query_ball_point(y, r)
    
    bn = np.zeros((n,3))
    J = sparse.lil_matrix((N,n))
    for i in range(n):
        vj = v[nn_idx[i],:]
        normal_diff = w[i] - vj
        weights = np.exp(-8 * np.sum(np.square(normal_diff),1,keepdims=True))
        bn[i] = np.sum(weights*vj,0) / np.sum(weights,0)
        
        #Set ith row of J
        normal_diff = bn[i]- vj
        weights = np.exp(-8 * np.sum(np.square(normal_diff),1))#,keepdims=True))
        J[nn_idx[i],i] = weights
        
    #Normalize rows of J
    RSM = sparse.spdiags((1 / np.sum(J,1)).ravel(),0,N,N)
    J = RSM @ J
    
    #Compute weight matrix W
    W = sparse.lil_matrix((n,n))
    for i in range(n):
        nj = bn[nodes_idx[i]]
        normal_diff = bn[i] - nj
        weights = np.exp(-32 * ((np.sqrt(np.sum(np.square(normal_diff),1)))/2)**p)
        W[i,nodes_idx[i]] = weights
    
    #Find nearest node to each vertex
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(y)
    instances, node_idx = nbrs.kneighbors(Pts)
    
    return W,J,ss_idx,node_idx


