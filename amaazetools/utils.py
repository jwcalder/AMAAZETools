import numpy as np
import scipy.sparse as sp
import scipy.spatial as spatial
from sklearn.neighbors import NearestNeighbors
from mayavi import mlab

def cplotsurf(x,y,z,triangles,C=-1):
    if C.any == -1: #if no C given
        C = np.ones((len(x),1))
        
    n = len(np.unique(C))
    C = C.astype(int)
    if n>20:
        mesh = mlab.triangular_mesh(x,y,z,triangles,scalars=C)
    else:
        col = (np.arange(1,n+1)) / n
        colors = col[C-1]
        mesh = mlab.triangular_mesh(x,y,z,triangles,scalars=colors)
        
    return mesh

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

def graph_setup(x,y,z,faces,n,r,p):
    
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
    ss_idx = np.matrix(np.random.choice(Pts.shape[0],n,False))
    y = np.squeeze(Pts[ss_idx,:])
    w = np.squeeze(v[ss_idx,:])

    xTree = spatial.cKDTree(Pts)
    nn_idx = xTree.query_ball_point(y, r)
    yTree = spatial.cKDTree(y)
    nodes_idx = yTree.query_ball_point(y, r)
    
    bn = np.zeros((n,3))
    J = sp.lil_matrix((N,n))
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
    RSM = sp.spdiags((1 / np.sum(J,1)).ravel(),0,N,N)
    J = RSM @ J
    
    #Compute weight matrix W
    W = sp.lil_matrix((n,n))
    for i in range(n):
        nj = bn[nodes_idx[i]]
        normal_diff = bn[i] - nj
        weights = np.exp(-32 * ((np.sqrt(np.sum(np.square(normal_diff),1)))/2)**p)
        W[i,nodes_idx[i]] = weights
    
    #Find nearest node to each vertex
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(y)
    instances, node_idx = nbrs.kneighbors(Pts)
    
    return W,J,ss_idx,node_idx
