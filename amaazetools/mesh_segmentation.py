import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import scipy.sparse as sparse
import scipy.spatial as spatial
from sklearn.neighbors import NearestNeighbors
import amaazetools.edge_detection as ed
import ipdb
#Version of Poisson learning to compute class medians
def poisson_median(W,g,min_iter=50):
    """Compute the median of a given set of vertices. Helper function for poisson_kmeans

        Args:
            W: Weight matrix of the graph. scipy.sparse.csr_matrix
            g: int32 numpy array containing indices of the vertices
            min_iter: minimum number of iterations
        Returns:
            u: numpy array of shape (n, num_classes)
    """
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
    """Run the poisson "k-means" clustering algorithm.
        Args:
            W: Weight matrix of the graph. scipy.sparse.csr_matrix
            num_classes: int scalar giving the number of classes
            ind: optional numpy array giving the indices of the centroid initializations. If not provided, they are selected randomly.
        Returns:
            u: numpy array of shape (n, num_classes). The index of the largest entry in each row corresponds to the assigned cluster.
            centroids: Indices of the initialized cluster centers. numpy array
    """
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
        print(f"I = {ind}, num_classes = {num_classes}")
        u, _ = gl.poisson2(W, ind, np.arange(num_classes), min_iter=1000)
        u = u.T
        u = u - np.mean(u,axis=0)
        l = np.argmax(u,axis=1)
        u = poisson_median(W,l)
        if u.shape[1] < num_classes:
            print(f"Warning: The number of clusters has decreased from {num_classes} to {u.shape[1]}")
            num_classes = u.shape[1]
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

def graph_setup(mesh,n,r,p, edgeSep=0):
    """Builds a graph by sampling a given mesh. Vertices are connected if they are within distance r and have similar normal vectors

        Args:
            mesh: An amaazetools.trimesh.mesh object
            n: int scalar signifying the number of vertices to sample for the graph
            r: float scalar. Radius for graph construction.
            p:  Weight matrix parameter
            edgeSep: optional float scalar. If given, we restrict sampling to points at least edgeSep from an edge point.
        Returns:
            W: Weight matrix describing similarities of normal vectors. Type scipy.sparse.lil_matrix having shape (n,n)
            J: Matrix with indices of nearest neighbors
            ss_idx: int32 numpy array containing indices of subsample
            node_idx: int32 numpy array containing indices of closest point in subsample.
    """

    #Pts = np.column_stack((x,y,z))
    Pts = mesh.Points
    faces = mesh.Triangles
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
    sample_mask = np.ones(Pts.shape[0])
    if edgeSep > 0: # Restrict the subsample to points at least edgeSep away from an edge point
        #edge_mask = ed.edge_graph_detect(mesh,1,1)
        # detect edges
        VOL,K1,K2,V1,V2,V3 = mesh.svipca([.2])
        #threshold svi
        E = VOL < (np.mean(VOL,axis=0)-.5*np.std(VOL,axis=0))
        edge_mask = E[:,0]
        if edge_mask.sum() == 0:
            raise Exception('There were no edges detected and edgeSep > 0')
        #Find nearest node to each vertex
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(Pts[edge_mask, :])
        distances, indices = nbrs.kneighbors(Pts)
        near_edge_mask = np.squeeze(distances) < edgeSep
        sample_mask[near_edge_mask] = 0
    prob_mask = sample_mask / sample_mask.sum()
    ss_idx = np.matrix(np.random.choice(Pts.shape[0],n,replace=False, p=prob_mask))
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


