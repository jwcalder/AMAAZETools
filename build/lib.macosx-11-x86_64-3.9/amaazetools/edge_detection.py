#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 01:56:22 2021

@author: rileywilde
"""

from . import trimesh as tm
import numpy as np
import scipy.spatial as spatial
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph
from collections import Counter
from itertools import chain
import numpy as np

def edgeplot(P,T,E,sz = 1):
    """ Plots mesh with edges outlined.

        Parameters
        ----------
        P : (n,3) float array
            A point cloud.
        T : (m,3) int array
            List of vertex indices for each triangle in the mesh.
        E : (k,1) int array
            List of edge point indices.
        sz : float, default is 1.0
            Scaling factor for final plot.

        Returns
        -------
        None
    """

    from mayavi import mlab

    #seeking alternative to points3d.
    mlab.triangular_mesh(P[:,0],P[:,1],P[:,2],T,color =(1,0,0))
    mlab.points3d(P[E,0],P[E,1],P[E,2],color = (0,0,1), scale_mode = 'none',scale_factor = sz)
    return

def knnsearch(y, x, k) :
    """ Finds k closest points in y to each point in x.

        Parameters
        ----------
        x : (n,3) float array
            A point cloud.
        y : (m,3) float array
            Another point cloud.
        k : int
            Number of nearest neighbors one wishes to compute.

        Returns
        -------
        ordered_neighbors : (n,k) int array
            List of k nearest neighbors to each point in x.
        dist : (n,k) flaot array
            List of distances between each nearest neighbor and the corresponding point in x.
    """
    
    x, y = map(np.asarray, (x, y))
    tree =spatial.cKDTree(y)
    ordered_neighbors = tree.query(x, k)[1] #sz x, k
    
    ID = np.transpose(np.matlib.repmat(np.arange(np.shape(x)[0]), k,1))
    
    dist = np.sum((x[ID,:]-y[ordered_neighbors,:])**2,axis=2)**.5
                    
    return ordered_neighbors, dist



def pdir_metric(P,V1,V2,K1,K2,r,ktol=None):
    """ Computes principal direction metric.

        Parameters
        ----------
        P : (n,3) float array
            A point cloud.
        V1 : (n,3) float array
            First principal direction.
        V2 : (n,3) float array
            Second principal direction.
        K1 : (n,1) float array
            First principal curvature.
        K2 : (n,1) float array
            Second principal curvature.
        r : float
            Radius to use for computation.
        ktol : float, default is None
            Search tolerance for knnsearch.

        Returns
        -------
        D : (n,1) float array
            Local principal direction metric for each point.
        Dav : (n,1) float array
            Local average metric for each point.
        st : (n,2) float array
            Local standard deviation of V1 and V2.
        sigma2 : float
            Smallest square of radius of curvature.
    """
    
    #NOTE!!!!: may need to change ktol
    if ktol ==None:
        ktol = 2000;
    idx,dist = knnsearch(P,P,ktol)
    
    if np.sum(np.sum(dist<1,1)==ktol)>0:
        print('use higher knnsearch tolerance (ktol)')

    sigma2 = np.minimum(K1**-2,K2**-2);
    
    n = np.shape(P)[0]
    
    Q = np.zeros((n,1))
    D = np.zeros((n,1))
    Dav = np.zeros((n,1))
    st = np.zeros((n,2))
        
    #this could all be vectorized if we used knnsearch instead:
    for i in np.arange(n):
        neigh = idx[i,dist[i,:]<r]
        
        s1 = np.sum(V1[i,:]*V1[neigh,:],1);
        s2 = np.sum(V2[i,:]*V2[neigh,:],1);
        st[i,:] = [np.std(s1),np.std(s2)];
        Q[i] = .5*np.mean(s1 + s2);
    
    for i in np.arange(n):
        D[i] = np.sum( np.exp(-(abs(Q[i]-Q[ idx[i,dist[i,:]<r]])))/sigma2[i]**2);
        
    for i in np.arange(n):
        Dav[i] = np.mean(D[idx[i,dist[i,:]<r]]);
    
    #D(D>1.2.*Dav) = 0; %this is *experimental*
    
    return D,Dav,st,sigma2



def edge_graph_detect(M,**kwargs):
    """ Finds edge/ridge points of a mesh.

        Parameters
        ----------
        M : amaazetools.trimesh.mesh object
        k1 : float, optional
            A constant on the minimum of the inverse of principal curvatures.
        k2 : float, optional
            A constant on the mean volume.
        VOL : (n,1) float array, optional
            Spherical volume corresponding to each point in the mesh.
        K1 : (n,1) float array, optional
            First principal curvature of each point.
        K2 : (n,1) float array, optional
            Second principal curvature of each point.
        V1 : (n,3) float array, optional
            First principal direction for each point.
        V2 : (n,3) float array, optional
            Second principal direction for each point.
        rvol : float, optional
            Radius to use for svipca.
        rpdir : float, optional
            Radius to use for the principal direction metric.

        Returns
        -------
        Edges : (n,1) boolean array
            A true value corresponds to that index being an edge point.
    """ 

    #RCWO
    #parse inputs:
    if ("k1" in kwargs):
        k1 = kwargs.get('k1')
    else:
        k1 = .05
        
        
    if ("k2" in kwargs):
        k2 = kwargs.get('k2')
    else:
        k2 = 1
        
        
    if ("rvol" in kwargs):
        rvol = kwargs.get('rvol')
    else:
        rvol = 1
        
       
    if ("rpdir" in kwargs):
        rpdir = kwargs.get('rpdir')
    else:
        rpdir = 3*rvol
        
        
    if ("ktol" in kwargs):
        ktol = kwargs.get('ktol')
    else:
        ktol = 2000
    
    if ('VOL' in kwargs and 'K1' in kwargs and 'K2' in kwargs and 'V1' and kwargs and 'V2' in kwargs):
        VOL = kwargs.get('VOL')
        K1  = kwargs.get('K1')
        K2  = kwargs.get('K2')
        V1  = kwargs.get('V1')
        V2  = kwargs.get('V2')
    else:
        VOL,K1,K2,V1,V2,V3 = M.svipca([rvol])

    P = M.points
    T = M.triangles
    
    n = np.shape(P)[0]
    
    D,Da,st,sigma = pdir_metric(P,V1,V2,K1,K2,rpdir,ktol);
    
    
    #Threshold:
    l = (np.sum(st**2>k1*sigma,1)>1) & (VOL[:,0]<k2*np.mean(VOL[:,0]));
                    #^ this varies with mesh resolution, but .05 works for CT
     
    #figure out what's connected:
    E = np.vstack( (T[:,[0,1]], T[:,[1,2]], T[:,[2,0]])) #edges of T
    ll = l[E];
    E = E[ll[:,0]&ll[:,1],:];
    E = np.vstack( (E,E[:,[1,0]]) )
        
    W = sparse.coo_matrix((np.ones((np.shape(E)[0])), (E[:,0],E[:,1])),shape=(n,n))
        
    ncomp,labels = csgraph.connected_components(W,directed=False)
    
    co = Counter(labels)
    co = np.array(list(co.items()))[:,1]
    thresh = 2000;
    googlabels = np.argwhere(co>thresh)
    
    Edges = np.zeros(n,dtype=bool)
    
    for i in googlabels:
        Edges = Edges|(labels==i)
    
    
    return Edges

