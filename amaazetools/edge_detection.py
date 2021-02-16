#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 01:56:22 2021

@author: rileywilde
"""

import amaazetools.trimesh as tm
from mayavi import mlab
import numpy as np    
import scipy.spatial as spatial
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph
from collections import Counter
from itertools import chain
import numpy as np

def edgeplot(P,T,E): 
    #seeking alternative to points3d. 
    mlab.triangular_mesh(P[:,0],P[:,1],P[:,2],T,color =(1,0,0))
    mlab.points3d(P[E,0],P[E,1],P[E,2],color = (0,0,1), scale_mode = 'none',scale_factor = 1)
    return

def knnsearch(y, x, k) :
    #finds k closest points in y to each point in x
    
    x, y = map(np.asarray, (x, y))
    tree =spatial.cKDTree(y)    
    ordered_neighbors = tree.query(x, k)[1] #sz x, k
    
    ID = np.transpose(np.matlib.repmat(np.arange(np.shape(x)[0]), k,1))
    
    dist = np.sum((x[ID,:]-y[ordered_neighbors,:])**2,axis=2)**.5
                    
    return ordered_neighbors, dist



def pdir_metric(P,V1,V2,K1,K2,r,ktol=None):
    #returns:
    #D is local principal metric, Dav is local average
    #st is local std of V1 and V2, 
    #sigma2 is smallest square of radius of curavture
    
    #NOTE!!!!: may need to change ktol 
    if ktol ==None:
        ktol = 1000;
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



def edge_graph_detect(M,k1=None,k2=None,VOL=None,K1=None,K2=None,V1=None,V2=None):
    '''
    finds edge/ridge points
    #RCWO

    INPUTS:::: 
        M - mesh structure with triangles & points

    #everything else is optional:
        k1  - constant on minimum of inverse of principal curvatures
        k2  - constant on mean volume 
        VOL - spherical volumen invariant: n*1 array
        K1  - first principal curvature: n*1 array
        K2  - second principal curvature: n*1 array
        V1  - first principal direction: n*3 array
        V2  - second principal direction: n*3 array

    OUTPUT::::
        Edges - n*1 logical array of what points are edges: 1= edge point, 0 = not
    '''
    if k1==None:
        k1 = .05;
    if k2==None:
        k2 = 1;
    
    if (VOL==None)|(K1==None)|(K2==None)|(V1==None)|(V2==None):
        [VOL,K1,K2,V1,V2,V3] = M.svipca([1]);
    
    P = M.Points;
    T = M.Triangles;
    
    n = np.shape(P)[0]
    
    D,Da,st,sigma = pdir_metric(P,V1,V2,K1,K2,3);
    
    
    #Threshold:
    l = (np.sum(st**2>k1*sigma,1)>1) & (VOL[:,0]<k2*np.mean(VOL[:,0]));
                    #^ this varies with mesh resolution, but .05 works for CT
     
    #figure out what's connected: easier to use built-in graph code:      
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


 
