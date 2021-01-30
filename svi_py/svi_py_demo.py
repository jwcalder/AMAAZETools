#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 00:27:52 2021

@author: rileywilde
"""
import svi_pyfunctions as svipy
import numpy as np
import scipy.sparse as sparse
from skimage import exposure
from plyfile import PlyData, PlyElement
from mayavi import mlab

#Read PLY file:
#plydata = PlyData.read('meshes/dragon.ply')  #High resolution
plydata = PlyData.read('meshes/dragon_d1.ply') #Medium resolution
#plydata = PlyData.read('meshes/dragon_d2.ply') #Low resolution

#Convert data formats
tri_data = plydata['face'].data['vertex_indices']
T = np.vstack(tri_data)
T = np.ascontiguousarray(T,dtype=np.int32)
x = plydata['vertex'].data['x']
y = plydata['vertex'].data['y']
z = plydata['vertex'].data['z']
P = np.vstack((x,y,z))
P = P.transpose()
P = np.ascontiguousarray(P,dtype=np.float64)
n = P.shape[0]  #Number of vertices
m = T.shape[0]  #Number of faces

r = [.5,1,2,3] #radius of computation - must be array
VOL = svipy.svi(P,T,r,[])

r = [.5]
S,K1,K2,V1,V2,V3=svipy.svipca(P,T,r,[])

mlab.triangular_mesh(P[:,0],P[:,1],P[:,2],T,scalars=S[:,0])    

