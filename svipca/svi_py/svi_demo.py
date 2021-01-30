import numpy as np
from skimage import exposure
from plyfile import PlyData, PlyElement
from c_code import svi_module
from mayavi import mlab


#Read PLY file
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

#Parameters for SVI call
ID = np.full((n), True)  #Bool indicating at which vertices to compute SVI 
S = np.zeros((n), dtype=np.float64) #Stores output SVI
G = np.zeros((n), dtype=np.float64) #Stores output Gamma
r = 1           #Radius
eps = 1.0       #Integration error tolerance
prog = 1.0      #Show progress (1=yes, 0=no)

#Run SVI code
svi_module.svi(P,T,ID,r,eps,prog,S,G)

#Normalize colors
C = S - np.amin(S)
C = C/np.max(C)
Se = exposure.equalize_hist(C,nbins=1000)
C = 1-Se

#Plotting
mlab.triangular_mesh(x,y,z,T,scalars=C)
