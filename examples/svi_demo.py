import amaazetools.trimesh as tm
from mayavi import mlab
import numpy as np
from skimage import exposure

def emph_colors(S):
    C = S - np.amin(S)
    return 1-exposure.equalize_hist(C/np.max(C),nbins=1000)

#Load mesh
mesh = tm.load_ply('dragon.ply')

#SVI
r = [.5,1] #radius of computation - must be array
vol,gamma = mesh.svi(r)
mlab.figure()
mlab.triangular_mesh(mesh.points[:,0],mesh.points[:,1],mesh.points[:,2],mesh.triangles,scalars=emph_colors(vol[:,0]))    

#SVIPCA
r = [1]
S,K1,K2,V1,V2,V3=mesh.svipca(r)
mlab.figure()
mlab.triangular_mesh(mesh.points[:,0],mesh.points[:,1],mesh.points[:,2],mesh.triangles,scalars=emph_colors(K1[:,0]))    

