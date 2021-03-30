#edge detection demo

import amaazetools.trimesh as tm
#import amaazetools.edge_detection as ed
import amaazetools.poisson_cluster as poi


from mayavi import mlab
import numpy as np
import scipy.spatial as spatial
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph
import math

#load mesh:
#M = tm.mesh('MN_12_1_RefitFrags1to10_Frag3_yezz0003_factor4.ply')
M = tm.load_ply('sample_mesh.ply')
E = M.edge_graph_detect() #using defaults for bone scans
mlab.figure()
M.plotsurf(C=np.double(E))


#try another mesh that has different scaling:
M = tm.load_ply('HalfAnnulus.ply')

#option 1: compute inside program:
#E = M.edge_graph_detect(rpdir=.6,rvol = .2)

#option 2: use already calculated values
VOL,K1,K2,V1,V2,V3 = M.svipca([.2])
E = M.edge_graph_detect(rpdir=.6,VOL=VOL,K1=K1,K2=K2,V1=V1,V2=V2)

#option 3: use edge_detection package (same usage, only M input)
#E = ed.edge_graph_detect(M,rvol = .2, rpdir =.6)


mlab.figure()
M.plotsurf(C=np.double(E))


