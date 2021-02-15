#edge detection demo

import amaazetools.trimesh as tm
import amaazetools.edge_detection as ed

from mayavi import mlab
import numpy as np    
import scipy.spatial as spatial
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph


M = tm.mesh('MN_12_1_RefitFrags1to10_Frag3_yezz0003_factor4.ply')

E = ed.edge_graph_detect(M,1,1)

mlab.figure()
ed.edgeplot(M.Points,M.Triangles,E)