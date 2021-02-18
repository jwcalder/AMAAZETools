import numpy as np
from mayavi import mlab
import amaazetools.trimesh as tm
import graphlearning as gl
from amaazetools.mesh_segmentation import poisson_kmeans, canonical_labels, graph_setup

#Load points and faces
mesh = tm.mesh("HalfAnnulus.ply")

P = mesh.Points
T = mesh.Triangles

#Graph setup
n = 500 #Number of nodes
r = 0.5 #Radius
p = 1 #Weight matrix param
W,J,ss_idx,node_idx = graph_setup(mesh, n, r, p)

#Labels
g = np.array([0,1,2,3,4,5])
I0 = [5464,1919,5399,3798,9234,5519] # Manually chosen vertices, one on each face
num_classes = len(g)
#Push labels to nearest graph nodes
I = node_idx[I0].flatten()
#Poisson clustering
u, centroids = poisson_kmeans(W, num_classes, ind=I)
initialization = centroids[0]


#Interpolation to original mesh
L = np.argmax(J@u,axis=1)
L[ss_idx[0, initialization]] = 7 # Change class label of centroid initializations, to display in plot
L = canonical_labels(L)

# Plot the figure
f = mlab.figure
tm.cplotsurf(P[:,0],P[:,1],P[:,2], T, L)
