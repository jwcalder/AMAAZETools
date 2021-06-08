# import libraries
import numpy as np
import amaazetools.trimesh as tm

# load desired mesh
mesh = tm.load_ply('HalfAnnulus.ply')

# labels and indices of user-selected points
g = np.array([0,1,2,3,4,5])
I = np.array([5464,1919,5399,3798,9234,5519])

# using default n,r,p values
mesh.poisson_label(g,I)
print(mesh.poisson_labels)

# using custom n,r,p values
n = 1000 # default num neighbors n is 5000
r = 1.0  # default radius r is 0.5
p = 2.0  # default weight paramter p is 1

# generate graph
mesh.graph_setup(n,r,p)

# labels and indices of user-selected points
g = np.array([0,1,2,3,4,5])
I = np.array([5464,1919,5399,3798,9234,5519])

# run segmentation
mesh.poisson_label(g,I)
print(mesh.poisson_labels)

# can visualize labelling via the following (commented to account for mayavi install issues)
# mesh.plotsurf(mesh.poisson_labels)
# mesh.to_gif(desired_file_name, mesh.poisson_labels)
