import numpy as np
from mayavi import mlab
import amaazetools.trimesh as tm
from amaazetools.mesh_segmentation import poisson_kmeans, canonical_labels, graph_setup
import scipy.sparse.csgraph as csgraph
import scipy.sparse as sparse
import graphlearning as gl

# Load points and faces
mesh = tm.mesh("HalfAnnulus.ply")
P = mesh.Points
T = mesh.Triangles

# Graph setup
n = 1000  # Number of nodes
r = 0.5  # Radius
p = 1  # Weight matrix param
edgeSep = 0.15  # Ensure sampled vertices are at least this far from any edge
seed = 9  # Interesting cases to study where clusters merge: seed = 3, 7
W, J, ss_idx, node_idx = graph_setup(mesh, n, r, p, edgeSep=edgeSep, seed=seed)
y_true = np.load("HalfAnnulus_Labels.npz")["labels"]
y_true_ss = y_true[ss_idx].flatten()
ncomps, labels = csgraph.connected_components(W, directed=False)
print(f"The graph has {ncomps} connected components")
try:
    L = gl.graph_laplacian(W, norm="none")
    vals, vec = sparse.linalg.eigsh(L, k=2, sigma=0, which="LM")
    print(f"First two eigenvalues of L: {sorted(vals)}")
except:
    print("Could not find eigenvalues")
I0 = [5464, 1919, 5399, 3798, 9234, 5519]  # Manually chosen vertices, one on each face
num_classes = 6
# Push labels to nearest graph nodes
ind = node_idx[I0].flatten()
# Poisson clustering
# weights = [(y_true_ss == l).mean() for l in range(num_classes)]
# weights.sort()
results = []
trials = 15
for _ in range(trials):
    result = poisson_kmeans(
        W, num_classes, ind=None, print_info=False, algorithm="poisson2"
    )
    results.append(result)
results.sort(key=lambda x: -x[4])
u, u_list, medians, centroids, energy = results[0]
print(f"best energy = {energy}")
l = np.argmax(u, axis=1)
# Interpolation to original mesh
L = J @ l
L[
    ss_idx[0, centroids[0]]
] = 7  # Change class label of centroid initializations, to display in plot
# L = canonical_labels(L)

# Plot the figure
f = mlab.figure
tm.plotsurf(P, T, L)
