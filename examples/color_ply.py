import amaazetools.trimesh as tm
import numpy as np

#Load mesh
mesh = tm.load_ply('dragon.ply')

#SVI computation
vol,gamma = mesh.svi([1])

#Write to color ply file 
mesh.to_ply('dragon_svi.ply',c=vol.flatten())

