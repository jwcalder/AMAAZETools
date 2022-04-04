from amaazetools import trimesh

mesh = trimesh.load_ply('IntersectingSpheres.ply')
angle,n1,n2,C = mesh.virtual_goniometer((1.0,0.7,1.6),0.8)
print(angle)
