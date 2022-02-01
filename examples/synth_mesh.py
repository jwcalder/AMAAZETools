from amaazetools import trimesh

num_pts = 10000
angle = 10
mesh = trimesh.synth_mesh(angle, num_pts)

print('Saving mesh to file...')
mesh.to_ply('synth_mesh_angle_%d_num_pts_%d.ply'%(angle,num_pts))

vg_angle,n1,n2,C = mesh.virtual_goniometer((0,0,0),0.3,k=7,SegParam=2)
print('Virtual Goniometer Angle: ',vg_angle)

