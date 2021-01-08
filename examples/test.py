import amaazetools as at

m1 = at.mesh('dragon.ply')
m2 = at.mesh(m1.Points,m1.Triangles)

surf_area1 = m1.surf_area()
print('Surface area = %f'%surf_area1)

volume1 = m1.volume()
print('Volume = %f'%volume1)

xdim1 = m1.bbox()
print('Bounding box dimensions = ',end='')
print(xdim1)

surf_area2 = m2.surf_area()
print('Surface area = %f'%surf_area2)

volume2 = m2.volume()
print('Volume = %f'%volume2)

xdim2 = m2.bbox()
print('Bounding box dimensions = ',end='')
print(xdim2)

