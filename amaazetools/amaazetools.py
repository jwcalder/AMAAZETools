import numpy as np
from plyfile import PlyData, PlyElement
import scipy.sparse as sparse

def readPly(fname):
    plydata = PlyData.read(fname) 

    #Convert data formats
    tri_data = plydata['face'].data['vertex_indices']
    T = np.vstack(tri_data)
    x = plydata['vertex'].data['x']
    y = plydata['vertex'].data['y']
    z = plydata['vertex'].data['z']
    P = np.vstack((x,y,z))
    P = P.transpose()

    return P,T.astype(int)

class mesh:

    def __init__(self,*args):
        if len(args) == 1 and type(args[0]) == str:
            fname = args[0]
            self.fname = fname
            P,T = readPly(fname)
            self.Points = P
            self.Triangles = T
            self.Normals = self.face_normals(False)
            self.Centers = self.face_centers()
        elif len(args) == 2 and args[0].dtype == np.floating and args[1].dtype == np.int:
            self.fname = ""
            self.Points = args[0]
            self.Triangles = args[1]
            self.Normals = self.face_normals(False)
            self.Centers = self.face_centers()
        else:
            raise ValueError("Incorrect mesh parameters given, see documentation.")
                        
    #Returns unit normal vectors
    def face_normals(self,normalize=True):
        P1 = self.Points[self.Triangles[:,0],:]
        P2 = self.Points[self.Triangles[:,1],:]
        P3 = self.Points[self.Triangles[:,2],:]

        N = np.cross(P2-P1,P3-P1)
        if normalize:
            N = (N.T/np.linalg.norm(N,axis=1)).T
        return N
        
    #Areas of all triangles in mesh
    def tri_areas(self):
        return np.linalg.norm(self.Normals,axis=1)/2

    #Surface area of mesh
    def surf_area(self):
        return np.sum(self.tri_areas())
       
    #Centers of each face
    def face_centers(self):
        P1 = self.Points[self.Triangles[:,0],:]
        P2 = self.Points[self.Triangles[:,1],:]
        P3 = self.Points[self.Triangles[:,2],:]

        return (P1 + P2 + P3)/3  
       
    #Volume enclosed by mesh
    def volume(self):
        X = self.Centers
        X = X - np.mean(X,axis=0)
        return np.sum(X*self.Normals)/6

    def pca(self,P):
        P = P - np.mean(P,axis=0)
        vals,vecs = np.linalg.eig(P.T@P)

        return vals,vecs
 
    def weighted_pca(self,P,W):
        P = P - np.mean(W*P,axis=0)
        vals,vecs = np.linalg.eig(P.T @ (W*P))

        return vals,vecs
   
    def bbox(self):

        X = self.Centers
        n = X.shape[0]
        A = self.tri_areas()

        W = sparse.spdiags(A**2,0,n,n)
        vals,vecs = self.weighted_pca(X,W)

        vecs = vecs.T
        X = X - np.mean(W*X,axis=0)
        m1 = np.sum(X*vecs[0,:],axis=1)
        l1 = np.max(m1) - np.min(m1)
        m2 = np.sum(X*vecs[1,:],axis=1)
        l2 = np.max(m2) - np.min(m2)
        m3 = np.sum(X*vecs[2,:],axis=1)
        l3 = np.max(m3) - np.min(m3)

        return [l1,l2,l3]
        
     
    #Plot triangulated surface
    def plot_surf(self):

        from mayavi import mlab
        mlab.triangular_mesh(self.Points[:,0],self.Points[:,1],self.Points[:,2],self.Triangles)
        
    #Write a ply file
    def write_ply(self,fname):

        f = open(fname,"w")

        #Write header
        f.write('ply\n')
        f.write('format binary_little_endian 1.0\n')
        f.write('element vertex %u\n'%self.Points.shape[0])
        f.write('property double x\n')
        f.write('property double y\n')
        f.write('property double z\n')
        f.write('element face %u\n'%self.Triangles.shape[0])
        f.write('property list int int vertex_indices\n')
        f.write('end_header\n')
        f.close()

        f = open(fname,"ab")

        #write vertices
        f.write(self.Points.astype('float64').tobytes())

        #write faces
        T = np.hstack((np.ones((self.Triangles.shape[0],1))*3,self.Triangles)).astype(int)
        f.write(T.astype('int32').tobytes())

        #close file
        f.close()
       
