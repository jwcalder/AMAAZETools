#tri_mesh.py 
#Class for working with triangulated meshes

from pyface.api import GUI
import numpy as np
from numpy import matlib
from plyfile import PlyData, PlyElement
import scipy.sparse as sparse
from mayavi import mlab
import moviepy.editor as mpy
import amaazetools.cextensions as cext

#Read a ply file
def read_ply(fname):
    plydata = PlyData.read(fname) 

    #Convert data formats
    try:
        tri_data = plydata['face'].data['vertex_indices']
    except:
        tri_data = plydata['face'].data['vertex_index']

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
            P,T = read_ply(fname)
            self.Points = P
            self.Triangles = T
            self.Normals = self.face_normals(False)
            self.Centers = self.face_centers()
        elif len(args) == 2:# and args[0].dtype.startswith('float') and args[1].dtype.startswith('int'):
            self.fname = ""
            self.Points = args[0]
            self.Triangles = args[1]
            self.Normals = self.face_normals(False)
            self.Centers = self.face_centers()
        else:
            raise ValueError("Incorrect mesh parameters given, see documentation.")
                        
    def face_normals(self,normalize=True):
        """Computes normal vectors to triangles (faces).

        Args:
            normalize: Whether or not to normalize to unit vectors. If False, then the magnitude of each vector is twice the area of the corresponding triangle. Default is True

        Returns:
            A Numpy array of size (num_tri,3) containing the face normal vectors.
        """

        P1 = self.Points[self.Triangles[:,0],:]
        P2 = self.Points[self.Triangles[:,1],:]
        P3 = self.Points[self.Triangles[:,2],:]

        N = np.cross(P2-P1,P3-P1)
        if normalize:
            N = (N.T/np.linalg.norm(N,axis=1)).T
        return N
        
    def flip_normals(self):
        """Reverses the orientation of all normal vectors in the mesh
        """
        self.Triangles = self.Triangles[:,::-1]

    #Areas of all triangles in mesh
    def tri_areas(self):
        """Computes areas of all triangles in the mesh.

        Returns:
            A Numpy array of size (num_tri,) containing the face normal vectors.
        """
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

        mlab.triangular_mesh(self.Points[:,0],self.Points[:,1],self.Points[:,2],self.Triangles)
        
    def to_gif(self,fname,color=(0.7,0.7,0.7),duration=7,fps=20,size=750):
        """Writes rotating gif

        Args:
            fname: Gif filename
            color: 3-tuple giving color of mesh (default: gray=(0,7,0.7,0.7))
            duration: length of gif in seconds (default: 7 seconds)
            fps: Frames per second (default: 20 fps)
            size: Size of gif images (default: 750)
        """

        #Make copy of points
        X = self.Points.copy()

        #PCA
        Mean = np.mean(X,axis=0)
        cov_matrix = (X-Mean).T@(X-Mean)
        Vals, P = np.linalg.eig(cov_matrix)
        idx = Vals.argsort()
        i = idx[2]
        idx[2] = idx[1]
        idx[1] = i
        Vals = Vals[idx]
        P = P[:,idx]
        P[:,2] = np.cross(P[:,0],P[:,1])

        #Rotate fragment
        X = X@P

        #Plot mesh
        f = mlab.figure(bgcolor=(1,1,1),size=(size,size))
        mlab.triangular_mesh(X[:,0],X[:,1],X[:,2],self.Triangles,color=color)

        #Function that makes gif animation
        def make_frame(t):
            mlab.view(0,180+t/duration*360)
            GUI().process_events()
            return mlab.screenshot(antialiased=True)

        animation = mpy.VideoClip(make_frame, duration=duration)
        animation.write_gif(fname, fps=fps)
        mlab.close(f)

    #Write a ply file
    def to_ply(self,fname):

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
    
    def svi(self,r,ID=None):
        """Computes spherical volume invariant.
        Args:
            P: n*3 float64 array of points
            T: m*3 int32 array of triangle point indices
            r: array of radii
            ID: optional boolean array indicating which points to compute volumes at. If [] input, all assigned true. 
        Returns:
            S: n*1 array of volumes corresponding to each point
            G: n*1 array of gamma values corresponding to each point
        """
        n = self.Points.shape[0]  #Number of vertices
        rlen = np.max(np.shape(r))
        
        if ID is None:
            ID = np.full((n), True)
        
        #Bool indicating at which vertices to compute SVI 
        Sout = np.zeros((n,rlen), dtype=np.float64) #Stores output SVI
        Gout = np.zeros((n,rlen), dtype=np.float64) #Stores output Gamma
        eps = 1.0       #Integration error tolerance
        prog = 1.0      #Show progress (1=yes, 0=no)
        
        #Output arrays
        S = np.zeros((n), dtype=np.float64)
        G = np.zeros((n), dtype=np.float64)

        #Contiguous arrays
        self.Triangles = np.ascontiguousarray(self.Triangles,dtype=np.int32)
        self.Points = np.ascontiguousarray(self.Points,dtype=np.float64)

        #Run SVI code
        for i in np.arange(0,rlen): 
            cext.svi(self.Points,self.Triangles,ID,r[i],eps,prog,S,G)
            Sout[:,i] = S
            Gout[:,i] = G
        return Sout,Gout

    def svipca(self,r):
        """Computes SVIPCA
            Args:
                P: n*3 float64 array of points
                T: m*3 int32 array of triangle point indices
                r: float scalar
                ID: optional boolean array indicating which points to compute volumes at. If [] input, all assigned true. 
            Returns:
                S: n*1 array of volumes corresponding to each point
                K1: n*1 first principle curvature
                K2: n*1 second principle curvature
                V1,V2,V3: principal directions
        """

        n = self.Points.shape[0]  #Number of vertices
        rlen = np.max(np.shape(r))
             
        eps_svi = 1.0       #Integration error tolerance for svi
        eps_pca = 1.0
        prog = 1.0      #Show progress (1=yes, 0=no)
            
        ID = np.full((n), True) #Bool indicating at which vertices to compute SVI 
                
        Sout = np.zeros((n,rlen), dtype=np.float64) #Stores output SVI
        K1 = np.zeros((n,rlen), dtype=np.float64)                              
        K2 = np.zeros((n,rlen), dtype=np.float64)
        V1 = np.zeros((n,3*rlen), dtype=np.float64)
        V2 = np.zeros((n,3*rlen), dtype=np.float64)
        V3 = np.zeros((n,3*rlen), dtype=np.float64)
            
        Z1 = np.zeros((n,1), dtype=np.float64)
        Z = np.zeros((n), dtype=np.float64)
        S = np.zeros((n), dtype=np.float64)
        M = np.zeros((9*n), dtype=np.float64) #Stores output PCA matrix
            
        VN = vertex_normals(self.Points,self.Triangles)
        Z3 = np.zeros((3), dtype=np.float64)
            
        #indexing for output:
        I = np.arange(0,n)
        I = I[I]
            
        #Contiguous arrays
        self.Triangles = np.ascontiguousarray(self.Triangles,dtype=np.int32)
        self.Points = np.ascontiguousarray(self.Points,dtype=np.float64)

        for k in np.arange(0,rlen):
            cext.svipca(self.Points,self.Triangles,ID,r[k],eps_svi,eps_pca,prog,S,M)
            Sout[:,k] = S
            
            l = np.arange(3*k,3*k+3)
                
            L1 = np.zeros((n), dtype=np.float64)
            L2 = np.zeros((n), dtype=np.float64)
            L3 = np.zeros((n), dtype=np.float64)
            
            for i in I:
                A = M[np.arange(9*i,9*(i+1))]
                D,V = np.linalg.eig([A[[0,1,2]],A[[3,4,5]],A[[6,7,8]]])
        
                a = VN[i,:]@V
        
                loc = np.where(np.abs(a)==max(np.abs(a)))
                
                if loc == 0:
                    L1[i] = D[1]
                    L2[i] = D[2]
                    L3[i] = D[0]
                    V1[i,l] = V[:,1]
                    V2[i,l] = V[:,2]
                    V3[i,l] = V[:,0]
                elif loc==1:
                    L1[i] = D[0]
                    L2[i] = D[2]
                    L3[i] = D[1]
                    V1[i,l] = V[:,0]
                    V2[i,l] = V[:,1]
                    V3[i,l] = V[:,2]
                else:
                    L1[i] = D[0]
                    L2[i] = D[1]
                    L3[i] = D[2]
                    V1[i,l] = V[:,0]
                    V2[i,l] = V[:,1]
                    V3[i,l] = V[:,2]
             
            Kdiff = (L1-L2)*24/(np.pi*r[k]**6);
            Ksum = 16*np.pi*(r[k]**3)/3 - 8*S/(np.pi*r[k]**4)
            k1t = (Kdiff + Ksum)/2;
            k2t = (Ksum - Kdiff)/2;
                
            #want to ensure k1>k2:
            J = np.double(k1t > k2t); #logical
            
            #the broadcasting here is super obnoxious:
            K1[:,k]= J*k1t + (1-J)*k2t #if k1 max, keep it as k1, else swap
            
            K2[:,k] = (1-J)*k1t + J*k2t 
            v1t = V1[:,l] 
            v2t = V2[:,l]
            V1[:,l] = J[:,None]*v1t + (1-J[:,None])*v2t #so V1 corresponds to K1
            V2[:,l] = (1-J[:,None])*v1t + J[:,None]*v2t
                   
            #now for quality control: if volume is not defined:
                
            #note: the broadcasting operations can be improved here
            visnegative = S != -1;
            vvneg = matlib.repmat(np.double(visnegative[:,None]),1,3)
            K1[visnegative,k] = 0; 
            K2[visnegative,k] = 0;
            V1[:,l] = vvneg*V1[:,l]    
            V2[:,l] = vvneg*V2[:,l]
            V3[:,l] = vvneg*V3[:,l]
            
            vecneg = -2*(np.double(np.sum(V3[:,l]*VN<0,1)<0)-.5)
            vecneg = matlib.repmat(vecneg[:,None],1,3)
            V3[:,l] = vecneg*V3[:,l];
            V2[:,l] = vecneg*V2[:,l];
            V1[:,l] = vecneg*V1[:,l];
                  
            #implementing right hand rule:
            rhr = -2*(np.double(np.sum(V3[:,l]*np.cross(V1[:,l],V2[:,l]),1) < 0)-.5);
            rhr = matlib.repmat(rhr[:,None],1,3)
            V1[:,l] = rhr*V1[:,l]   
        return Sout,K1,K2,V1,V2,V3
     
#######################################################################################################
#All code below needs to be converted to object oriented format
#######################################################################################################

#Converts from (x,y,z) to index of closest point
def get_index(point,P):
    if type(point) in [np.int,np.int32,np.int64]:
        point_ind=point
    elif type(point) == np.ndarray and len(point)==3:
        point_ind = np.argmin(np.linalg.norm(P - point,axis=1))
    elif type(point) in [tuple,list] and len(point)==3:
        point_ind = np.argmin(np.linalg.norm(P - np.array(point),axis=1))
    else:
        sys.exit("'point' must be an integer index, or a length 3 list, tuple, or numpy ndarray (x,y,z)")
    return point_ind

def edge_points(P,u,k=7,return_mask=False,number=None):
    num_verts = P.shape[0]

    I,J,D = gl.knnsearch(P,k)
    W = gl.weight_matrix(I,J,D,k,f=lambda x : np.ones_like(x),symmetrize=False)
    d = gl.degrees(W)
    mask = d*u != W@u

    #Select a few points spaced out along edge
    if number is not None:
        edge_ind = np.arange(num_verts)[mask]
        edge_points = P[mask,:]
        num_edge_points = len(edge_points)

        #PCA
        mean = np.mean(edge_points,axis=0)
        cov = (edge_points-mean).T@(edge_points-mean)
        l,v = sparse.linalg.eigs(cov,k=1,which='LM')
        proj = (edge_points-mean)@v.real

        #Sort along princpal axis
        sort_ind = np.argsort(proj.flatten())
        dx = (num_edge_points-1)/(number-1)
        spaced_edge_ind = edge_ind[sort_ind[np.arange(0,num_edge_points,dx).astype(int)]]
        mask = np.zeros(num_verts,dtype=bool)
        mask[spaced_edge_ind]=True

    if return_mask:
        return mask.astype(int)
    else: #return indices
        return np.arange(num_verts)[mask]

def geodesic_patch(point,P,r,k=7,return_mask=False):

    num_verts = P.shape[0]

    I,J,D = gl.knnsearch(P,k)
    W = gl.dist_matrix(I,J,D,k)

    point_ind = get_index(point,P)
    dist = gl.cDijkstra(W,np.array([point_ind]),np.array([0]))
    mask = dist < r

    if return_mask:
        return mask.astype(int)
    else: #return indices
        return np.arange(num_verts)[mask]

def cplotsurf(x,y,z,triangles,C=-1):
    if C.any == -1: #if no C given
        C = np.ones((len(x),1))
        
    n = len(np.unique(C))
    C = C.astype(int)
    if n>20:
        mesh = mlab.triangular_mesh(x,y,z,triangles,scalars=C)
    else:
        col = (np.arange(1,n+1)) / n
        colors = col[C-1]
        mesh = mlab.triangular_mesh(x,y,z,triangles,scalars=colors)
        
    return mesh

#Withness is a measure of how well 1D data clusters into two groups
def withiness(x):

   x = np.sort(x)
   sigma = np.std(x)
   n = x.shape[0]
   v = np.zeros(n-1,)
   for i in range(n-1):
      x1 = x[:(i+1)]
      x2 = x[(i+1):]
      m1 = np.mean(x1);
      m2 = np.mean(x2);
      v[i] = (np.sum((x1-m1)**2) + np.sum((x2-m2)**2))/(sigma**2*n);
   ind = np.argmin(v)
   m = x[ind]
   w = v[ind]
   return w,m

#Plot triangulated surface
def plotsurf(P,T,C=None):

    if C is None:
        mlab.triangular_mesh(P[:,0],P[:,1],P[:,2],T)
    else:
        mlab.triangular_mesh(P[:,0],P[:,1],P[:,2],T,scalars=C)

#Write a ply file
def write_color_ply(P,T,color,fname):

    f = open(fname,"w")

    #Write header
    f.write('ply\n')
    f.write('format binary_little_endian 1.0\n')
    f.write('element vertex %u\n'%P.shape[0])
    f.write('property double x\n')
    f.write('property double y\n')
    f.write('property double z\n')
    f.write('property uchar red\n')
    f.write('property uchar green\n')
    f.write('property uchar blue\n')
    f.write('element face %u\n'%T.shape[0])
    f.write('property list int int vertex_indices\n')
    f.write('end_header\n')
    f.close()

    f = open(fname,"ab")

    #write vertices
    for i in range(P.shape[0]):
        f.write(P[i,:].astype('float64').tobytes())
        f.write(color[i,:].astype('uint8').tobytes())

    #write faces
    T = np.hstack((np.ones((T.shape[0],1))*3,T)).astype(int)
    f.write(T.astype('int32').tobytes())

    #close file
    f.close()

#vertex-triangle adjacencey matrix
#Returns num_verts x num_tri sparse matrix F with F_ij = 1 if vertex i belongs to triangle j
#If normalize=True, then each row is divided by the number of adjacent triangles, 
#so F can be used to interplate from triangles to vertices
def tri_vert_adj(P,T,normalize=False):
    
    num_verts = P.shape[0]
    num_tri = T.shape[0]
    ind = np.arange(num_tri)

    I = np.hstack((T[:,0],T[:,1],T[:,2]))
    J = np.hstack((ind,ind,ind))
    F = sparse.coo_matrix((np.ones(len(I)), (I,J)),shape=(num_verts,num_tri)).tocsr()

    if normalize:
        num_adj_tri = F@np.ones(num_tri)
        F = sparse.spdiags(1/num_adj_tri,0,num_verts,num_verts)@F

    return F

#Returns unit normal vectors to vertices (averaging adjacent faces and normalizing)
def vertex_normals(P,T):

    m = mesh(P,T)
    fn = m.face_normals()
    F = tri_vert_adj(P,T)
    vn = F@fn
    norms = np.linalg.norm(vn,axis=1)
    norms[norms==0] = 1

    return vn/norms[:,np.newaxis]

#Power method to find principle eigenvector
def power_method(A,tol=1e-12):

    n = A.shape[0]
    x = np.random.rand(n,1)
    err = 1
    i = 1
    while err > tol:
        x = A@x
        x = x/np.linalg.norm(x)
        l = np.transpose(x)@A@x
        err = np.linalg.norm(A@x - l*x)
        i = i+1
    return l,x

def pca_smallest_eig_powermethod(X,center=True):

    if center:
        m = np.mean(X,axis=0)
        cov = np.transpose(X-m)@(X-m)/X.shape[0]
    else:
        cov = np.transpose(X)@X/X.shape[0]
    lmax,v = power_method(cov)
    w,v = np.linalg.eig(cov)
    l,v = power_method(cov - (lmax+1)*np.eye(3))
    return v.flatten()

def pca_smallest_eig(X,center=True):

    if center:
        m = np.mean(X,axis=0)
        cov = np.transpose(X-m)@(X-m)
    else:
        cov = np.transpose(X)@X
    w,v = np.linalg.eig(cov)
    i = np.argmin(w)
    return v[:,i]

#Virtual goniometer
#Input:
#   point = location to take measurement (index, or (x,y,z) coordinates)
#   P = nx3 numpy array of vertices of mesh
#   T = mx3 numpy array of triangles in mesh
#Output:
#   theta = Angle
#   n1,n2 = Normal vectors between two patches (theta=angle(n1,n2))
#   C = Clusters (C=1 and C=2 are the two detected clusters, C=0 indicates outside of patch)
#   E (optional) = array of indices of edge points
def VirtualGoniometer(point,P,T,r,k=7,SegParam=2,return_edge_points=False,number_edge_points=None):

    patch_ind = geodesic_patch(point,P,r,k=k)
    patch = P[patch_ind,:]
    normals = vertex_normals(P,T)[patch_ind,:]
    theta,n1,n2,C_local = __VirtualGoniometer__(patch,normals,SegParam=SegParam)

    C = np.zeros(P.shape[0])
    C[patch_ind] = C_local


    if return_edge_points:
        E = edge_points(patch,C_local,k=k,number=number_edge_points)
        E = patch_ind[E]
        return theta,n1,n2,C,E
    else:
        return theta,n1,n2,C

#Virtual goniometer (internal function)
#Input:
#   P = nx3 numpy array of vertices of points in patch
#   N = nx3 array of vertex normals
#   Can also use N as face normals, and P as face centroids
#Output:
#   theta = Angle
#   n1,n2 = Normal vectors between two patches (theta=angle(n1,n2))
#   C = Clusters (C=1 and C=2 are the two detected clusters)
def __VirtualGoniometer__(P,N,SegParam=2,UsePCA=True,UsePower=False):

    n = P.shape[0]

    if UsePower:
        N1 = pca_smallest_eig_powermethod(N,center=False)
        N1 = np.reshape(N1,(3,))
    else:
        N1 = pca_smallest_eig(N,center=False)

    N2 = np.sum(N,axis=0)
    v = np.cross(N1,N2)
    v = v/np.linalg.norm(v)
    
    m = np.mean(P,axis=0)
    dist = np.sqrt(np.sum((P - m)**2,axis=1))
    i = np.argmin(dist)
    radius = np.max(dist)
    D = (P - P[i,:])/radius

    #The SegParam=2 is just hand tuned. Larger SegParam encourages the clustering to split the patch in half
    #SegParam=0 is the previous version of the virtual goniometer
    x = np.sum(v*N,axis=1) + SegParam*np.sum(v*D,axis=1)

    #Clustering
    w,m = withiness(x)
    C = np.zeros(n,)
    C[x>m] = 1
    C[x<=m] = 2

    if UsePCA:

        P1 = P[C==1,:]
        P2 = P[C==2,:]
        if UsePower:
            n1 = pca_smallest_eig_powermethod(P1)
            n2 = pca_smallest_eig_powermethod(P2)
        else:
            n1 = pca_smallest_eig(P1)
            n2 = pca_smallest_eig(P2)

        s1 = np.mean(N[C==1,:],axis=0)
        if np.dot(n1,s1) < 0:
            n1 = -n1

        s2 = np.mean(N[C==2,:],axis=0)
        if np.dot(n2,s2) < 0:
            n2 = -n2
    else: #Use average of surface normals

        n1 = np.average(N[C==1,:],axis=0)
        n1 = n1/np.linalg.norm(n1)
        n2 = np.average(N[C==2,:],axis=0)
        n2 = n2/np.linalg.norm(n2)
        
    #Angle between
    theta = 180-np.arccos(np.dot(n1,n2))*180/np.pi;

    return theta,n1,n2,C






