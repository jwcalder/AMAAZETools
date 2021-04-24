#tri_mesh.py 
#Class for working with triangulated meshes

from pyface.api import GUI
import numpy as np
from numpy import matlib
from plyfile import PlyData, PlyElement
import scipy.sparse as sparse
from mayavi import mlab
import moviepy.editor as mpy
from . import svi
from . import edge_detection 

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
            self.points = P
            self.triangles = T
            self.normals = self.face_normals(False)
            self.centers = self.face_centers()
        elif len(args) == 2:# and args[0].dtype.startswith('float') and args[1].dtype.startswith('int'):
            self.fname = ""
            self.points = args[0]
            self.triangles = args[1]
            self.normals = self.face_normals(False)
            self.centers = self.face_centers()
        else:
            raise ValueError("Incorrect mesh parameters given, see documentation.")
                        
    def face_normals(self,normalize=True):
        """Computes normal vectors to triangles (faces).

        Args:
            normalize: Whether or not to normalize to unit vectors. If False, then the magnitude of each vector is twice the area of the corresponding triangle. Default is True

        Returns:
            A Numpy array of size (num_tri,3) containing the face normal vectors.
        """

        P1 = self.points[self.triangles[:,0],:]
        P2 = self.points[self.triangles[:,1],:]
        P3 = self.points[self.triangles[:,2],:]

        N = np.cross(P2-P1,P3-P1)
        if normalize:
            N = (N.T/np.linalg.norm(N,axis=1)).T
        return N
        
    def flip_normals(self):
        """Reverses the orientation of all normal vectors in the mesh
        """
        self.triangles = self.triangles[:,::-1]

    #Areas of all triangles in mesh
    def tri_areas(self):
        """Computes areas of all triangles in the mesh.

        Returns:
            A Numpy array of size (num_tri,) containing the face normal vectors.
        """
        return np.linalg.norm(self.normals,axis=1)/2

    #Surface area of mesh
    def surf_area(self):
        return np.sum(self.tri_areas())
       
    #Centers of each face
    def face_centers(self):
        P1 = self.points[self.triangles[:,0],:]
        P2 = self.points[self.triangles[:,1],:]
        P3 = self.points[self.triangles[:,2],:]

        return (P1 + P2 + P3)/3  
       
    #Volume enclosed by mesh
    def volume(self):
        X = self.centers
        X = X - np.mean(X,axis=0)
        return np.sum(X*self.normals)/6

    def pca(self,P):
        P = P - np.mean(P,axis=0)
        vals,vecs = np.linalg.eig(P.T@P)

        return vals,vecs
 
    def weighted_pca(self,P,W):
        P = P - np.mean(W*P,axis=0)
        vals,vecs = np.linalg.eig(P.T @ (W*P))

        return vals,vecs
   
    def bbox(self):

        X = self.centers
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

        mlab.triangular_mesh(self.points[:,0],self.points[:,1],self.points[:,2],self.triangles)
        
    def to_gif(self,fname,color = [],duration=7,fps=20,size=750,histeq = True):
        """Writes rotating gif

        Args:
            fname: Gif filename
            color: 1D array the length of Self.Points
            duration: length of gif in seconds (default: 7 seconds)
            fps: Frames per second (default: 20 fps)
            size: Size of gif images (default: 750)
        """
        
        from skimage import exposure
        
        
        
        #Make copy of points
        X = self.points.copy()
        
        if np.shape(color)[0] == np.shape(X)[0]: #scalars for plot
            opt = 2
            if histeq:
                color = color - np.amin(color)
                color = 1-exposure.equalize_hist(color/np.max(color),nbins=1000)
                
            if np.shape(np.shape(color))[0]>1: #handle input
                color = color[:,0]
        elif max(np.shape(color)) == 3: #single rgb color
            opt = 1
        else : #not input - default to single color
            color = (0.7,0.7,0.7)
            opt = 1
        
        
        
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
        if opt == 1:
            mlab.triangular_mesh(X[:,0],X[:,1],X[:,2],self.triangles,color=color)
        else :
            mlab.triangular_mesh(X[:,0],X[:,1],X[:,2],self.triangles,scalars=color)

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
        f.write('element vertex %u\n'%self.points.shape[0])
        f.write('property double x\n')
        f.write('property double y\n')
        f.write('property double z\n')
        f.write('element face %u\n'%self.triangles.shape[0])
        f.write('property list int int vertex_indices\n')
        f.write('end_header\n')
        f.close()

        f = open(fname,"ab")

        #write vertices
        f.write(self.points.astype('float64').tobytes())

        #write faces
        T = np.hstack((np.ones((self.triangles.shape[0],1))*3,self.triangles)).astype(int)
        f.write(T.astype('int32').tobytes())

        #close file
        f.close()
        
    

    def svi(self,r,ID=None):
        """Computes spherical volume invariant.
        Args:
            r: array of radii
            ID: optional boolean array indicating which points to compute volumes at. If [] input, all assigned true. 
        Returns:
            S: n*1 array of volumes corresponding to each point
            G: n*1 array of gamma values corresponding to each point
        """
   
        return svi.svi(self.points,self.triangles,r,ID=ID)

    def svipca(self,r):
        """Computes SVIPCA
            Args:
                r: float scalar
            Returns:
                S: n*1 array of volumes corresponding to each point
                K1: n*1 first principle curvature
                K2: n*1 second principle curvature
                V1,V2: principal directions
        """

        return svi.svipca(self.points,self.triangles,r)

#######################################################################################################
#All code below needs to be converted to object oriented format
#######################################################################################################

def edge_graph_detect(M,**kwargs):
    """ Detects edges using SVIPCA and principal direction metric.
        Every input but M is optional. Example usages:
        Default (CT scan scale parameters): E = edge_graph_detect(M)
        Computing SVIPCA at radius 3, pdir metric at 5: E = edge_graph_detect(M,rvol = 3, rpdir = 5)
        Use existing SVIPCA data: E = edge_graph_detect(M,VOL=VOL,K1=K1,K2=K2,V1=V1,V2=V2)
        Args:
            M: mesh structure
            k1: constant for thresholding principal directions. Default: .05
            k2: constant on voleue for thresholding volume. Default: 1
            VOL: spherical volumen invariant: n*1 array
            K1: first principal curvature: n*1 array
            K2: second principal curvature: n*1 array
            V1: first principal direction: n*3 array
            V2: second principal direction: n*3 array
            rvol: radius to use for SVIPCA. Default: 1
            rpdir: radius to use for principal direction metric. Default: 3*rvol
            ktol: k tolerance in pdir knn search
        Returns:
            E: n*0 boolean array of detected edge points. 1 = edge point.
    """
    #note: this resists being converted to self structure - requires self object passed into function...
    #have tried to remedy by placing svipca execution here and passing points & triangles into modified
    #version... has not worked for some undetermined reason



    return edge_detection.edge_graph_detect(M,**kwargs)



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






