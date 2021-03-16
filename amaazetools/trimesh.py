#tri_mesh.py 
#Class for working with triangulated meshes

import graphlearning as gl
import moviepy.editor as mpy
import numpy as np
from numpy import matlib
from plyfile import PlyData, PlyElement
from pyface.api import GUI
import scipy.sparse as sparse
import scipy.spatial as spatial
from . import svi
import sys
import urllib.request as url

#Enable plotting if possible
try:
    from mayavi import mlab
except:
    print("Could not find mayavi, plotting functionality will be disabled.")

#Non-Class Specific Functions

def withiness(x):
    """Computes withiness (how well 1-D data clusters into two groups).

        Args:
            x: A 1-D collection of data.
        
        Returns:
            w: The withiness of the data as a float.
            m: The point at which to split the data into 2 clusters as a float.
        """    
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

def pca(P):
    """Computes principal component analysis (PCA) on a point cloud P.

        Args:
            P: A point cloud in the form of an (n,d) array of coordinates. 
        
        Returns:
            vals: A Numpy array of size (d,) of the variances among each principal component.
            vecs: A Numpy array of size (d,d) of the principal component vectors.
        """
    P = P - np.mean(P,axis=0)
    vals,vecs = np.linalg.eig(P.T@P)

    return vals,vecs
 
def weighted_pca(P,W):
    """Computes weighted principal component analysis (PCA) on a point cloud P.

        Args:
            P: A point cloud in the form of an (n,d) array of coordinates. 
            W: An array of size (n,1) containing the weights of the points.
        
        Returns:
            vals: A Numpy array of size (d,) of the variances among each principal component.
            vecs: A Numpy array of size (d,d) of the principal component vectors.
        """
    P = P - np.mean(W*P,axis=0)
    vals,vecs = np.linalg.eig(P.T@(W*P))

    return vals,vecs

#Power method to find principle eigenvector
def power_method(A,tol=1e-12):
    """Computes the smallest (in absolute value) eigenvalue and its corresponding eigenvector using the power method.

        Args:
            A: A square matrix that one wishes to find the smallest (in absolute value) eigenvalue and corresponding eigenvector of.
            tol: The desired tolerance threshold after which to stop iteration. Default is 1e-12.
        
        Returns:
            l: The smallest (in absolute value) eigenvalue of A, as a float.
            x: A Numpy array of size (n,1) containing the eigenvector corresponding to the smallest (in absolute value) eigenvalue of A.
        """
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
    """Computes the last principal component of a point cloud X using the power method.

        Args:
            X: A point cloud in the form of an (n,3) array of coordinates. 
            center: Optional boolean that centers data if True (by subtracting mean from data) and does not if False. Default is True.
        
        Returns:
            A Numpy array of size(3,) containing the last principal component vector.
        """
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
    """Computes the last principal component of a point cloud X.

        Args:
            X: A point cloud in the form of an (n,3) array of coordinates. 
            center: Optional boolean that centers data if True (by subtracting mean from data) and does not if False. Default is True.
        
        Returns:
            A Numpy array of size(3,) containing the last principal component vector.
        """
    if center:
        m = np.mean(X,axis=0)
        cov = np.transpose(X-m)@(X-m)
    else:
        cov = np.transpose(X)@X
    w,v = np.linalg.eig(cov)
    i = np.argmin(w)
    return v[:,i]

### Mesh Class ###

#Read a ply file
def read_ply(fname):
    """Reads the vertex and triangle data stored in a .ply file.

        Args:
            fname: Name of the file to read from.
        
        Returns:
            P: A Numpy array of size (num_verts,3) containing the coordinates of the vertices of the mesh.
            T: A Numpy array of size (num_tri,3) containing the indices of the triangles of the mesh.
        """
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

#Load a ply file
def load_ply(path):
    """Loads a file path or url and creates a mesh object.

        Args:
            path: URL or file path at which to access .ply file.
        
        Returns:
            A mesh object generated from a .ply file found at the file path location.
        """
    try:
      url.urlopen(path)
      is_url = True
    except:
      is_url = False

    if is_url:
      fname = path.rsplit('/', 1)[-1]
      url.urlretrieve(path,fname)
    else:
      fname = path

    points,triangles = read_ply(fname)
    return mesh(points,triangles)

class mesh:

    def __init__(self,*args):
        self.points = args[0]
        self.triangles = args[1]
        self.unit_norms = None
        self.norms = None 
        self.centers = None 
        self.knn_I = None
        self.knn_J = None
        self.knn_D = None
        self.tri_vert_adj_I = None
        self.tri_vert_adj_J = None

    #Get number of vertices
    def num_verts(self):
        """Computes number of vertices in the mesh.

        Returns:
            The number of vertices in the mesh as an integer.
        """      
        return self.points.shape[0]

    #Get number of triangles
    def num_tri(self):
        """Computes number of triangles in the mesh.

        Returns:
            The number of triangles in the mesh as an integer.
        """      
        return self.triangles.shape[0]

    #Converts from (x,y,z) to index of closest point
    def get_index(self,point):
        """Computes the index of a given point.

        Args:
            point: A vertex in the mesh, specified by either an integer index or its coordinates.
        
        Returns:
            The index of the given point as an integer.
        """          
        if type(point) in [np.int,np.int32,np.int64]:
            point_ind=point
        elif type(point) == np.ndarray and len(point)==3:
            point_ind = np.argmin(np.linalg.norm(self.points - point,axis=1))
        elif type(point) in [tuple,list] and len(point)==3:
            point_ind = np.argmin(np.linalg.norm(self.points - np.array(point),axis=1))
        else:
            sys.exit("'point' must be an integer index, or a length 3 list, tuple, or numpy ndarray (x,y,z)")
        return point_ind

    def edge_points(self,u,k=7,return_mask=False,number=None):
        """Computes the edge points of the mesh.

        Args:
            u: A (num_verts,1) Numpy array of point labels.
            k: Optional integer number of nearest neighbors to use. Default is 7.
            return_mask: Optional boolean to return edge_points as a (num,verts,) boolean Numpy array. Default is False.
            number: Optional max number of edge points to return. Default is None, meaning all are returned.

        Returns:
            A Numpy array containing the edge point indices.
        """          
        if np.any(self.knn_I) is None or np.any(self.knn_J) is None or np.any(self.knn_D) is None:
            self.knn_I,self.knn_J,self.knn_D = gl.knnsearch(self.points,20)
        I = self.knn_I[:,:k]
        J = self.knn_J[:,:k]
        D = self.knn_D[:,:k]
        W = gl.weight_matrix(I,J,D,k,f=lambda x : np.ones_like(x),symmetrize=False)
        d = gl.degrees(W)
        mask = d*u != W@u

        #Select a few points spaced out along edge
        if number is not None:
            edge_ind = np.arange(self.num_verts())[mask]
            edge_points = self.points[mask,:]
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
            mask = np.zeros(self.num_verts(),dtype=bool)
            mask[spaced_edge_ind]=True

        if return_mask:
            return mask.astype(int)
        else: #return indices
            return np.arange(self.num_verts())[mask]

    def geodesic_patch(self,point,r,k=7,return_mask=False):
        """Computes a geodesic patch around a specified point.

        Args:
            point: A mesh vertex, as a coordinate or index.
            r: Radius used to build patch, as a float.
            k: Optional integer number of nearest neighbors to use. Default is 7.
            return_mask: Optional boolean to return the patch as a (num,verts,) boolean Numpy array. Default is False.

        Returns:
            A Numpy array containing the patch point indices.
        """       
        if np.any(self.knn_I) is None or np.any(self.knn_J) is None or np.any(self.knn_D) is None:
            self.knn_I,self.knn_J,self.knn_D = gl.knnsearch(self.points,20)
        I = self.knn_I[:,:k]
        J = self.knn_J[:,:k]
        D = self.knn_D[:,:k]
        W = gl.dist_matrix(I,J,D,k)

        point_ind = self.get_index(point)
        dist = gl.cDijkstra(W,np.array([point_ind]),np.array([0]))
        mask = dist < r

        if return_mask:
            return mask.astype(int)
        else: #return indices
            return np.arange(self.num_verts())[mask]

    #vertex-triangle adjacencey matrix
    #Returns num_verts x num_tri sparse matrix F with F_ij = 1 if vertex i belongs to triangle j
    #If normalize=True, then each row is divided by the number of adjacent triangles, 
    #so F can be used to interplate from triangles to vertices
    def tri_vert_adj(self,normalize=False):
        """Computes a sparse vertex-triangle adjacency matrix.

        Args:
            normalize: Optional boolean that divides the rows by the number of adjacent triangles if True. Default is False.
        
        Returns:
            A Numpy array of size (num_verts,num_tri) F with F_{ij} = 1 if vertex i belongs to triangle j.
        """            
        num_verts = self.num_verts()
        ind = np.arange(self.num_tri())

        if np.any(self.tri_vert_adj_I) is None or np.any(self.tri_vert_adj_J) is None:
            self.tri_vert_adj_I = np.hstack((self.triangles[:,0],self.triangles[:,1],self.triangles[:,2]))
            self.tri_vert_adj_J = np.hstack((ind,ind,ind))
        I = self.tri_vert_adj_I
        J = self.tri_vert_adj_J
        F = sparse.coo_matrix((np.ones(len(I)), (I,J)),shape=(self.num_verts(),self.num_tri())).tocsr()

        if normalize:
            num_adj_tri = F@np.ones(self.num_tri())
            F = sparse.spdiags(1/num_adj_tri,0,self.num_verts(),self.num_verts())@F

        return F

    #Returns unit normal vectors to vertices (averaging adjacent faces and normalizing)
    def vertex_normals(self):
        """Computes normal vectors to vertices.
        
        Returns:
            A Numpy array of size (num_verts,3) containing the vertex normal vectors.
        """          
        if self.unit_norms is None:
            self.face_normals()
        fn = self.unit_norms
        F = self.tri_vert_adj()
        vn = F@fn
        norms = np.linalg.norm(vn,axis=1)
        norms[norms==0] = 1

        return vn/norms[:,np.newaxis]
                  
    #Returns unit normal vectors
    def face_normals(self,normalize=True):
        """Computes normal vectors to triangles (faces).

        Args:
            normalize: Whether or not to normalize to unit vectors. If False, then the magnitude of each vector is twice the area of the corresponding triangle. Default is True.
        
        Returns:
            A Numpy array of size (num_tri,3) containing the face normal vectors.
        """      
        P1 = self.points[self.triangles[:,0],:]
        P2 = self.points[self.triangles[:,1],:]
        P3 = self.points[self.triangles[:,2],:]

        N = np.cross(P2-P1,P3-P1)
        if normalize:
            N = (N.T/np.linalg.norm(N,axis=1)).T
            self.unit_norms = N
            return N
        else:
          self.norms = N
          return N
          
    def flip_normals(self):
        """Reverses the orientation of all normal vectors in the mesh
        """
        self.Triangles = self.Triangles[:,::-1] 

    #Areas of all triangles in mesh
    def tri_areas(self):
        """Computes areas of all triangles in the mesh.
        
        Returns:
            A Numpy array of size (num_tri,) containing the areas of each triangle (face).
        """
        if self.norms is None:
            self.face_normals(False)
        return np.linalg.norm(self.norms,axis=1)/2

    #Surface area of mesh
    def surf_area(self):
        """Computes surface area of the mesh.
        
        Returns:
            The surface area of the entire mesh as an integer.
        """
        return np.sum(self.tri_areas())
       
    #Centers of each face
    def face_centers(self):
        """Computes coordinates of the center of each triangle (face).
        
        Returns:
            A Numpy array of size (num_tri,3) containing the coordinates of the face centers.
        """      
        P1 = self.points[self.triangles[:,0],:]
        P2 = self.points[self.triangles[:,1],:]
        P3 = self.points[self.triangles[:,2],:]

        result = (P1 + P2 + P3)/3
        self.centers = result
        return result 
       
    #Volume enclosed by mesh
    def volume(self):
        """Computes the volume of the mesh.
        
        Returns:
            The volume of the mesh as an integer.
        """      
        if self.centers is None:
            self.face_centers()
        X = self.centers
        X = X - np.mean(X,axis=0)
        if self.norms is None:
            self.face_normals(False)
        return np.sum(X*self.norms)/6
   
    def bbox(self):
        """Computes the bounding box of the mesh.
        
        Returns:
            A Numpy array of size (3,) containing the dimensions of the bounding box.
        """      
        if self.centers is None:
            self.face_centers()
        X = self.centers
        n = X.shape[0]
        A = self.tri_areas()

        W = sparse.spdiags(A**2,0,n,n)
        vals,vecs = weighted_pca(X,W)

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
    def plotsurf(self,C=None):
        """Plots the mesh as a surface using mayavi.

        Args:
            C: An optional per-vertex labeling scheme to use with shape (num_vert,3). Default is None.
        
        Returns:
            A visualization of the mesh.
        """      
        if C is None:
            mlab.triangular_mesh(self.points[:,0],self.points[:,1],self.points[:,2],self.triangles)
        else:
            mlab.triangular_mesh(self.points[:,0],self.points[:,1],self.points[:,2],self.triangles,scalars=C)

    def cplotsurf(self,C=-1):
        """Plots the mesh as a surface using mayavi.

        Args:
            C: An optional per-vertex labeling scheme to use with shape (num_vert,3). Default is -1.
        
        Returns:
            A colored visualization of the mesh.
        """      
        if C.any == -1: #if no C given
            C = np.ones((len(x),1))
            
        n = len(np.unique(C))
        C = C.astype(int)
        if n>20:
            mesh = mlab.triangular_mesh(self.points[:,0],self.points[:,1],self.points[:,2],self.triangles,scalars=C)
        else:
            col = (np.arange(1,n+1)) / n
            colors = col[C-1]
            mesh = mlab.triangular_mesh(self.points[:,0],self.points[:,1],self.points[:,2],self.triangles,scalars=colors)
            
        return mesh
        
    #Write a ply file
    def to_ply(self,fname):
        """Writes the mesh to a .ply file.

        Args:
            fname: The name of the .ply file to write the mesh to.
        """
        f = open(fname,"w")

        #Write header
        f.write('ply\n')
        f.write('format binary_little_endian 1.0\n')
        f.write('element vertex %u\n'%self.num_verts())
        f.write('property double x\n')
        f.write('property double y\n')
        f.write('property double z\n')
        f.write('element face %u\n'%self.num_tri())
        f.write('property list int int vertex_indices\n')
        f.write('end_header\n')
        f.close()

        f = open(fname,"ab")

        #write vertices
        f.write(self.points.astype('float64').tobytes())

        #write faces
        T = np.hstack((np.ones((self.num_tri(),1))*3,self.triangles)).astype(int)
        f.write(T.astype('int32').tobytes())

        #close file
        f.close()
       
    #Write a ply file
    def write_color_ply(self,color,fname):
        """Writes the colored mesh to a .ply file.

        Args:
            color: An array of length num_verts of color data.
            fname: The name of the .ply file to write the colored mesh to.
        """
        f = open(fname,"w")

        #Write header
        f.write('ply\n')
        f.write('format binary_little_endian 1.0\n')
        f.write('element vertex %u\n'%self.num_verts())
        f.write('property double x\n')
        f.write('property double y\n')
        f.write('property double z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('element face %u\n'%self.num_tri())
        f.write('property list int int vertex_indices\n')
        f.write('end_header\n')
        f.close()

        f = open(fname,"ab")

        #write vertices
        for i in range(self.num_verts()):
            f.write(P[i,:].astype('float64').tobytes())
            f.write(color[i,:].astype('uint8').tobytes())

        #write faces
        T = np.hstack((np.ones((self.num_tri(),1))*3,T)).astype(int)
        f.write(T.astype('int32').tobytes())

        #close file
        f.close()

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
                V1,V2,V3: principal directions
        """

        return svi.svipca(self.points,self.triangles,r)

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
    def virtual_goniometer(self,point,r,k=7,SegParam=2,return_edge_points=False,number_edge_points=None):
        """Runs a virtual goniometer to measure break angles.

        Args:
            point: A mesh vertex, as a coordinate or index.
            r: Radius used to build patch, as a float.
            k: Optional integer number of nearest neighbors to use. Default is 7.
            SegParam: Optional segmentation parameter that encourages splitting patch in half as it increases in size. Default is 2.
            return_edge_points: Optional boolean to return edge points in patch. Default is False.
            number_edge_points: Optional boolean to specify how many edge points to return. Default is None.
        
        Returns:
            theta: The break angle.
            n1: A (3,) Numpy array containing the normal vector of one break surface.
            n2: A (3,) Numpy array containing the normal vector of the other surface.
            C: A (num_verts,) Numpy array containing the cluster (1 or 2) of each point in the patch. Points not in the patch are assigned a 0.
            E: Optional (number_edge_points,1) Numpy array of edge point indices. Is not returned by default.

        """
        patch_ind = self.geodesic_patch(point,r,k=k)
        patch = self.points[patch_ind,:]
        normals = self.vertex_normals()[patch_ind,:]
        theta,n1,n2,C_local = __virtual_goniometer__(patch,normals,SegParam=SegParam)

        C = np.zeros(self.num_verts())
        C[patch_ind] = C_local


        if return_edge_points:
            E = self.edge_points(C_local,k=k,number=number_edge_points)
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
def __virtual_goniometer__(P,N,SegParam=2,UsePCA=True,UsePower=False):
    """Internal function used within class method virtual_goniometer to measure break angles.

    Args:
        P: A (n,3) Numpy array of vertices in a patch.
        N: A (n,3) Numpy array of vertex normal vectors.
        SegParam: Optional segmentation parameter that encourages splitting patch in half as it increases in size. Default is 2.
        UsePCA: Optional boolean that uses PCA instead of averaged surface normals if True. Default is True.
        UsePower: Optional boolean that uses the power method when doing PCA if True. Default is False.
    
    Returns:
        theta: The break angle.
        n1: A (3,) Numpy array containing the normal vector of one break surface.
        n2: A (3,) Numpy array containing the normal vector of the other surface.
        C: A (num_verts,) Numpy array containing the cluster (1 or 2) of each point in the patch. Points not in the patch are assigned a 0.
    """
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
    theta = 180-np.arccos(np.dot(n1,n2))*180/np.pi
    return theta,n1,n2,C