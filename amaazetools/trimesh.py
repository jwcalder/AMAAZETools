#tri_mesh.py
#Class for working with triangulated meshes

#test

import graphlearning as gl
import numpy as np
from numpy import matlib
from plyfile import PlyData, PlyElement
import scipy.sparse as sparse
import scipy.spatial as spatial
from sklearn.neighbors import NearestNeighbors
from . import svi
from . import edge_detection
import sys
import urllib.request as url


#Enable plotting if possible
try:
    from mayavi import mlab
    from pyface.api import GUI
    import moviepy.editor as mpy
except:
    print("Could not find mayavi, plotting functionality will be disabled.")

#Non-Class Specific Functions

def withiness(x):
    """ Computes withiness (how well 1-D data clusters into two groups).

        Parameters
        ----------
        x : (n,1) float array
            A 1-D collection of data.
        
        Returns
        -------
        w : float
            The withiness of the data.
        m : float
            The point at which to split the data into 2 clusters.
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
    """ Computes principal component analysis (PCA) on a point cloud P.

        Parameters
        ----------
        P : (n,d) float array
            A point cloud.
        
        Returns
        -------
        vals : (d,) float arrayy
            The variances among each principal component.
        vecs : (d,d) float array
            The principal component vectors.
    """
        
    P = P - np.mean(P,axis=0)
    vals,vecs = np.linalg.eig(P.T@P)
    idx = np.argsort(-vals)

    return vals[idx],vecs[:,idx]
 
def weighted_pca(P,W):
    """ Computes weighted principal component analysis (PCA) on a point cloud P.

        Parameters
        ----------
        P : (n,d) float array
            A point cloud.
        W : (n,) float array
            An array containing the weights of the points.
        
        Returns
        -------
        vals : (d,) float array
            The variances among each principal component.
        vecs : (d,d) float array
            The principal component vectors.
        sign : boolean
            True when the first principal component direction is positively oriented.
    """

    P = P - np.sum(P*W[:,None],axis=0)/np.sum(W)
    vals,vecs = np.linalg.eig(P.T@(P*W[:,None]))

    idx = np.argsort(-vals)
    return vals[idx], vecs[:,idx]

#Power method to find principle eigenvector
def power_method(A,tol=1e-12):
    """ Computes the smallest (in absolute value) eigenvalue and its corresponding eigenvector using the power method.

        Parameters
        ----------
        A : (n,n) float array
            A square matrix that one wishes to find the smallest (in absolute value) eigenvalue and corresponding eigenvector of.
        tol : float, default is 1e-12
            The desired tolerance threshold after which to stop iteration.
        
        Parameters
        ----------
        l : float
            The smallest (in absolute value) eigenvalue of A.
        x : (n,1) float array
            Array containing the eigenvector corresponding to the smallest (in absolute value) eigenvalue of A.
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
    """ Computes the last principal component of a point cloud X using the power method.

        Parameters
        ----------
        X : (n,3) float array
            A point cloud.
        center : boolean, default is True
            Data is centered if True.
        
        Returns
        -------
        A float array of size (3,) containing the last principal component vector.
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
    """ Computes the last principal component of a point cloud X.

        Parameters
        ----------
        X : (n,3) float array
            A point cloud.
        center : boolean, default is True
            Data is centered if True.
        
        Returns
        -------
        A float array of size (3,) containing the last principal component vector.
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
    """ Reads the vertex and triangle data stored in a .ply file.

        Parameters
        ----------
        fname: str
            Name of the file to read from.
        
        Returns
        -------
        P : (num_verts,3) float array
            The coordinates of the vertices of the mesh.
        T : (num_tri,3) int array
            The indices of the triangles of the mesh.
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
    """ Loads a file path or url and creates a mesh object.

        Parameters
        ----------
        path : str
            URL or file path at which to access .ply file.
    
        Returns
        -------
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
        self.poisson_W_matrix = None
        self.poisson_J_matrix = None
        self.poisson_node_idx = None
        self.poisson_labels = None

    #Get number of vertices
    def num_verts(self):
        """ Computes number of vertices in the mesh.

            Returns
            -------
            The number of vertices in the mesh as an integer.
        """

        return self.points.shape[0]

    #Get number of triangles
    def num_tri(self):
        """ Computes number of triangles in the mesh.

            Returns
            -------
            The number of triangles in the mesh as an integer.
        """

        return self.triangles.shape[0]

    #Converts from (x,y,z) to index of closest point
    def get_index(self,point):
        """ Computes the index of a given point.

            Parameters
            ----------
            point : int or (1,3) float array
                A vertex in the mesh, specified by either an integer index or its coordinates.

            Returns
            -------
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
        """ Computes the edge points of the mesh.

            Parameters
            ----------
            u : (num_verts,1) int array
                Array of labels for each point.
            k : int, default is 7
                Number of nearest neighbors to use.
            return_mask : boolean, default is False
                If True, return edge_points as a (num,verts,) boolean array.
            number : int, default is None
                Max number of edge points to return.

            Returns
            -------
            An int array containing the edge point indices.
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
        """ Computes a geodesic patch around a specified point.

            Parameters
            ----------
            point : int or (1,3) float array
                A mesh vertex.
            r : float
                Radius used to build patch.
            k : int, default is 7
                Number of nearest neighbors to use.
            return_mask : boolean, default is False
                If True, return the patch as a (num,verts,) boolean array

            Returns
            -------
            An int array containing the patch point indices.
        """

        if np.any(self.knn_I) is None or np.any(self.knn_J) is None or np.any(self.knn_D) is None:
            self.knn_I,self.knn_J,self.knn_D = gl.knnsearch(self.points,20)
        I = self.knn_I[:,:k]
        J = self.knn_J[:,:k]
        D = self.knn_D[:,:k]
        W = gl.dist_matrix(I,J,D,k)
        W = gl.sparse_max(W,W.transpose())

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
        """ Computes a sparse vertex-triangle adjacency matrix.
       
            Parameters
            ----------
            normalize : boolean, default is False
                If True, each row is divided by the number of adjacent triangles.

            Returns
            -------
            F : (num_verts,num_tri) boolean array
                Adjacency matrix; F[i,j] = 1 if vertex i belongs to triangle j.
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
        """ Computes normal vectors to vertices.
        
            Returns
            -------
            A (num_verts,3) float array containing the vertex normal vectors.
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
        """ Computes normal vectors to triangles (faces).
        
            Parameters
            ----------
            normalize: boolean, default is True
                Whether or not to normalize to unit vectors; if False, vector magnitude is twice the area of the corresponding triangle.
        
            Returns
            -------
            N : (num_tri,3) float array
                Array containing the face normal vectors.
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
        """ Reverses the orientation of all normal vectors in the mesh
        """

        self.triangles = self.triangles[:,::-1]

    #Areas of all triangles in mesh
    def tri_areas(self):
        """ Computes areas of all triangles in the mesh.
        
            Returns
            -------
            A (num_tri,) float array containing the areas of each triangle (face).
        """

        if self.norms is None:
            self.face_normals(False)
        return np.linalg.norm(self.norms,axis=1)/2

    #Surface area of mesh
    def surf_area(self):
        """ Computes surface area of the mesh.
        
            Returns
            -------
            The surface area of the entire mesh as a float.
        """

        return np.sum(self.tri_areas())
       
    #Centers of each face
    def face_centers(self):
        """ Computes coordinates of the center of each triangle (face).
        
            Returns
            -------
            A (num_tri,3) float array containing the coordinates of the face centers.
        """

        P1 = self.points[self.triangles[:,0],:]
        P2 = self.points[self.triangles[:,1],:]
        P3 = self.points[self.triangles[:,2],:]

        result = (P1 + P2 + P3)/3
        self.centers = result
        return result
       
    #Volume enclosed by mesh
    def volume(self):
        """ Computes the volume of the mesh.
        
            Returns
            -------
            The volume of the mesh as a float.
        """

        if self.centers is None:
            self.face_centers()
        X = self.centers
        X = X - np.mean(X,axis=0)
        if self.norms is None:
            self.face_normals(False)
        return np.sum(X*self.norms)/6
   
    def bbox(self):
        """ Computes the bounding box of the mesh.
        
            Returns
            -------
            A (3,) float array containing the dimensions of the bounding box.
        """

        #This code is old, when the weighted version was used
        #if self.centers is None:
        #    self.face_centers()
        #X = self.centers.copy()
        #X -= np.mean(X,axis=0)
        #A = self.tri_areas()
        #vals,vecs = weighted_pca(X,A**2)

        X = self.points.copy()
        X -= np.mean(X,axis=0)
        vals,vecs = pca(X)

        Y = X@vecs
        bb = np.max(Y,axis=0) - np.min(Y,axis=0)
        
        return bb
        
     
    #Plot triangulated surface
    def plotsurf(self,C=None):
        """ Plots the mesh as a surface using mayavi.

            Parameters
            ----------
            C : (num_verts,3) int array, default is None
                An optional per-vertex labeling scheme to use.
        
            Returns
            -------
            A visualization of the mesh.
        """

        if C is None:
            mlab.triangular_mesh(self.points[:,0],self.points[:,1],self.points[:,2],self.triangles)
        else:
            mlab.triangular_mesh(self.points[:,0],self.points[:,1],self.points[:,2],self.triangles,scalars=C)

    def cplotsurf(self,C=-1):
        """ Plots the mesh as a surface using mayavi.

            Parameters
            ----------
            C : (num_verts,3) int array, default is -1
                An optional per-vertex labeling scheme to use.
        
            Returns
            -------
            mesh : amaazetools.trimesh.mesh object
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
        """ Writes the mesh to a .ply file.

            Parameters
            ----------
            fname : str
                The name of the .ply file to write the mesh to.
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
        """ Writes the colored mesh to a .ply file.

            Parameters
            ----------
            color : (num,verts,3) float array
                An array of color data for each point.
            fname : str
                The name of the .ply file to write the colored mesh to.
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

    def to_gif(self,fname,color = [],duration=7,fps=20,size=750,histeq = True):
        """ Writes rotating gif

            Parameters
            ----------
            fname : str
                gif filename
            color : (1,3) or (num_verts,1) or (num_verts,2) float array, default is (.7,.7,.7)
                3-tuple 0 to 1 RGB for single color over surface OR array the length of Self.Points for interpolation (1D or 2D - if 2D, uses first column).
            duration : float, default is 7
                length of gif in seconds
            fps: float, default is 20
                frames per second
            size: float, default is 750
                size of gif images
            histeq : boolean, default is True
                Performs histogram equalization on scalar color array; else should normalize prior to input.
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

    def svi(self,r,ID=None):
        """ Computes spherical volume invariant.
        
            Parameters
            ----------
            r : (k,1) float array
                List of radii to use.
            ID : (n,1) boolean array, default is None
                Spherical volume is only computed at points with True indices. 
        
            Returns
            -------
            S : (n,1) float array
                The volumes corresponding to each point.
            G : (n,1) float array
                The  gamma values corresponding to each point.
        """
   
        return svi.svi(self.points,self.triangles,r,ID=ID)

    def svipca(self,r):
        """ Computes SVIPCA

            Parameters
            ----------
            r : (k,1) float array
                List of radii to use.

            Returns
            -------
            S : (n,1) float array
                The volumes corresponding to each point.
            K1 : (n,1) float array
                The first principle curvature for each point.
            K2 : (n,1) float array
                The second principle curvature for each point.
            V1 : (n,3) float array
                The first principal direction for each point. 
            V2 : (n,3) float array
                The second principal direction for each point.
            V3 : (n,3) float array
                The third principal direction for each point.
        """

        return svi.svipca(self.points,self.triangles,r)

    def edge_graph_detect(self,**kwargs):
        """ Detects edges using SVIPCA and principal direction metric.
            
            Parameters
            ----------
            M : amaazetools.trimesh.mesh object
            k1 : float, optional
                A constant on the minimum of the inverse of principal curvatures.
            k2 : float, optional
                A constant on the mean volume.
            VOL : (n,1) float array, optional
                Spherical volume corresponding to each point in the mesh.
            K1 : (n,1) float array, optional
                First principal curvature of each point.
            K2 : (n,1) float array, optional
                Second principal curvature of each point.
            V1 : (n,3) float array, optional
                First principal direction for each point.
            V2 : (n,3) float array, optional
                Second principal direction for each point.
            rvol : float, optional
                Radius to use for svipca.
            rpdir : float, optional
                Radius to use for the principal direction metric.

            Returns
            -------
            Edges : (n,1) boolean array
                A true value corresponds to that index being an edge point.
        """
        
        return edge_detection.edge_graph_detect(self,**kwargs)

    def graph_setup(self,n,r,p,seed=None):
        """ Creates the graph to use for poisson learning.

            Parameters
            ----------
            n : int
                The number of vertices to sample for the graph.
            r : float
                Radius for graph construction.
            p : float
                Weight matrix parameter.
            seed : int, default is None
                Optional seed for random number generator.
        
            Returns
            -------
            poisson_W_matrix : (n,n) scipy.sparse.lil_matrix
                Weight matrix describing similarities of normal vectors.
            poisson_J_matrix : (num_verts,n) scipy.sparse.lil_matrix
                Matrix with indices of nearest neighbors.
            poisson_node_idx : (num_verts,1) int array
                The indices of the closest point in the sample.
        """

        rng = (
            np.random.default_rng(seed=seed)
            if seed is not None
            else np.random.default_rng()
        )

        if self.poisson_W_matrix is None or self.poisson_J_matrix is None or self.poisson_node_idx is None:

            v = self.vertex_normals()
            N = self.num_verts()
        
            #Random subsample
            ss_idx = np.matrix(rng.choice(self.points.shape[0],n,replace=False))
            y = np.squeeze(self.points[ss_idx,:])
            w = np.squeeze(v[ss_idx,:])

            xTree = spatial.cKDTree(self.points)
            nn_idx = xTree.query_ball_point(y, r)
            yTree = spatial.cKDTree(y)
            nodes_idx = yTree.query_ball_point(y, r)
        
            bn = np.zeros((n,3))
            J = sparse.lil_matrix((N,n))
            for i in range(n):
                vj = v[nn_idx[i],:]
                normal_diff = w[i] - vj
                weights = np.exp(-8 * np.sum(np.square(normal_diff),1,keepdims=True))
                bn[i] = np.sum(weights*vj,0) / np.sum(weights,0)
            
                #Set ith row of J
                normal_diff = bn[i]- vj
                weights = np.exp(-8 * np.sum(np.square(normal_diff),1))#,keepdims=True))
                J[nn_idx[i],i] = weights
            
            #Normalize rows of J
            RSM = sparse.spdiags((1 / np.sum(J,1)).ravel(),0,N,N)
            J = RSM @ J
        
            #Compute weight matrix W
            W = sparse.lil_matrix((n,n))
            for i in range(n):
                nj = bn[nodes_idx[i]]
                normal_diff = bn[i] - nj
                weights = np.exp(-32 * ((np.sqrt(np.sum(np.square(normal_diff),1)))/2)**p)
                W[i,nodes_idx[i]] = weights
        
            #Find nearest node to each vertex
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(y)
            instances, node_idx = nbrs.kneighbors(self.points)

            self.poisson_W_matrix = W
            self.poisson_J_matrix = J
            self.poisson_node_idx = node_idx
        
        return self.poisson_W_matrix, self.poisson_J_matrix, self.poisson_node_idx   

    def poisson_label(self,g,I,n=5000,r=0.5,p=1,s=None,graph_setup=False):
        """ Performs poisson learning on the mesh.

            Parameters
            ----------
            g : (k,1) int array
                Labels to assign to vertices.
            I : (k,1) int array
                User-selected vertices.
            n : int, default is 5000
                The number of nodes to sample.
            r : float, default is 0.5
                The radius for nearest neighbor search.
            p : float, default is 1.0
                The weight matrix parameter.
            s : default is None
                Weights for fine-tuning Poisson learning.
            graph_setup : boolean, default is False
                Force graph construction if True.
        
            Returns
            -------
            L : (num_verts,1) int array
                Poisson labelling of each point in mesh.
        """
    
        if graph_setup or (self.poisson_node_idx is None):
            self.graph_setup(n,r,p)

        I = self.poisson_node_idx[I]
        W = self.poisson_W_matrix
        u = poisson_learning(W,g,I)
        J = self.poisson_J_matrix

        if s is None:
            L = np.argmax(J@u,1)
        else:
            k = np.max(g)+1  #Number of classes, assuming 0,1,2,3,..,k-1 are  used
            L = np.argmax(J@(u*s[:k]),1)
        L = canonical_labels(L)

        self.poisson_labels = L

        return L
    
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
        """ Runs a virtual goniometer to measure break angles.

            Parameters
            ----------

            point : (1,3) float array or int
                A mesh vertex, as a coordinate or index.
            r : float
                Radius used to build patch.
            k: int, default is 7
                Number of nearest neighbors to use.
            SegParam : float, default is 2
                Segmentation parameter that encourages splitting patch in half as it increases in size.
            return_edge_points : boolean, default is False
                If True, return edge points in patch.
            number_edge_points : boolean, default is None
                Specifies how many edge points to return.
        
            Returns
            -------
            theta : float
                The break angle.
            n1 : (3,) float array
                Contains the normal vector of one break surface.
            n2 : (3,) float array
                Contains the normal vector of the other surface.
            C : (num_verts,) int array
                Contains the cluster (1 or 2) of each point in the patch; points not in the patch are assigned a 0.
            E : (number_edge_points,1) int array, not returned by default
                List of  edge point indices.
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
    """ Internal function used within class method virtual_goniometer to measure break angles.

        Parameters
        ----------
        P : (n,3) float array
            Vertices in a patch.
        N : (n,3) float array
            Vertex normal vectors.
        SegParam : float, default is 2
            Segmentation parameter that encourages splitting patch in half as it increases in size.
        UsePCA: boolean, default is True
            Uses PCA instead of averaged surface normals if True. 
        UsePower : boolean, default is False
            Uses the power method when doing PCA if True.
    
        Returns
        -------
        theta : float
            The break angle.
        n1 : (3,) float array
            Contains the normal vector of one break surface.
        n2 : (3,) float array
            Contains the normal vector of the other surface.
        C : (num_verts,) int array
            Contains the cluster (1 or 2) of each point in the patch; points not in the patch are assigned a 0.
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
    
def conjgrad(A,b,x,T,tol):
    """ Performs conjugate gradient descent.

        Parameters
        ----------
        A : matrix multiplying x
        b : vector equal to product of A and x
        x : initial estimate for x 
        T : int
            Number of time steps allowed.
        Tol : float
            Desired convergence tolerance of result.
        
        Returns
        -------
        x : calculated value for x
        i : int
            Number of iterations required for convergence.
    """
        
    r = b - A@x
    p = r
    rsold = np.sum(r * r,0)
    for i in range(int(T)):
        Ap = A@p
        alpha = rsold / np.sum(p*Ap,0)
        x = x + alpha*p
        r = r - alpha*Ap
        rsnew = np.sum(r*r,0)
        if np.sqrt(np.sum(rsnew)) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x,i

def poisson_learning(W,g,I):
    """ Performs poisson learning.

        Parameters
        ----------
        W : (n,n) float array
            Weight matrix of subsampled graph of mesh.
        g : (m,1) int array
            Labels to assign to selected vertices.
        I : (m,1) int array
            Indices of user-selected vertices.
        
        Returns
        -------
        u : (num_verts,1) int array
            Poisson labels for each vertex in the mesh.
    """
        
    k = len(np.unique(g))
    n = W.shape[0]
    m = len(I)
    I = I - 1
    g = g.T - 1

    F = np.zeros((n,k))
    for i in range(m):
        F[I[i],g[i]] = 1
    c = np.ones((1,n)) @ F / len(g)
    F[I] -= c
    
    deg = np.sum(W,1)
    D = sparse.spdiags(deg.T,0,n,n)
    L = D-W #Unnormalized graph laplacian matrix
    
    #Preconditioning
    Dinv2 = sparse.spdiags(np.power(np.sum(W,1),-1/2).T,0,n,n) 
    Lnorm = Dinv2 @ L @ Dinv2
    F = Dinv2 @ F
    
    #Conjugate Gradient Solver
    u,i = conjgrad(Lnorm,F,np.zeros((n,k)),1e5, np.sqrt(n)*1e-10)
    
    #Undo preconditioning
    u = Dinv2 @ u
    return u

def canonical_labels(u):
    """ Reorders a label vector into canonical order.

        Parameters
        ----------
        u : (num_verts,1) int array
            A label vector.
        
        Returns
        -------
        u : (num_verts,1) int array
            A reodered label vector.
    """
        
    n = len(u)
    k = len(np.unique(u))
    label_set = np.zeros((k,1))
    label = 0
    
    for i in range(n):
        if u[i] > label:
            label += 1
            l = u[i]
            I = u == label
            J = u == l
            u[I] = l
            u[J] = label
    return u

