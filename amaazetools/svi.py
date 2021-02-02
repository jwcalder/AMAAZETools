#svi.py
#Spherical Volume Invariant
import numpy as np
from numpy import matlib
import amaazetools.cextensions as cext
from . import trimesh as tm


def svi(P,T,r,ID=None):
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
    n = P.shape[0]  #Number of vertices
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
    T = np.ascontiguousarray(T,dtype=np.int32)
    P = np.ascontiguousarray(P,dtype=np.float64)

    #Run SVI code
    for i in np.arange(0,rlen): 
        cext.svi(P,T,ID,r[i],eps,prog,S,G)
        Sout[:,i] = S
        Gout[:,i] = G

    return Sout,Gout

def svipca(P,T,r):
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

    n = P.shape[0]  #Number of vertices
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
        
    VN = tm.vertex_normals(P,T)
    Z3 = np.zeros((3), dtype=np.float64)
        
    #indexing for output:
    I = np.arange(0,n)
    I = I[I]
        
    #Contiguous arrays
    T = np.ascontiguousarray(T,dtype=np.int32)
    P = np.ascontiguousarray(P,dtype=np.float64)

    for k in np.arange(0,rlen):
        cext.svipca(P,T,ID,r[k],eps_svi,eps_pca,prog,S,M)
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

