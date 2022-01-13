#Utilities for processing dicom volumes
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom
import scipy.ndimage as ndimage
import scipy.stats as stats
from skimage import measure
from skimage.transform import rescale
import os, multiprocessing
import sys
from joblib import Parallel, delayed
from . import trimesh as tm

#Withiness is a measure of how well 1D data clusters into two groups
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


def read_dicom_list(files):
    """ Reads dicom images from a provided list files, stores in a volume I, and returns resolution dx and dz and a list of files; some may be ommitted if they are not dicom.

        Parameters
        ----------
        files : str list
            List of files to read from.
        
        Returns
        -------
        I : Volume of images.
        dx : Image resolution dimension.
        dz : Image resolution dimension.
        dicom_files : List of dicom files.
    """

    num_slices = len(files)
    dicom_files = []
    i = 0
    for f in files:
        try:
            A = dicom.dcmread(f)
            dicom_files.append(f)
            if i == 0:
                s = A.pixel_array.shape
                I = np.zeros((num_slices, s[0], s[1]), dtype=int)
                dx = np.zeros((num_slices, 2))
                dz = np.zeros(num_slices)
            I[i, :, :] = A.pixel_array
            dx[i, :] = A.PixelSpacing
            dz[i] = A.SliceThickness
            i += 1
        except:
            print(f + ' is not a DICOM file.')

    if i == 0:
        print('No DICOM files found.')
        return (None, None, None, None)
    else:
        print('Found %d DICOM files.' % i)
        return (I[:i, :, :], dx[:i, :], dz[:i], dicom_files)


def find_dicom_subdir(directory):
    """ Finds subdirectory with the most dicom files

        Parameters
        ----------
        directory : str
            Directory to search within.

        Returns
        -------
        dicom_dir : str
            Subdirectory with the most dicom files.
    """

    dicom_dir = None
    num_dicom_files = 0
    for root, subdirs, files in os.walk(directory):
        num = len(files)
        if num > num_dicom_files:
            num_dicom_files = num
            dicom_dir = root
    return dicom_dir


def read_dicom_dir(directory):
    """ Finds and reads dicom volume from directory; finds the subdirectory with the most dicom files and loads those into a volume.

        Parameters
        ----------
        directory : str
            Directory to read from.

        Returns
        -------
        I : Volume of images.
        dx : Image resolution dimension.
        dz : Image resolution dimension.
        dicom_files : List of dicom files.
    """

    dicom_dir = find_dicom_subdir(directory)
    files = os.listdir(dicom_dir)
    files.sort()
    files = [os.path.join(dicom_dir, f) for f in files]
    return read_dicom_list(files)

def add_border(I):
    """ Adds border to a given image.
        
        Parameters
        ----------
        I : float array
            Image to add border to.

        Returns
        -------
        I : float array
            Image with border added.
    """

    I[0,:]=1
    I[-1,:]=1
    I[:,0]=1
    I[:,-1]=1

    return I

def bone_overview(I,mask=False):
    """ Returns an overview of the scan from top and side.

        Parameters
        ----------
        I : float array
            Image of scan.
        mask : boolean, default is False
            If True, only work with positive values of I.
        
        Returns
        -------
        J : float array
            Overview of scan from top and side.
    """
    
    m0 = I.shape[0]
    m1 = I.shape[1]
    m2 = I.shape[2]

    I0 = np.sum(I, axis=0)
    I1 = np.sum(I, axis=1)
    I2 = np.sum(I, axis=2)

    if mask:
        I0 = I0>0
        I1 = I1>0
        I2 = I2>0
    else:
        I0 = I0 / np.max(I0)
        I1 = I1 / np.max(I1)
        I2 = I2 / np.max(I2)

    I0 = add_border(I0)
    I1 = add_border(I1)
    I2 = add_border(I2)

    J = np.zeros((max(m0,m1),2*m2+m1))
    J[:m1,:m2] = I0
    J[:m0,m2:2*m2] = I1
    J[:m0,2*m2:] = I2

    return J


def scan_overview(I,true_mean=False):
    """ Returns an overview of the scan from top and side.

        Parameters
        ----------
        I : float array
            Image of scan.
        true_mean : boolean, default is False
            If True, take mean instead.

        Returns
        -------
        An overview of scan from top and side.
    """

    if true_mean:
        I1 = np.mean(I,axis=1)
        I2 = np.mean(I,axis=2)
    else:
        I1 = np.sum(I, axis=1)
        I2 = np.sum(I, axis=2)
        I1 = I1 / np.max(I1)
        I2 = I2 / np.max(I2)

    return np.hstack((I1, I2)).T

def trim(I, v, padding=20, erosion_width=5):
    """ Trim to v level with padding.

        Parameters
        ----------
        I : float array
            Image of scan.
        v : float
            Level to be trimmed to.
        padding : float, default is 20
        erosion_width : float, default is 5
    
        Returns
        -------
        I : float array
            Trimmed image.
    """

    J = I > v

    #Add extra to account for erosion
    #padding = padding + 4*erosion_width

    #kernel = np.ones((3,3), dtype=int)
    #erosion = cv2.erode(J.astype(float),kernel,iterations=erosion_width)
    #J = cv2.dilate(erosion,kernel,iterations=erosion_width)

    K = np.sum(np.sum(J,axis=2),axis=1) > 0
    ind = np.arange(len(K))[K]
    z1 = max(ind.min()-padding,0)
    z2 = min(ind.max()+padding,I.shape[0])

    K = np.sum(np.sum(J,axis=0),axis=1) > 0
    ind = np.arange(len(K))[K]
    x1 = max(ind.min()-padding,0)
    x2 = min(ind.max()+padding,I.shape[1])

    K = np.sum(np.sum(J,axis=0),axis=0) > 0
    ind = np.arange(len(K))[K]
    y1 = max(ind.min()-padding,0)
    y2 = min(ind.max()+padding,I.shape[2])

    return I[z1:z2,x1:x2,y1:y2]


def chop_up_scan(I, num_bones=5):
    """ Chop up a scan into num_bones fragments.
        
        Parameters
        ----------
        I : float array
            Image of scan.
        num_bones : int, default is 5
            Number of fragments to be chopped into.

        Returns
        -------
        The image of the bone split into fragments.
    """

    #Project to 1D
    J = np.sum(I, axis=2)
    J = np.sum(J, axis=1)
    J = J / J.max()

    #Gaussian filtering
    sigma = len(J) / (num_bones * 8)
    K = ndimage.gaussian_filter1d(J, sigma)

    #Compute locations of maxima and minima
    Km = K[1:-1]
    minima = np.arange(len(J) - 2)[((Km < K[2:]) & (Km < K[:-2]))]
    maxima = np.arange(len(J) - 2)[((Km > K[2:]) & (Km > K[:-2]))]
    m1 = maxima.min()
    m2 = maxima.max()

    #Get chop locations
    chop_loc = minima[((minima > m1) & (minima < m2))]

    #Some sanity checking
    if len(chop_loc) >= num_bones:
        vals = Km[chop_loc]
        sort = np.sort(vals)
        cutoff = sort[(num_bones - 1)]
        chop_loc = chop_loc[(vals < cutoff)]

    if len(chop_loc) < num_bones - 1:
        print('Warning: Did not find enough bones when chopping up!')

    #Adjust the final positions with a finer Gaussian fileter
    sigma = len(J) / (num_bones * 16)
    K = ndimage.gaussian_filter1d(J, sigma)
    Km = K[1:-1]
    minima = np.arange(len(J) - 2)[((Km < K[2:]) & (Km < K[:-2]))]
    for i in range(len(chop_loc)):
        chop_loc[i] = int((chop_loc[i] + minima[np.argmin(np.absolute(minima - chop_loc[i]))]) / 2)
    
    return chop_loc + 1


def imshow(J):
    """ Make showing grayscale images easier.

        Parameters
        ----------
        J : float array
            Image to show.

        Returns
        -------
        None
    """

    plt.figure()
    plt.imshow(J, cmap='gray')

def surface_bones(directory, iso=2500, write_gif=False):
    """ Processes all npz files in directory creating surface and saving to a ply file.

        Parameters
        ----------
        directory : str
            Directory to work within.
        iso : float (optional), default is 2500
            Iso level to be used for surfacing.
        write_gif : bool (optional), default=False
            Whether to output rotating gifs for each object. Requires mayavi, which can be hard to install.

        Returns
        -------
        None
    """

    for filename in os.listdir(directory):
        if filename.endswith(".npz"):
            print('Loading '+filename+'...')
            M = np.load(os.path.join(directory,filename))
            I = M['I']; dx = M['dx']; dz = M['dz']

            #Rescale image to account for different dx/dz dimensions
            J = rescale(I.astype(float),(dz/dx,1,1),mode='constant')

            #Marching cubes for isosurface
            iso_level = iso
            verts,faces,normals,values = measure.marching_cubes(J,iso_level)
            mesh = tm.mesh(dx*verts,faces) #Multiplication by dx fixes units

            #Reverse orientation of triangles (marching_cubes returns inward normals)
            mesh.flip_normals()

            #Write to ply file
            mesh_filename = os.path.join(directory,filename[:-4]+'_iso%d'%iso_level)
            print('Saving mesh to '+mesh_filename+'...')
            mesh.to_ply(mesh_filename+'.ply')

            if write_gif:
                mesh.to_gif(mesh_filename+'.gif')


def process_dicom(directory, scanlayout, CTdir='ScanOverviews', Meshdir='Meshes', 
                             chopsheet=None, threshold=2000, padding=15):
    """ Processes all dicom scans from scanlayout in a given directory, producing
        CT volumes for each object. 

        Parameters
        ----------
        directory : str
            Directory to be working within.
        scanlayout : pandas DataFrame
            1st column is CT scan data, 2nd column is ScanPacket, 
            listing the subdirectories for each scan, third column 
            indicating L2R or R2L, and the next columns indicating 
            the specimens in that scan
        CTdir : str, default is 'ScanOverviews'
            The directory to save all results. 
        Meshdir : str, default is 'Meshes'
            The directory to save individual bone fragments
        chopsheet : default is None
            Gives the option for the user to adjust the automatic chopping of bones.
        threshold : float, default is 2000
            Threshold to use with CT_side_seg function.
        padding : float, default is 15
            Padding to use when drawing bounding boxes.

        Returns
        -------
        df : pandas DataFrame
            Chopsheet recording locations where objects were cropped. Returned only if chopsheet=None.
   """

    #Number of bones in each scan
    num_bones = scanlayout.count(axis=1) - 4

    #Number of scans
    num_scans = len(scanlayout)

    #Make CTdir if it doesn't exist
    if not os.path.isdir(CTdir):
        os.mkdir(CTdir)

    #Make Meshdir if it doesn't exist
    if not os.path.isdir(Meshdir):
        os.mkdir(Meshdir)
 
    #Dataframe to hold chopsheet
    if chopsheet is None:
        df = pd.DataFrame(columns=['ScanPacket','Process','x1','x2','y1','y2','z1','z2'])

    #Loop over all scans
    for i in range(num_scans):

        #Check if we should process this scan or not
        if (chopsheet is None) or chopsheet['Process'][i]:

            #Get packet name
            subdir = scanlayout['ScanPacket'][i]
            if not isinstance(subdir, str):
                subdir = str(subdir)
            d = os.path.join(directory, subdir)
            print('\nLoading scan ' + d + '...')
            
            #Get bone names
            bone_names = scanlayout.iloc[i, 4:4+num_bones[i]].values.tolist()
            if scanlayout['CTHead2Tail'][i] =='R2L':
                print('Reversed')
                bone_names.reverse()

            #Read CT volume
            I, dx, dz, dicom_files = read_dicom_dir(d)

            #Process resolutions and check that they are all the same
            if np.sum(dx[:,0]!=dx[:,1]):
                sys.exit('Error: x,y resolutions are different!!')
            dx = dx[:,0]

            dx_mode = stats.mode(dx).mode.flatten()[0]
            dz_mode = stats.mode(dz).mode.flatten()[0]

            ind = (dx != dx_mode) | (dz != dz_mode)
            num_diff = np.sum(ind)
            if num_diff:
                print('Found %d DICOM images with different resolution, removing those...'%num_diff)
                I = I[~ind,:,:]
            dx = dx_mode
            dz = dz_mode

            if I is not None:

                if chopsheet is None: #Then chop and save to spreadsheet

                    K1 = CT_side_seg(I,num_bones[i],threshold=threshold,axis=1)    
                    K2 = CT_side_seg(I,num_bones[i],threshold=threshold,axis=2)    
                    x1,x2,y1,y2,z1,z2 = bone_bounding_boxes(K1,K2)

                    str_x1 = np.array2string(x1, separator=',')
                    str_x2 = np.array2string(x2, separator=',')
                    str_y1 = np.array2string(y1, separator=',')
                    str_y2 = np.array2string(y2, separator=',')
                    str_z1 = np.array2string(z1, separator=',')
                    str_z2 = np.array2string(z2, separator=',')

                    df = df.append({'ScanPacket':subdir, 'Process':True, 'x1':str_x1, 'x2':str_x2, 
                                                                         'y1':str_y1, 'y2':str_y2, 
                                                                         'z1':str_z1, 'z2':str_z2,}, ignore_index=True)
                else: #Load chop locations from spreadsheet

                    x1 = np.fromstring(chopsheet['x1'][i][1:-1], sep=',').astype(int)
                    x2 = np.fromstring(chopsheet['x2'][i][1:-1], sep=',').astype(int)
                    y1 = np.fromstring(chopsheet['y1'][i][1:-1], sep=',').astype(int)
                    y2 = np.fromstring(chopsheet['y2'][i][1:-1], sep=',').astype(int)
                    z1 = np.fromstring(chopsheet['z1'][i][1:-1], sep=',').astype(int)
                    z2 = np.fromstring(chopsheet['z2'][i][1:-1], sep=',').astype(int)

                J = draw_bounding_boxes(I,x1,x2,y1,y2,z1,z2,padding=padding)
                plt.imsave(os.path.join(CTdir, subdir + '.png'), J, cmap='gray')

                #Chop, and create overview of CT image of bone
                for j in range(len(bone_names)):
                    bonename = bone_names[j] + '_' + scanlayout['CT'][i]
                    print('Saving '+bonename+'...')

                    #Chop
                    Isub = crop_image(I,x1[j],x2[j],y1[j],y2[j],z1[j],z2[j],padding=padding)

                    #Correct for mirroring on some scans
                    if scanlayout['Mirrored'][i] == 'yes':
                        print('Mirrored')
                        Isub = Isub[::-1,:,:]

                    #Create and save overview of bone
                    Jsub = bone_overview(Isub)
                    plt.imsave(os.path.join(Meshdir,bonename + '.png'), Jsub, cmap='gray')

                    #Save 3D bone volume
                    np.savez_compressed(os.path.join(Meshdir,bonename + '.npz'), I=Isub, dx=dx, dz=dz, bonename=bonename)

    if chopsheet is None:
        return df



#Segment bones on a side view of CT scanning bed 
def CT_side_seg(I,num_bones,threshold=3000,axis=1):
    """ Segments bones on a side view of the CT scanning bed.
    
        Parameters
        ----------
        I : float array
            Image of scan.
        num_bones : int
            Number of bones present.
        threshold : float, default is 3000
            Threshold value to use on I.
        axis : int, default is 1
            Axis on which to sum I.

        Returns 
        -------
        K : float array
            Segmented image.
    """

    J = np.sum(I > threshold,axis=axis).T > 0
    J = J.astype(float)
    n = J.shape[0]; m = J.shape[1]
    labels = measure.label(J)

    #Restrict to the largest num_bones regions
    L = np.unique(labels)
    num = len(L)
    sizes = np.zeros(num)
    for i in range(num):
        sizes[i] = np.sum(labels==L[i])
    sort_ind = np.argsort(-sizes)[1:num_bones+1]
    L = L[sort_ind]

    #Sort the bones from left to right in the scanning bed
    pos = np.zeros(num_bones)
    X = np.ones((n,1))@np.reshape(np.arange(m),(1,m))
    for i in range(num_bones):
        pos[i] = np.mean(X[labels == L[i]])
    sort_ind = np.argsort(pos)
    L = L[sort_ind]
    
    #Format segmented image to return
    K = np.zeros_like(J)
    for i in range(num_bones):
        K[labels == L[i]] = i+1

    return K

#Finds bounding boxes for bones from CT_side_seg images
def bone_bounding_boxes(K1,K2):
    """ Finds the bounding boxes for the bones produced by CT_side_seg.

        Parameters
        ----------
        K1 : float array
            First axis of side segmentation.
        K2 : float array
            Second axis of side segmentation.

        Returns
        -------
        x1 : (num_bones,1) float array
            Min x dimension for each fragment.
        x2 : (num_bones,1) float array
            Max x dimension for each fragment.
        y1 : (num_bones,1) float array
            Min y dimension for each fragment.
        y2 : (num_bones,1) float array
            Max y dimension for each fragment.
        z1 : (num_bones,1) float array
            Min z dimension for each fragment.
        z2 : (num_bones,1) float array
            Max z dimension for each fragment.
    """

    n = K1.shape[0]; m = K1.shape[1]
    num_bones = int(np.max(K1))

    X = np.ones((n,1))@np.reshape(np.arange(m),(1,m))
    Z = np.reshape(np.arange(n),(n,1))@np.ones((1,m))
    x1 = np.zeros(num_bones,dtype=int)
    x2 = np.zeros(num_bones,dtype=int)
    y1 = np.zeros(num_bones,dtype=int)
    y2 = np.zeros(num_bones,dtype=int)
    z1 = np.zeros(num_bones,dtype=int)
    z2 = np.zeros(num_bones,dtype=int)
    for i in range(num_bones):
        x1[i] = np.min(X[K1 == i+1])
        x2[i] = np.max(X[K1 == i+1])
        y1[i] = np.min(Z[K2 == i+1])
        y2[i] = np.max(Z[K2 == i+1])
        z1[i] = np.min(Z[K1 == i+1])
        z2[i] = np.max(Z[K1 == i+1])

    return x1,x2,y1,y2,z1,z2

#Crop with padding 
def crop_image(I,x1,x2,y1,y2,z1,z2,padding=15):
    """ Crop an image with padding.

        Parameters
        ----------
        I : float array
            Image of scan.
        x1 : (num_bones,1) float array
            Min x dimension for each fragment.
        x2 : (num_bones,1) float array
            Max x dimension for each fragment.
        y1 : (num_bones,1) float array
            Min y dimension for each fragment.
        y2 : (num_bones,1) float array
            Max y dimension for each fragment.
        z1 : (num_bones,1) float array
            Min z dimension for each fragment.
        z2 : (num_bones,1) float array
            Max z dimension for each fragment.
        padding : float, default is 15
            Padding to use around bounding boxes.

        Returns
        -------
        Cropped image.
    """

    #Add padding to boxes
    x1 = max(x1 - padding,0)
    y1 = max(y1 - padding,0)
    z1 = max(z1 - padding,0)
    x2 = min(x2 + padding,I.shape[0]-1)
    y2 = min(y2 + padding,I.shape[1]-1)
    z2 = min(z2 + padding,I.shape[2]-1)

    return I[x1:x2,y1:y2,z1:z2]


#Draw bounding boxes on scan
def draw_bounding_boxes(I,x1,x2,y1,y2,z1,z2,padding=15):
    """ Draw bounding boxes on a scan.
    
        Parameters
        ----------
        I : float array
            Image of a scan.
        x1 : (num_bones,1) float array
            Min x dimension for each fragment.
        x2 : (num_bones,1) float array
            Max x dimension for each fragment.
        y1 : (num_bones,1) float array
            Min y dimension for each fragment.
        y2 : (num_bones,1) float array
            Max y dimension for each fragment.
        z1 : (num_bones,1) float array
            Min z dimension for each fragment.
        z2 : (num_bones,1) float array
            Max z dimension for each fragment.
        padding : float, default is 15
            Padding to use around bounding boxes.

        Returns
        -------
        Image with bounding boxes around it.
    """

    num_bones = len(x1)

    #Create side views
    I1 = np.sum(I, axis=1)
    I2 = np.sum(I, axis=2)
    I1 = I1 / np.max(I1)
    I2 = I2 / np.max(I2)

    #Add padding to boxes
    x1 = np.maximum(x1 - padding,0)
    y1 = np.maximum(y1 - padding,0)
    z1 = np.maximum(z1 - padding,0)
    x2 = np.minimum(x2 + padding,I.shape[0]-1)
    y2 = np.minimum(y2 + padding,I.shape[1]-1)
    z2 = np.minimum(z2 + padding,I.shape[2]-1)

    #Draw boxes on I1
    for i in range(num_bones):
        I1[x1[i]:x2[i],z1[i]] = 1
        I1[x1[i]:x2[i],z2[i]] = 1
        I1[x1[i],z1[i]:z2[i]] = 1
        I1[x2[i],z1[i]:z2[i]] = 1

    #Draw boxes on I2
    for i in range(num_bones):
        I2[x1[i]:x2[i],y1[i]] = 1
        I2[x1[i]:x2[i],y2[i]] = 1
        I2[x1[i],y1[i]:y2[i]] = 1
        I2[x2[i],y1[i]:y2[i]] = 1

    return np.hstack((I1,I2)).T

def image_segmentation(I,lam=1,eps=0.5,min_iter=20,max_iter=200,stopping_crit=10,visualize=False):
    """ Segment a given image.

        Parameters
        ----------
        I : float array
            Image of a scan.
        lam : float, default is 1
            Used to calculate changes.
        eps : float, default is 0.5
            Used to calculate changes.
        min_iter : int, default is 20
            Minimum number of iterations on which to run.
        max_iter : int, default is 200
            Maximum number of iterations on which to run.
        stopping_crit : float, default is 10
            Used to stop before max iterations.
        visualize : boolean, default is False
            If True, plot results.

        Returns
        -------
        Positive values of u.
    """

    n = I.shape[0]
    m = I.shape[1]
    
    Y,X = np.mgrid[:n,:m]
    u =  np.sin(20*X*np.pi/m)*np.sin(20*Y*np.pi/n)
    print(np.max(X))
    print(np.max(Y))

    num_changed = np.inf
    count = 0
    tin_old = np.inf

    #Set up indexing arrays
    Yp = np.arange(1,n+1); Yp[n-1]=n-2
    Ym = np.arange(-1,n-1); Ym[0]=1
    Xp = np.arange(1,m+1); Xp[m-1]=m-2
    Xm = np.arange(-1,m-1); Xm[0]=1

    if visualize:
        plt.imshow(I,cmap='gray')
        plt.axis('off')
        seg = plt.contour(X,Y,u,levels=[0],colors=('r'))

    dt = np.pi*eps**2/4/lam
    while (count < min_iter) or (num_changed > stopping_crit and count < max_iter):
        Rin = u>0; Rout = u<=0
        Iin = I[Rin]; Iout = I[Rout]
        tin = np.sum(Rin); tout= np.sum(Rout)

        cin = 0; cout =0
        if tin > 0:
           cin = np.mean(Iin)
        if tout > 0:
           cout = np.mean(Iout)
        num_changed = abs(tin - tin_old)
        tin_old = tin;

        #Compute gradient
        GE = u[Yp,:] - u 
        GW = u[Ym,:] - u
        GN = u[:,Xm] - u  
        GS = u[:,Xp] - u 

        delta = eps/(np.pi*(eps**2 + u**2))
        div = (GN/np.sqrt(eps**2 + GN**2) + GE/np.sqrt(eps**2 + GE**2) + GS/np.sqrt(eps**2 + GS**2) + GW/np.sqrt(eps**2 + GW**2))
        fid = (I - cout)**2 - (I - cin)**2

        u = u + dt*delta*(lam*div + fid)
    
        if visualize:
            seg.collections[0].remove()
            seg = plt.contour(X,Y,u,levels=[0],colors=('r'))
            plt.pause(0.01)

        count = count+1;
        print(count)

    if cin < cout:
        u = -u

    return u > 0

def seg_plot(I,u):
    """ Plot the segmented image.

        Parameters
        ----------
        I : float array
            Image of a scan.
        u : boolean array
            The segmentation of that image.

        Returns
        -------
        None
    """

    n = I.shape[0]
    m = I.shape[1]
    Y,X = np.mgrid[:n,:m]
    plt.imshow(I,cmap='gray')
    plt.axis('off')
    plt.contour(X,Y,u,levels=[0],colors=('r'))

def seg_adjacency_matrix(u):

    n = I.shape[0]
    m = I.shape[1]

    Y,X = np.mgrid[:n,:m]
    X = X.flatten()
    Y = Y.flatten()
    u = u.flatten()

    C = np.ravel_multi_index((X,Y),(m,n),mode='clip')
    E = np.ravel_multi_index((X+1,Y),(m,n),mode='clip')
    W = np.ravel_multi_index((X-1,Y),(m,n),mode='clip')
    N = np.ravel_multi_index((X,Y-1),(m,n),mode='clip')
    S = np.ravel_multi_index((X,Y+1),(m,n),mode='clip')
    
    WE = u(C) == u(E)
    WW = u(C) == u(W)
    WS = u(C) == u(S)
    WN = u(C) == u(N)

    ME = sparse.coo_matrix((WE, (C,E)),shape=(n*m,n*m)).to_csr()
    MW = sparse.coo_matrix((WW, (C,W)),shape=(n*m,n*m)).to_csr()
    MS = sparse.coo_matrix((WS, (C,S)),shape=(n*m,n*m)).to_csr()
    MN = sparse.coo_matrix((WN, (C,N)),shape=(n*m,n*m)).to_csr()

    M = ME + MW + MS + MN
    M = M - sparse.spdiags(M.diagonal(),0,n*m,n*m)
    
    ind = u > 0
    M = M[u>0,:]
    M = M[:,u>0]
    X = X[u>0]
    Y = Y[u>0]

    return M,X,Y

