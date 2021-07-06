# Begin with any necessary import statements, as usual

def function(arg1,arg2=None):
    """ One line general description of what function does.

        Parameters
        ----------
        arg1 : data type and shape of arg1
               High level description of what arg1 is.
        arg2 : data type and shape of arg2, followed by optional keyword
               High level description of what arg2 is.

        Returns
        -------
        out1 : data type and shape of out1
               High level description of what out1 is.
        out2 : data type and shape of out2
               High level description of what out2 is. 
        
        Raises
        ------
        ValueError
            Condition that causes above error type to be raised.
        AttributeError
            Condition that causes above error type to be raised.
        """

    out1 = arg1
    out2 = arg2

    return out1,out2

def function_example(3d_point_cloud,bool_flag=False):   
    """ One line general description of what function does.

        Parameters
        ----------
        3d_point_cloud : (n,3) numpy array
            Array containing the x-y-z coords of a 3d point cloud.
        bool_flag : boolean, optional
            A boolean flag to toggle error checking.

        Returns
        -------
        out : (n,3) numpy array
            Array containing the x-y-z coords of a 3d point cloud.
        
        Raises
        ------
        ValueError
            If the length of our point cloud exceeds 10,000.
        """

    if bool and len(3d_point_cloud) > 10000:
        raise ValueError('length of 3d_point_cloud cannot exceed 10,000 points')

    out = 3d_point_cloud
    return out
