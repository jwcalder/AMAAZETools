/* cextensions.c - C extensions for AMAAZETools
 *
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include "vector_operations.h"
#include "memory_allocation.h"
#include "mesh_operations.h"
#include "svi_computations.h"

/*  wrapped cosine function */
static PyObject* svipca_py(PyObject* self, PyObject* args)
{

   double r;
   double eps_svi;
   double eps_pca;
   double prog;
   PyArrayObject *P_array;
   PyArrayObject *T_array;
   PyArrayObject *ID_array;
   PyArrayObject *S_array;
   PyArrayObject *M_array;
    
   /*  parse single numpy array argument */
   if (!PyArg_ParseTuple(args, "O!O!O!ddddO!O!", &PyArray_Type, &P_array, &PyArray_Type, &T_array, &PyArray_Type, &ID_array,&r,&eps_svi,&eps_pca,&prog,&PyArray_Type, &S_array, &PyArray_Type,&M_array))
      return NULL;

   npy_intp *dim =  PyArray_DIMS(P_array);
   int n = dim[0]; //Number of vertices
   dim =  PyArray_DIMS(T_array);
   int m = dim[0]; //Number of faces
   double *P = (double *) PyArray_DATA(P_array);
   int *T = (int *) PyArray_DATA(T_array);
   bool *ID = (bool *) PyArray_DATA(ID_array);
   double *S = (double *) PyArray_DATA(S_array);
   double *M = (double *) PyArray_DATA(M_array);

   svipca(P,n,T,m,ID,r,eps_svi,eps_pca,(bool)prog,S,M);

   Py_INCREF(Py_None);
   return Py_None;
}

/*  wrapped cosine function */
static PyObject* svi_py(PyObject* self, PyObject* args)
{

   double r;
   double eps;
   double prog;
   PyArrayObject *P_array;
   PyArrayObject *T_array;
   PyArrayObject *ID_array;
   PyArrayObject *S_array;
   PyArrayObject *G_array;

   /*  parse single numpy array argument */
   if (!PyArg_ParseTuple(args, "O!O!O!dddO!O!", &PyArray_Type, &P_array, &PyArray_Type, &T_array, &PyArray_Type, &ID_array,&r,&eps,&prog,&PyArray_Type, &S_array,&PyArray_Type, &G_array))
      return NULL;

   npy_intp *dim =  PyArray_DIMS(P_array);
   int n = dim[0]; //Number of vertices
   dim =  PyArray_DIMS(T_array);
   int m = dim[0]; //Number of faces
   double *P = (double *) PyArray_DATA(P_array);
   int *T = (int *) PyArray_DATA(T_array);
   bool *ID = (bool *) PyArray_DATA(ID_array);
   double *S = (double *) PyArray_DATA(S_array);
   double *G = (double *) PyArray_DATA(G_array);

   svi(P,n,T,m,ID,r,eps,(bool)prog,S,G);

   Py_INCREF(Py_None);
   return Py_None;
}


/*  define functions in module */
static PyMethodDef CExtensionsMethods[] =
{
   {"svipca", svipca_py, METH_VARARGS, "Computes the spherical volume invariant & pca for a triangulated surface"},
    {"svi", svi_py, METH_VARARGS, "Computes the spherical volume invariant for a triangulated surface"},
   {NULL, NULL, 0, NULL}
};

/* module initialization */
/* Python version 3*/
static struct PyModuleDef cModPyDem =
{
   PyModuleDef_HEAD_INIT,
   "cextensions", "C code extensions for AMAAZETools package",
   -1,
   CExtensionsMethods
};

PyMODINIT_FUNC PyInit_cextensions(void)
{
   import_array();
   return PyModule_Create(&cModPyDem);
}



