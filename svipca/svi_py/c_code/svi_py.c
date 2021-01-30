/* svi_py.c - Spherical volume invariant
 *
 *  Computes spherical volume invariant using method from
 *
 *  "Computation of circular area and spherical volume invariants via boundary integrals",
 *  in preparation, 2018.
 *
 *  Usage from Python is
 *
 *     import svi_module
 *     svi_module.svi(P,T,ID,r,eps,prog,S,G)
 *  
 *  where
 *
 * Inputs:
 *  P = nx3 array of vertices
 *  T = mx3 array of triangle indices (m=number triangles)
 *  r = radius for sphere
 *  ID = boolean vector of vertices at which to compute volume 
 *  eps = Error tolerance for mesh refinement integration method
 *  prog = toggles whether to display progress
 *
 * Outputs:
 *  S = Spherical volume invariant
 *  G = Gamma values for each vertex
 *
 *  Compile with command
 *       python3 svi_setup.py build_ext --inplace
 *
 *  Author: Jeff Calder, 2018.
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
static PyMethodDef SVIMethods[] =
{
   {"svi", svi_py, METH_VARARGS, "Computes the spherical volume invariant for a triangulated surface"},
   {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
/* module initialization */
/* Python version 3*/
static struct PyModuleDef cModPyDem =
{
   PyModuleDef_HEAD_INIT,
   "svi_module", "Some documentation",
   -1,
   SVIMethods
};

PyMODINIT_FUNC
PyInit_svi_module(void)
{
   import_array();
   return PyModule_Create(&cModPyDem);
}

#else

/* module initialization */
/* Python version 2 */
PyMODINIT_FUNC
initsvi_module(void)
{
   (void) Py_InitModule("svi_module", SVIMethods);
   /* IMPORTANT: this must be called */
   import_array();
}
#endif




