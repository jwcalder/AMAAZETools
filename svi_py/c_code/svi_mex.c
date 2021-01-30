/* svi_mex.c - Spherical volume invariant
 *
 *  Computes spherical volume invariant using method from
 *
 *  "Computation of circular area and spherical volume invariants via boundary integrals",
 *  in preparation, 2018.
 *
 *  Use the wrapper svi.m from Matlab. Direct usage from Matlab is
 *
 *    [S,G] = svi_mex(P,T,r,ID,eps,prog);
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
 *  Author: Jeff Calder, 2018.
 *
 */

#include "mex.h"
#include "math.h"
#include "vector_operations.h"
#include "memory_allocation.h"
#include "mesh_operations.h"
#include "svi_computations.h"
 

//Main subroutine
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ){

   //Input argument handling
   int i, j, n, m, *T_ptr;
   double r,*P_ptr, *S, *Q, eps;
   bool *ID, prog;

   /* Check for proper number of arguments. */
   if(nrhs<6) {
      mexErrMsgIdAndTxt("MATLAB:svi_mex:invalidNumInputs","Inputs: (P,T,r,ID,eps,prog).");
   } 

   /* Retrieve input.*/
   n = mxGetN(prhs[0]); /*Num vertices*/
   m = mxGetN(prhs[1]); /*Num triangles*/
   r = *((double *) mxGetPr(prhs[2])); /*Retrieve radius*/
   eps = *((double *) mxGetPr(prhs[4])); /*Retrieve epsilon*/
   prog = *((bool *) mxGetPr(prhs[5])); /*Retrieve progress toggle*/
   P_ptr = (double *) mxGetPr(prhs[0]);
   T_ptr = (int *) mxGetPr(prhs[1]);
   ID = (bool *) mxGetPr(prhs[3]);

 
   /*Allocate output arrays*/
   plhs[0] = mxCreateDoubleMatrix((mwSize)n, (mwSize)1, mxREAL);
   S = mxGetPr(plhs[0]);
   plhs[1] = mxCreateDoubleMatrix((mwSize)n, (mwSize)1, mxREAL);
   Q = mxGetPr(plhs[1]);

   svi(P_ptr,n,T_ptr,m,ID,r,eps,prog,S,Q);
}

