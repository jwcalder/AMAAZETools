/* svipca_mex.c - Spherical volume invariant and PCA on local neighborhoods
 *
 *  Computes spherical volume invariant and PCA on local neighborhoods using method from
 *
 *  "Computation of circular area and spherical volume invariants via boundary integrals",
 *  in preparation, 2018.
 *
 *  Use the wrapper svipca.m from Matlab. Direct usage from Matlab is
 *
 *    [S,M] = svipca_mex(P,T,r,ID,eps_svi,eps_pca,prog);
 *  
 *  where
 *
 * Inputs:
 *  P = nx3 array of vertices
 *  T = mx3 array of triangle indices (m=number triangles)
 *  r = radius for sphere
 *  ID = boolean vector of vertices at which to compute volume 
 *  eps_svi,eps_pca = Error tolerance for mesh refinement integration method
 *  prog = toggles whether to display progress
 *
 * Outputs:
 *  S = Spherical volume invariant
 *  M = PCA matrix
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
   double r,*P_ptr, *S, *M_ptr, eps_svi, eps_pca;
   bool *ID, prog;

   /* Check for proper number of arguments. */
   if(nrhs<5) {
      mexErrMsgIdAndTxt("MATLAB:svipca_mex:invalidNumInputs","Inputs: (P,T,r,ID,eps_svi,eps_pca,prog).");
   } 

   /* Retrieve input.*/
   n = mxGetN(prhs[0]); /*Num vertices*/
   m = mxGetN(prhs[1]); /*Num triangles*/
   r = *((double *) mxGetPr(prhs[2])); /*Retrieve radius*/
   eps_svi = *((double *) mxGetPr(prhs[4])); /*Retrieve epsilon*/
   eps_pca = *((double *) mxGetPr(prhs[5])); /*Retrieve epsilon*/
   prog = *((bool *) mxGetPr(prhs[6])); /*Retrieve progress toggle*/
   P_ptr = (double *) mxGetPr(prhs[0]);
   T_ptr = (int *) mxGetPr(prhs[1]);
   ID = (bool *) mxGetPr(prhs[3]);

   /*Allocate output arrays*/
   plhs[0] = mxCreateDoubleMatrix((mwSize)n, (mwSize)1, mxREAL);
   plhs[1] = mxCreateDoubleMatrix((mwSize)9, (mwSize)n, mxREAL);
   S = mxGetPr(plhs[0]);
   M_ptr = mxGetPr(plhs[1]);
   
   svipca(P_ptr,n,T_ptr,m,ID,r,eps_svi,eps_pca,prog,S,M_ptr);
}

