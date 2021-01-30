/* mesh_operations.c - 
 *
 *  Basic subroutines for computing with triangulated meshes
 *
 *  Author: Jeff Calder, 2018.
 */
 

#include "stdlib.h"
#include "stdbool.h"
#include "stdio.h"
#include "math.h"
#include "vector_operations.h"
#include "mesh_operations.h"
#include "memory_allocation.h"

/*Compute minimum distance from edge xy to origin*/
static inline double edge_dist(double *x, double *y){

   double p[3], t;

   sub(y,x,p);
   t = dot(p,y)/norm_squared(p);
   t = MAX(MIN(t,1),0);
   mult(p,t,p);
   return dist_squared(y,p);

}

/*Compute unit outward normal of triangle xyz
 * Also returns triangle area*/
static inline double tri_normal(double *x, double *y, double *z, double *nu){
   
   double p[3], q[3], nnu, t;

   sub(y,x,p);
   sub(z,x,q);
   cross(p,q,nu);
   nnu = norm(nu);
   t = 1/nnu;mult(nu,t,nu);
   return nnu/2;
}

/*Gram-Schmidt othornoralization: 
 * Returns normal vector in direction x - proj_y x
 * x and y assumed to be unit normal vectors
 */
static inline void gram_schmidt(double *x, double *y, double *p){

   double a, b;
   b = dot(x,y);
   mult(y,b,p);
   sub(x,p,p);
   a = 1/norm(p);
   mult(p,a,p);
}

/*Compute minimum distance from triangle xyz to origin
 * Works by projecting origin onto triangle
 * x,y,z = vertices of triangle
 * Returns distance to origin
 * Also returns angle theta used in exact integration
 */
double tri_dist(double *x, double *y, double *z){

   double p[3], nxy[3], nyz[3], nzx[3], pxy[3], pyz[3], pzx[3], xs[3], nu[3], txy, tyz, tzx, t;

   sub(y,x,pxy); t=1/norm(pxy); mult(pxy,t,pxy);
   sub(z,y,pyz); t=1/norm(pyz); mult(pyz,t,pyz);
   sub(x,z,pzx); t=1/norm(pzx); mult(pzx,t,pzx);

   gram_schmidt(pyz,pxy,nxy);  //Normal to xy
   gram_schmidt(pzx,pyz,nyz);  //Normal to yz
   gram_schmidt(pxy,pzx,nzx);  //Normal to zx
   
   txy = dot(x,nxy);
   tyz = dot(y,nyz);
   tzx = dot(z,nzx);

   //Compute projection xs of origin (0,0,0) onto triangle xyz
   if(txy > 0){ //projection to plane lies outside of edge xy
      t = MIN(MAX(-dot(x,pxy),0),dist(x,y));
      mult(pxy,t,p);
      add(x,p,xs);
   }else if(tyz > 0){ //projection to plane lies outside of edge yz
      t = MIN(MAX(-dot(y,pyz),0),dist(y,z));
      mult(pyz,t,p);
      add(y,p,xs);
   }else if(tzx > 0){ //projection to plane lies outside of edge zx
      t = MIN(MAX(-dot(z,pzx),0),dist(z,x));
      mult(pzx,t,p);
      add(z,p,xs);
   }else{ //Projection lies inside triangle
      tri_normal(x,y,z,nu);
      t = dot(x,nu);
      mult(nu,t,xs); //projection of origin to triangle plane
   }   

   return norm_squared(xs);
}

/*Returns for each vertex the indices of adjacent triangles
 * n = number vertices
 * m = number of triangles
 * T = triangle connectivity list
 * Returns nxk array L[i][j] = index of jth triangle adjacent to vertex i
 *                   L[i][0] = number of adjacent triangles
 */
int** vertex_to_triangle_list(int **T, int n, int m){
  
   int **L;
   int i,j,k,l,p,q;
   int *C = vector_int(n,0); 
   for(i=0;i<m;i++){
      C[T[i][0]]++; C[T[i][1]]++; C[T[i][2]]++;
   }
   k = C[0];
   for(i=0;i<n;i++)
      k = MAX(k,C[i]);
   L = array_int(n,k+1,-1);
   for(i=0;i<n;i++)
      L[i][0] = C[i];

   for(i=0;i<m;i++){
      for(j=0;j<3;j++){
         p = T[i][j];
         bool addi = true;
         q = 1;
         for(l=1;l<=L[p][0];l++){
            addi = addi && L[p][l] != i;
            if(L[p][l] != -1)
               q++;
         }
         if(addi)
            L[p][q] = i;
      }       
   }

   return L; 
}

/*Returns the face graph (triangle adjacency matrix)
 * n = number vertices 
 * m = number of triangles, 
 * T = triangle connectivity list
 * L = vertex to triangle list
 * Returns mx3 array F[i][j] = index of jth triangle adjacent to triangle i
 * Also updates B to encode mesh boundary points (B[p]=false iff p boundary vertex)
 */
int** face_graph(int **T, int **L, bool *B, int n, int m){
  
   int i,j,k,l,l2,p,q;
   
   //Allocate memory
   int **F = array_int(m,3,-1);

   for(i=0;i<m;i++){
      for(l=0;l<3;l++){
         l2 = (l+1)%3;
         p = T[i][l]; q = T[i][l2];
         for(j=1;j<=L[p][0];j++){
            for(k=1;k<=L[q][0];k++){
               if(L[p][j] == L[q][k] && L[p][j] != i)
                  F[i][l] = L[p][j]; 
            }
         }
         if(F[i][l] == -1){
            //printf("boundary!!\n");
            B[p] = false; B[q] = false;
         }
      }
   }
  
   return F; 

}
/*Non-recursive depth first search to find local patch for integration
 * Replaces expensive range-search
 * T = triangles
 * P = vertices
 * F = face graph adjacency
 * NN = output neighbor indices
 * v = vector of visited triangle indices
 * b = output boundary triangles
 * num = current triangle number
 * ind = index of root triangle
 * r2 = r^2
 * i = current index
 * Returns true if B(x,r) does not interesect boundary of mesh, and false otherwise
 */
bool depth_first_search_nr(int **T, double **P, int **F, int *NN, bool *v, bool *b, int *num, int ind, double r2, int i, int *stack){

   int j,p,q,k,t,stack_size;
   bool bi = true;
   double x[3], y[3], z[3], rmin, rmax;

   stack[0] = ind;
   stack_size = 1;  //Next position in stack

   while(stack_size > 0){

      stack_size--;
      ind = stack[stack_size];
      if(!(v[ind])){ //If not already visited

         //Extract vertices
         tri_vertices(T,P,x,y,z,ind,i);

         //Compute distance from triangle to origin
         rmin = tri_dist(x,y,z);
         rmax = MAX3(norm_squared(x),norm_squared(y),norm_squared(z));
         
         //Check if within B(x,r)
         if(rmin < r2){
            v[ind] = true;
            NN[*num] = ind;
            if(rmax > r2)
               b[*num] = true;
            else
               b[*num] = false;
            ++*num;

            //Add neighbors to stack
            for(j=0;j<3;j++){
               t = F[ind][j];
               if(t == -1){
                  k = (j+1)%3;
                  p = T[ind][j]; q = T[ind][k]; 
                  x[0] = P[p][0] - P[i][0]; x[1] = P[p][1] - P[i][1]; x[2] = P[p][2] - P[i][2];
                  y[0] = P[q][0] - P[i][0]; y[1] = P[q][1] - P[i][1]; y[2] = P[q][2] - P[i][2];
                  bi = bi && (edge_dist(x,y) >= r2);
               }else if(!(v[t])){
                  stack[stack_size] = t;
                  stack_size++;
               }
            }
         }
      }
   }
   return bi;
}
/*Non-recursive breadth first search to find local patch for integration
 * Replaces expensive range-search
 * T = triangles
 * P = vertices
 * F = face graph adjacency
 * NN = output neighbor indices
 * v = vector of visited triangle indices
 * b = output boundary triangles
 * num = current triangle number
 * ind = index of root triangle
 * r2 = r^2
 * i = current index
 * Returns true if B(x,r) does not interesect boundary of mesh, and false otherwise
 */
bool breadth_first_search(int **T, double **P, int **F, int *NN, bool *v, bool *b, int *num, int ind, double r2, int i, int *stack){

   int j,p,q,k,t,stack_next,stack_curr;
   bool bi = true;
   double x[3], y[3], z[3], rmin, rmax;

   stack[0] = ind;
   stack_next = 1;  //Next position in stack
   stack_curr = 0;  //Current position in stack

   while(stack_curr < stack_next){

      ind = stack[stack_curr];
      stack_curr++;
      if(!(v[ind])){ //If not already visited

         //Extract vertices
         tri_vertices(T,P,x,y,z,ind,i);

         //Compute distance from triangle to origin
         rmin = tri_dist(x,y,z);
         rmax = MAX3(norm_squared(x),norm_squared(y),norm_squared(z));
         
         //Check if within B(x,r)
         if(rmin < r2){
            v[ind] = true;
            NN[*num] = ind;
            if(rmax > r2)
               b[*num] = true;
            else
               b[*num] = false;
            ++*num;

            //Add neighbors to stack
            for(j=0;j<3;j++){
               t = F[ind][j];
               if(t == -1){
                  k = (j+1)%3;
                  p = T[ind][j]; q = T[ind][k]; 
                  x[0] = P[p][0] - P[i][0]; x[1] = P[p][1] - P[i][1]; x[2] = P[p][2] - P[i][2];
                  y[0] = P[q][0] - P[i][0]; y[1] = P[q][1] - P[i][1]; y[2] = P[q][2] - P[i][2];
                  bi = bi && (edge_dist(x,y) >= r2);
               }else if(!(v[t])){
                  stack[stack_next] = t;
                  stack_next++;
               }
            }
         }
      }
   }
   return bi;
}
/*Depth first search to find local patch for integration
 * Replaces expensive range-search
 * T = triangles
 * P = vertices
 * F = face graph adjacency
 * NN = output neighbor indices
 * v = vector of visited triangle indices
 * b = output boundary triangles
 * num = current triangle number
 * ind = index of current triangle
 * r2 = r^2
 * i = current index
 * Returns true if B(x,r) does not interesect boundary of mesh, and false otherwise
 */
bool depth_first_search(int **T, double **P, int **F, int *NN, bool *v, bool *b, int *num, int ind, double r2, int i){

   int j,p,q,k,t;
   bool bi = true;
   double x[3], y[3], z[3], rmin, rmax;

   if(!(v[ind])){

      //Extract vertices
      tri_vertices(T,P,x,y,z,ind,i);

      //Compute distance from triangle to origin
      rmin = tri_dist(x,y,z);
      rmax = MAX3(norm_squared(x),norm_squared(y),norm_squared(z));

      //Check if within B(x,r)
      if(rmin < r2){
         v[ind] = true;
         NN[*num] = ind;
         if(rmax > r2)
            b[*num] = true;
         else
            b[*num] = false;
         ++*num;

         //Call depth first search at neighbors
         for(j=0;j<3;j++){
            t = F[ind][j];
            if(t == -1){
               k = (j+1)%3;
               p = T[ind][j]; q = T[ind][k]; 
               x[0] = P[p][0] - P[i][0]; x[1] = P[p][1] - P[i][1]; x[2] = P[p][2] - P[i][2];
               y[0] = P[q][0] - P[i][0]; y[1] = P[q][1] - P[i][1]; y[2] = P[q][2] - P[i][2];
               bi = bi && (edge_dist(x,y) >= r2);
            }else{
               bi = bi && depth_first_search(T,P,F,NN,v,b,num,t,r2,i); 
            }
         }
      }
   }

   return bi;
}

/*Numerical integration via triangle refinement. Used at boundary triangles
 * x,y,z = vertices of triangle
 * dx,dy,dz = squared norms of x,y,z
 * A = area of triangle
 * eta = x dot triangle unit normal (same as y or z)
 * eps = tolerance
 * r,r2,r3 = r,r^2,r^3
 * level = current triangle refinement level
 * maxlevel = maximum refinement level
 * num_subtri = running count of number of subtriangles
 * Returns value of integral 
 */

   
double integrate(double *x, double *y, double *z, double dx, double dy, double dz, double A, double eta, double eps, double r, double r2, double r3, short level, short *maxlevel, int *num_subtri, int i, int j, double (*integrate_approx) (double *, double *, double *, double, double, double, double, int, int), double (*integrate_exact) (double *, double *, double *, double, double, int, int), bool (*error_computation) (double, double, double, double, double)){

   double p[3], dp, o=0, rmin;
   double A2 = A/2;
   double Lxy = dist_squared(x,y);
   double Lyz = dist_squared(y,z);
   double Lxz = dist_squared(x,z);

   //Update level counters
   if(level == -1)
      *num_subtri = 0;
   level++; *maxlevel = MAX(*maxlevel,level);
   
   //If error not small enough, continue refining
   if(error_computation(MAX3(Lxy,Lyz,Lxz), eta, r, r2, eps)){ 

      //Split triangle along longest side and integrate two halves separately
      if(Lxy >= MAX(Lyz,Lxz)){ //Split along edge xy, p=midpoint
         average(x,y,p); dp = norm_squared(p);

         //Triangle xpz
         rmin = tri_dist(x,p,z);
         if(MAX3(dx,dp,dz) <= r2){ 
            o = o + integrate_exact(x,p,z,r2,r3,i,j);
            ++*num_subtri;
         }else if(rmin < r2){
            o = o + integrate(x,p,z,dx,dp,dz,A2,eta,eps,r,r2,r3,level,maxlevel,num_subtri,i,j,integrate_approx,integrate_exact,error_computation);
         }else{
            ++*num_subtri;
         }

         //Triangle yzp
         rmin = tri_dist(y,z,p);
         if(MAX3(dy,dz,dp) <= r2){
            o = o + integrate_exact(y,z,p,r2,r3,i,j);
            ++*num_subtri;
         }else if(rmin < r2){
            o = o + integrate(y,z,p,dy,dz,dp,A2,eta,eps,r,r2,r3,level,maxlevel,num_subtri,i,j,integrate_approx,integrate_exact,error_computation);
         }else{
            ++*num_subtri;
         }
      }else if(Lyz >= MAX(Lxy,Lxz)){  //Split along edge yz, p=midpoint
         average(y,z,p); dp = norm_squared(p);

         //Triangle xyp
         rmin = tri_dist(x,y,p);
         if(MAX3(dx,dy,dp) <= r2){
            o = o + integrate_exact(x,y,p,r2,r3,i,j);
            ++*num_subtri;
         }else if(rmin < r2){
            o = o + integrate(x,y,p,dx,dy,dp,A2,eta,eps,r,r2,r3,level,maxlevel,num_subtri,i,j,integrate_approx,integrate_exact,error_computation);
         }else{
            ++*num_subtri;
         }
         //Triangle xpz
         rmin = tri_dist(x,p,z);
         if(MAX3(dx,dp,dz) <= r2){
            o = o + integrate_exact(x,p,z,r2,r3,i,j);
            ++*num_subtri;
         }else if(rmin < r2){
            o = o + integrate(x,p,z,dx,dp,dz,A2,eta,eps,r,r2,r3,level,maxlevel,num_subtri,i,j,integrate_approx,integrate_exact,error_computation);
         }else{
            ++*num_subtri;
         }
      }else{ //Split along edge xz, p=midpoint
         average(x,z,p); dp = norm_squared(p);

         //Triangle xyp
         rmin = tri_dist(x,y,p);
         if(MAX3(dx,dy,dp) <= r2){
            o = o + integrate_exact(x,y,p,r2,r3,i,j);
            ++*num_subtri;
         }else if(rmin < r2){
            o = o + integrate(x,y,p,dx,dy,dp,A2,eta,eps,r,r2,r3,level,maxlevel,num_subtri,i,j,integrate_approx,integrate_exact,error_computation);
         }else{
            ++*num_subtri;
         }
         //Triangle yzp
         rmin = tri_dist(y,z,p);
         if(MAX3(dy,dz,dp) <= r2){
            o = o + integrate_exact(y,z,p,r2,r3,i,j);
            ++*num_subtri;
         }else if(rmin < r2){
            o = o + integrate(y,z,p,dy,dz,dp,A2,eta,eps,r,r2,r3,level,maxlevel,num_subtri,i,j,integrate_approx,integrate_exact,error_computation);
         }else{
            ++*num_subtri;
         }
      }
   }else{ //Error condition is met; use approximate integration
      o = integrate_approx(x,y,z,A,eta,r2,r3,i,j);
      ++*num_subtri;
   }

   return o;
}


