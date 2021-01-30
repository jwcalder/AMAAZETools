/* svi_computations.c - 
 *
 *  Subroutines specific to computing the spherical volume invariant
 *
 *  Author: Jeff Calder, 2018.
 */

//NOTE: The two following lines should be commented out for compliation with Python, and uncommented for compliation with Matlab.
//#include "mex.h"
//#define MEX


#include "stdlib.h"
#include "stdio.h"
#include "stdbool.h"
#include "math.h"
#include "vector_operations.h"
#include "mesh_operations.h"
#include "memory_allocation.h"
#include "svi_computations.h"

/*Compute unit outward normal of triangle xyz
 * Also returns triangle area*/
static inline double tri_normal(double *x, double *y, double *z, double *nu){
   
   double p[3], q[3], nnu, t;

   sub(y,x,p);
   sub(z,x,q);
   cross(p,q,nu);
   nnu = norm(nu);
   t = 1/nnu; mult(nu,t,nu);
   return nnu/2;
}

//Main subroutine to compute spherical volume invaraint and PCA on local neighborhoods
void svipca(double *P_ptr, int n, int *T_ptr, int m, bool *ID, double r, double eps_svi, double eps_pca, bool prog, double *S, double *M_ptr){

   double dx,dy,dz,eta,A,sgam, x[3], y[3], z[3], nu[3], mi[3], cij[9], **P, **M;
   double r3 = r*r*r;
   double r2 = r*r;
   double vol = 4*PI*r3/3;
   int i, j, t, q, **L, **F, *NN, *stack, tri_num, num_subtri, max_num_subtri, ki, kj, **T;
   short maxlevel;
   bool *B, *visited, *boundary, bi;

   //Setup 2D array pointers to input
   P = (double**)malloc(n * sizeof(double*));
   T = (int**)malloc(m * sizeof(int*));
   for(i=0;i<n;i++)
      P[i] = P_ptr + 3*i;
   for(i=0;i<m;i++)
      T[i] = T_ptr + 3*i;

   //Setup 2D array pointers to output
   M = (double**)malloc(n * sizeof(double*));
   for(i=0;i<n;i++)
      M[i] = M_ptr + 9*i;

   B = vector_bool(n,true);  //Records boundary points of mesh
   NN = vector_int(m,-1);    //Nearest neighbors
   stack = vector_int(m,-1);    //Stack
   visited = vector_bool(m,false); //Records which triangles have alread been visited in depth first search
   boundary = vector_bool(m,false);//Records which triangles intersect boundary of ball B(x,r)

   //Compute face graph
   L = vertex_to_triangle_list(T, n, m);
   F = face_graph(T, L, B, n, m);

   maxlevel = 0;
   max_num_subtri = 0;
   int interval = n/100;
   if(prog){
#ifdef MEX
      mexPrintf("Progress...0%%"); mexEvalString("drawnow;"); 
#else 
      printf("Progress...0%%"); fflush(stdout); 
#endif
   }
   int perc = 0;
   /*Main loop over vertices*/
   for(i=0;i<n;i++){

      S[i] = -1; //Default value if not computed later
      for(j=0;j<9;j++)
         M[i][j] = j;

      if((i % interval == 1 || i == n-1) && prog){
         if(perc < 10){
#ifdef MEX
            mexPrintf("\b\b%d%%",(100*(i+1))/n); mexEvalString("drawnow;"); 
#else
            printf("\b\b%d%%",(100*(i+1))/n); fflush(stdout); 
#endif
         }else{
#ifdef MEX
            mexPrintf("\b\b\b%d%%",(100*(i+1))/n); mexEvalString("drawnow;"); 
#else
            printf("\b\b\b%d%%",(100*(i+1))/n); fflush(stdout); 
#endif
         }
         perc = (100*(i+1))/n;
      }

      if(B[i] && ID[i] && L[i][0] !=0){ //If vertex i is not a boundary vertex, then compute Gamma
         sgam = svigamma(T,P,L,i);
         S[i] = vol*sgam;

         //Depth first search to find neighboring triangles
         tri_num = 0;
         //bi = depth_first_search(T,P,F,NN,visited,boundary,&tri_num,L[i][1],r2,i); 
         //bi = breadth_first_search(T,P,F,NN,visited,boundary,&tri_num,L[i][1],r2,i,stack); 
         bi = depth_first_search_nr(T,P,F,NN,visited,boundary,&tri_num,L[i][1],r2,i,stack); 

         //Zero out PCA variables
         mult(mi,0,mi);
         for(j=0;j<9;j++)
            cij[j]=0;
          
         //Loop over neighboring triangles in B(x,r)
         for(j=0;j<tri_num;j++){ //Compute spherical volume invariant
            t = NN[j];
            visited[t] = false;

            tri_vertices(T,P,x,y,z,t,i); //retrieve vertices
            A = tri_normal(x,y,z,nu);
            if(bi && A != 0){ //If B(x,r) does not intersect mesh boundary and triangle non-degenerate
               if(boundary[j]){  //If triangle intersects partial B(x,r), use refinement integration
                  dx = norm_squared(x); dy = norm_squared(y); dz = norm_squared(z);
                  eta = dot(x,nu); 
                  if(MIN3(dx,dy,dz) != 0 && eta != 0){
                     S[i] = S[i] + integrate(x,y,z,dx,dy,dz,A,eta,eps_svi,r,r2,r3,-1,&maxlevel,
                                             &num_subtri,0,0,svi_integrate_approx,svi_integrate_exact,
                                             svi_error_computation);
                     max_num_subtri = MAX(num_subtri,max_num_subtri);
                  }
                  q=0;
                  for(ki=0;ki<3;ki++){
                     mi[ki] = mi[ki] + integrate(x,y,z,dx,dy,dz,A,eta,eps_pca,r,r2,r3,-1,&maxlevel,
                                                 &num_subtri,ki,0,pcami_integrate_approx,
                                                 pcami_integrate_exact,pcami_error_computation);
                     max_num_subtri = MAX(num_subtri,max_num_subtri);
                     for(kj=0;kj<3;kj++){
                        if(kj >= ki){
                           cij[q] = cij[q] + integrate(x,y,z,dx,dy,dz,A,eta,eps_pca,r,r2,r3,-1,&maxlevel,
                                                    &num_subtri,ki,kj,pcacij_integrate_approx,
                                                    pcacij_integrate_exact, 
                                                    pcacij_error_computation);   
                           max_num_subtri = MAX(num_subtri,max_num_subtri);
                        }
                        q++;
                     }
                  }
               }else{ //If triangle is interior to ball, use analytic formula
                  S[i] = S[i] + svi_integrate_exact(x,y,z,r2,r3,0,0);
                  q=0;
                  for(ki=0;ki<3;ki++){
                     mi[ki] = mi[ki] + pcami_integrate_exact(x,y,z,r2,r3,ki,kj);
                     for(kj=0;kj<3;kj++){
                        if(kj >= ki)
                           cij[q] = cij[q] + pcacij_integrate_exact(x,y,z,r2,r3,ki,kj); 
                        q++;
                     }
                  }
               }
            }
         }
         if(bi){
            //Assemble parts to get matrix M
            q=0;
            for(ki=0;ki<3;ki++){
               for(kj=0;kj<3;kj++){
                  if(kj >= ki)
                     M[i][q] = cij[q] - mi[ki]*mi[kj]/S[i]; 
                  if(ki == kj)
                     M[i][q] += r2*S[i]/5;
                  q++;
               }
            }
            q=0;
            for(ki=0;ki<3;ki++){
               for(kj=0;kj<3;kj++){
                  if(kj < ki)
                     M[i][q] = M[i][ki+3*kj];
                  q++;
               }
            }
         }
      }
   }
   printf("\n");
   //printf("\nMax number of refinements = %d\n",maxlevel);
   //printf("Max number of subtriangles = %d\n",max_num_subtri);
}


//Main subroutine to compute spherical volume invaraint
void svi(double *P_ptr, int n, int *T_ptr, int m, bool *ID, double r, double eps, bool prog, double *S, double *Q){

   double dx,dy,dz,eta,A,sgam,x[3], y[3], z[3],  nu[3], **P;
   double r3 = r*r*r;
   double r2 = r*r;
   double vol = 4*PI*r3/3;
   int i, j, t, **L, **F, *NN, *stack, tri_num, num_subtri, max_num_subtri, **T;
   short maxlevel;
   bool *B, *visited, *boundary, bi;

   //Setup 2D array pointers to input
   P = (double**)malloc(n * sizeof(double*));
   T = (int**)malloc(m * sizeof(int*));
   for(i=0;i<n;i++)
      P[i] = P_ptr + 3*i;
   for(i=0;i<m;i++)
      T[i] = T_ptr + 3*i;

   B = vector_bool(n,true);  //Records boundary points of mesh
   NN = vector_int(m,-1);    //Nearest neighbors
   stack = vector_int(m,-1);    //Stack
   visited = vector_bool(m,false); //Records which triangles have alread been visited in depth first search
   boundary = vector_bool(m,false);//Records which triangles intersect boundary of ball B(x,r)

   //Compute face graph
   L = vertex_to_triangle_list(T, n, m);
   F = face_graph(T, L, B, n, m);

   maxlevel = 0;
   max_num_subtri = 0;
   int interval = n/100;

   if(prog){
#ifdef MEX
      mexPrintf("Progress...0%%"); mexEvalString("drawnow;"); 
#else 
      printf("Progress...0%%"); fflush(stdout); 
#endif
   }

   int perc = 0;
   /*Main loop over vertices*/
   for(i=0;i<n;i++){

      S[i] = -1.0; //Default value if not computed later
      Q[i] = -1.0;

      if((i % interval == 1 || i == n-1) && prog){
         if(perc < 10){
#ifdef MEX
            mexPrintf("\b\b%d%%",(100*(i+1))/n); mexEvalString("drawnow;"); 
#else 
            printf("\b\b%d%%",(100*(i+1))/n); fflush(stdout); 
#endif
         }else{
#ifdef MEX
            mexPrintf("\b\b\b%d%%",(100*(i+1))/n); mexEvalString("drawnow;"); 
#else 
            printf("\b\b\b%d%%",(100*(i+1))/n); fflush(stdout); 
#endif
         }
         perc = (100*(i+1))/n;
      }

      if(B[i] && ID[i] && L[i][0] !=0){ //If vertex i is not a boundary vertex, then compute Gamma
         sgam = svigamma(T,P,L,i);
         S[i] = vol*sgam;
         Q[i] = sgam;

         //Depth first search to find neighboring triangles
         tri_num = 0;
         //bi = depth_first_search(T,P,F,NN,visited,boundary,&tri_num,L[i][1],r2,i); 
         //bi = breadth_first_search(T,P,F,NN,visited,boundary,&tri_num,L[i][1],r2,i,stack); 
         bi = depth_first_search_nr(T,P,F,NN,visited,boundary,&tri_num,L[i][1],r2,i,stack); 
           
         //Loop over neighboring triangles in B(x,r)

         for(j=0;j<tri_num;j++){
            t = NN[j];
            visited[t] = false;

            if(bi){ //If B(x,r) does not intersect mesh boundary
               tri_vertices(T,P,x,y,z,t,i); //retrieve vertices
               if(boundary[j]){  //If triangle intersects boundary of B(x,r), use refinement integration
                  dx = norm_squared(x); dy = norm_squared(y); dz = norm_squared(z);
                  A = tri_normal(x,y,z,nu);
                  eta = dot(x,nu); 
                  if(MIN3(dx,dy,dz) != 0 && eta != 0){
                     num_subtri = 0;
                     S[i] = S[i] + integrate(x,y,z,dx,dy,dz,A,eta,eps,r,r2,r3,-1,&maxlevel,&num_subtri,0,0,svi_integrate_approx,svi_integrate_exact,svi_error_computation);
                     max_num_subtri = MAX(num_subtri,max_num_subtri);
                  }
               }else //If triangle is interior to ball, use analytic formula
                  S[i] = S[i] + svi_integrate_exact(x,y,z,r2,r3,0,0);
            }
         }
      }
   }
   printf("\n");
   //printf("\nMax number of refinements = %d\n",maxlevel);
   //printf("Max number of subtriangles = %d\n",max_num_subtri);

}

/*Change of basis used in exact integration*/
static inline double change_of_basis(double *x, double *y, double *nu, double *xtilde, double *qi, double *pi, double *pi1){

   double e1[3], e2[3], t[3], a;

   /*Compute basis*/
   sub(y,x,e1);
   a=1/sqrt(dot(e1,e1)); mult(e1,a,e1);
   cross(e1,nu,e2);
   mult(e2,-1,e2);

   /*Change of basis formulas*/
   sub(x,xtilde,t);
   *pi = dot(t,e1);
   *qi = dot(t,e2);
   sub(y,xtilde,t);
   *pi1 = dot(t,e1);

   return dot(x,e2);
}

/*Computes stopping condition for triangle refinement
 * lmax = squared max side length of triangle
 * eta = x \cdot \nu on triangle
 * r2 = r^2
 * eps = error tolerance
 *
 * Should return True if error is too large and further refinement necessary
 * Otherwise returns false
 */
bool svi_error_computation(double lmax, double eta, double r, double r2, double eps){

   double err = MAX(r-sqrt(lmax),0);
   err = eps*err*err*err*err/r2;
   return (lmax > err || lmax > r2) && eps < 100;
}


/*Approximate integration over triangle for SVI
 * x,y,z = vertices of triangle
 * r3 = r^3
 * theta = angle used in integration 
 */
double svi_integrate_approx(double *x, double *y, double *z, double A, double eta, double r2, double r3, int i, int j){
      double p[3];
      centroid(x,y,z,p);
      return MIN(1 - r3*pow(norm_squared(p),-1.5),0)*A*eta/3;
}

//Returns angle between yx and yz
double angle(double *x, double *y, double *z){
   
   double p[3], q[3], a, b, c;
   sub(x,y,p); sub(z,y,q);
   a = norm(p); b = norm(q); c = dot(p,q);

   return acos(c/(a*b));
}

/*Analytic integration of hypersingular kernel 1/||x||^3$ over triangle xyz
 * x,y,z = vertices of triangle
 * r3 = r^3
 */
double svi_integrate_exact(double *x, double *y, double *z, double r2, double r3, int i, int j){

   double xtilde[3], nu[3];
   double qi, pi, pi1, a, num, den, eta, A, dx, dy, dz, o=0, txy, tyz, tzx, theta;
   bool bxy, byz, bzx;

   dx = norm(x); dy = norm(y); dz = norm(z);
   A = tri_normal(x,y,z,nu);
   eta = dot(x,nu); 

   if(MIN3(dx,dy,dz) != 0 && eta != 0){

      mult(nu,eta,xtilde);

      txy = change_of_basis(x,y,nu,xtilde,&qi,&pi,&pi1);
      num = -2*pi*qi*eta*dx;
      den = qi*qi*dx*dx - pi*pi*eta*eta;
      a = atan2(num,den);
      num = -2*pi1*qi*eta*dy;
      den = qi*qi*dy*dy - pi1*pi1*eta*eta;
      a = a - atan2(num,den);
      
      tyz = change_of_basis(y,z,nu,xtilde,&qi,&pi,&pi1);
      num = -2*pi*qi*eta*dy;
      den = qi*qi*dy*dy - pi*pi*eta*eta;
      a = a + atan2(num,den);
      num = -2*pi1*qi*eta*dz;
      den = qi*qi*dz*dz - pi1*pi1*eta*eta;
      a = a - atan2(num,den);
    
      tzx = change_of_basis(z,x,nu,xtilde,&qi,&pi,&pi1);
      num = -2*pi*qi*eta*dz;
      den = qi*qi*dz*dz - pi*pi*eta*eta;
      a = a + atan2(num,den);
      num = -2*pi1*qi*eta*dx;
      den = qi*qi*dx*dx - pi1*pi1*eta*eta;
      a = a - atan2(num,den);

      theta = 0;
      if(txy <=0 && tyz <= 0 && tzx <= 0){
         bxy = txy < 0; byz = tyz < 0; bzx = tzx < 0;
         if(bxy && byz && bzx){
            theta = 2*PI;
         }else if((bxy && byz) || (byz && bzx) || (bzx && bxy)){
            theta = PI;
         }else if(!bxy && !byz){
            theta = angle(x,y,z);
         }else if(!byz && !bzx){
            theta = angle(y,z,x);
         }else if(!bzx && !bxy){
            theta = angle(z,x,y);
         }
      }
      o = A*eta/3 - (r3/3)*(a/2 + theta*SIGN(eta));
   }

   return o;
}

/*Computes gamma
 * T = triangle list
 * P = vertices
 * L = vertex to triangle list
 * i = vertex
 */
double svigamma(int **T, double **P, int **L, int i){

   double p[3], q[3], nu[3], e1[3], e2[3], e3[3];
   double alpha, d, phi1, phi2, Gam, v1, v2, v3, xx[3], yy[3], zz[3], *x, *y, *z, *temp, ne1, ne3;
   int k = L[i][0]; //Number of adjacent triangles
   int j,t;
   x=xx;y=yy;z=zz;

   nu[0] = 0; nu[1] = 0; nu[2] = 0;
  
   //Compute outward unit normal vector 
   for(j=1;j<=k;j++){
      t = L[i][j];
      tri_vertices(T,P,x,y,z,t,i);
      tri_normal(x,y,z,p);
      add(nu,p,nu);
   }
   e3[0] = -nu[0]; e3[1] = -nu[1]; e3[2] = -nu[2];
   ne3 = norm(e3);
   d=1/ne3; mult(e3,d,e3);

   //Choose e1 orthogonal to e3
   e1[0] = 0; e1[1] = 0; e1[2] = 0;
   if(e3[0] != 0 || e3[1] !=0){
      e1[0] = -e3[1];
      e1[1] = e3[0];
   }else{
      e1[1] = -e3[2];
      e1[2] = e3[1];
   }
   ne1 = norm(e1);
   d=1/ne1; mult(e1,d,e1); 

   //Compute e2 by cross product
   cross(e1,e3,e2);
   mult(e2,-1,e2);
  
   Gam = 0; 
   for(j=1;j<=k;j++){
      t = L[i][j];
      tri_vertices(T,P,x,y,z,t,i);

      //vertex i should be first
      if(T[t][1] == i){
         temp = x; x=y; y=z; z=temp;
      }else if(T[t][2] == i){
         temp = z; z=y; y=x; x=temp;
      }

      //Change basis
      new_coordinates(x,e1,e2,e3);
      new_coordinates(y,e1,e2,e3);
      new_coordinates(z,e1,e2,e3);

      tri_normal(x,y,z,nu);
      sub(y,x,q);
      sub(z,x,p);
      alpha = atan2(nu[1],nu[0]);
      d = sqrt(nu[0]*nu[0] + nu[1]*nu[1]);
      phi1 = atan2(p[1],p[0]);
      phi2 = atan2(q[1],q[0]);

      Gam = Gam + asin(d*sin(phi2-alpha)) - asin(d*sin(phi1-alpha));
   }
   Gam = 0.5 - Gam/(4*PI);
   return Gam;
}

/*Computes stopping condition for triangle refinement
 * lmax = squared max side length of triangle
 * eta = x \cdot \nu on triangle
 * r2 = r^2
 * eps = error tolerance
 *
 * Should return True if error is too large and further refinement necessary
 * Otherwise returns false
 */
bool pcami_error_computation(double lmax, double eta, double r, double r2, double eps){
   return (lmax > r2*eps*eps || lmax > r2) && eps < 100;
}
bool pcacij_error_computation(double lmax, double eta, double r, double r2, double eps){
   return (lmax > r2*eps*eps || lmax > r2) && eps < 100;
}
/*Analytic integration for PCA on local neighborhoods
 * x,y,z = vertices of triangle
 * r3 = r^3
 */
double pcami_integrate_exact(double *x, double *y, double *z, double r2, double r3, int i, int j){
   
   double nu[3];
   double A = tri_normal(x, y, z, nu);
   double eta = dot(x,nu);

   return (A/4)*(eta*(x[i] + y[i] + z[i])/3 - r2*nu[i]);
}

double pcacij_integrate_exact(double *x, double *y, double *z, double r2, double r3, int i, int j){
   
   double nu[3];
   double A = tri_normal(x, y, z, nu);
   double eta = dot(x,nu);
   double ai = A*(x[i] + y[i] + z[i])/3;
   double aj = A*(x[j] + y[j] + z[j])/3;

   double bij = (A/12)*(2*x[i]*x[j] + 2*y[i]*y[j] + 2*z[i]*z[j] + x[i]*y[j] + x[j]*y[i] + x[i]*z[j] + x[j]*z[i] + y[i]*z[j] + y[j]*z[i]);

   return (eta/5)*bij - (r2/10)*(nu[i]*aj + nu[j]*ai);
}

/*Approximate integration for PCA on local neighborhoods
 * x,y,z = vertices of triangle
 * r3 = r^3
 * theta = angle used in integration 
 */
double pcami_integrate_approx(double *x, double *y, double *z, double A, double eta, double r2, double r3, int i, int j){

   double p[3], nu[3];
   tri_normal(x, y, z, nu);
   centroid(x,y,z,p);

   double integrand = 0;
   if(norm_squared(p) <= r2)
      integrand = eta*x[i] - r2*nu[i];

   return integrand*A/4;
}

double pcacij_integrate_approx(double *x, double *y, double *z, double A, double eta, double r2, double r3, int i, int j){

   double p[3], nu[3];
   tri_normal(x, y, z, nu);
   centroid(x,y,z,p);

   double integrand = 0;
   if(norm_squared(p) <= r2)
      integrand = 2*p[i]*p[j]*eta - r2*(p[j]*nu[i] + p[i]*nu[j]);

   return integrand*A/10;
}


