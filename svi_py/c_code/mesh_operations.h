/* mesh_operations.h - 
 *
 *  Basic subroutines for computing with triangulated meshes
 *
 *  Author: Jeff Calder, 2018.
 */
 
#include "stdbool.h"
#define tri_vertices(T,P,x,y,z,t,i) sub(P[T[t][0]],P[i],x);sub(P[T[t][1]],P[i],y);sub(P[T[t][2]],P[i],z)

double tri_dist(double *x, double *y, double *z);
int** vertex_to_triangle_list(int **T, int n, int m);
int** face_graph(int **T, int **L, bool *B, int n, int m);
bool depth_first_search(int **T, double **P, int **F, int *NN, bool *v, bool *b, int *num, int ind, double r2, int i);
bool breadth_first_search(int **T, double **P, int **F, int *NN, bool *v, bool *b, int *num, int ind, double r2, int i, int *stack);
bool depth_first_search_nr(int **T, double **P, int **F, int *NN, bool *v, bool *b, int *num, int ind, double r2, int i, int *stack);
double integrate(double *x, double *y, double *z, double dx, double dy, double dz, double A, double eta, double eps, double r, double r2, double r3, short level, short *maxlevel, int *num_subtri, int i, int j, double (*integrate_approx) (double *, double *, double *, double, double, double, double, int, int), double (*integrate_exact) (double *, double *, double *, double, double, int, int), bool (*error_computation) (double, double, double, double, double));
