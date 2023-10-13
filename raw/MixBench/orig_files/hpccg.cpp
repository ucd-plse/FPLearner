#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cctype>
#include <cmath>

#define DOUBLE 0
#define FLOAT 1
#define INT 2
#define LONG 3
using std::fabs;

void writeData(double* ptr, size_t size, int type,  char *_name);
void writeData(float* ptr, size_t size, int type,  char *_name);
void writeData(size_t* ptr, size_t size, int type,  char *_name);



void stopMeasure();
void startMeasure();

 int max_external = 100000;
 int max_num_messages = 500;
 int max_num_neighbors = max_num_messages;

char   *HPC_title;
int HPC_start_row;
int HPC_stop_row;
int HPC_total_nrow;
long long HPC_total_nnz;
int HPC_local_nrow;
int HPC_local_ncol;  // Must be defined in make_local_matrix
int HPC_local_nnz;
int  * HPC_nnz_in_row;
double ** HPC_ptr_to_vals_in_row;
int ** HPC_ptr_to_inds_in_row;
double ** HPC_ptr_to_diags;
double *HPC_list_of_vals;   //needed for cleaning up memory
int *HPC_list_of_inds;      //needed for cleaning up memory


int ddot ( int n,  double *  x,  double *  y, 
        double *  result, double& time_allreduce)
{  
    double local_result = 0.0;
    int i;
    if (y==x)
        for ( i=0; i<n; i++) local_result = local_result+ x[i]*x[i];
    else
        for (i=0; i<n; i++) local_result = local_result + x[i]*y[i];

    *result = local_result;
    return(0);
}
//waxpby.cpp
int waxpby ( int n,  double alpha,  double *  x, 
         double beta,  double *  y, 
        double *  w)
{  
    int i;
    if (alpha==1.0) {
        for (i=0; i<n; i++) w[i] = x[i] + beta * y[i];
    }
    else if(beta==1.0) {
        for (i=0; i<n; i++) w[i] = alpha * x[i] + y[i];
    }
    else {
        for (i=0; i<n; i++) w[i] = alpha * x[i] + beta * y[i];
    }
    return(0);
}

int compute_residual( int n,  double *  v1, 
         double *  v2, double *  residual)
{
    double local_residual = 0.0;
    int i;
    for (i=0; i<n; i++) {
        double diff = fabs(v1[i] - v2[i]);
        if (diff > local_residual) local_residual = diff;
    }
    *residual = local_residual;
    return(0);
}

//HPC_sparsemv.cpp

int HPC_sparsemv( double *  x, double *  y)
{
     int nrow = HPC_local_nrow;
    int i;
    int j;

    for (i=0; i< nrow; i++)
    {
        double sum = 0.0;
         double *  cur_vals = ( double * ) HPC_ptr_to_vals_in_row[i];

         int    *  cur_inds = 
            ( int    * ) HPC_ptr_to_inds_in_row[i];

         int cur_nnz = ( int) HPC_nnz_in_row[i];

        for (j=0; j< cur_nnz; j++)
            sum += cur_vals[j]*x[cur_inds[j]];
        y[i] = sum;
    }
    return(0);
}


//generate_matrix.cpp
//
void generate_matrix(int nx, int ny, int nz, double **x, double **b, double **xexact)
{
    int size = 1; // Serial case (not using MPI)
    int rank = 0;
    HPC_title = 0;
    bool use_7pt_stencil = false;
    int local_nrow = nx*ny*nz; // This is the size of our subblock
    int local_nnz = 27*local_nrow; // Approximately 27 nonzeros per row (except for boundary nodes)
    int total_nrow = local_nrow*size; // Total number of grid points in mesh
    long long total_nnz = 27* (long long) total_nrow; // Approximately 27 nonzeros per row (except for boundary nodes)
    int start_row = local_nrow*rank; // Each processor gets a section of a chimney stack domain
    int stop_row = start_row+local_nrow-1;
    // Allocate arrays that are of length local_nrow
    HPC_nnz_in_row = new int[local_nrow];
    HPC_ptr_to_vals_in_row = new double*[local_nrow];
    HPC_ptr_to_inds_in_row = new int   *[local_nrow];
    HPC_ptr_to_diags       = new double*[local_nrow];

    *x = new double[local_nrow];
    *b = new double[local_nrow];
    *xexact = new double[local_nrow];

    HPC_list_of_vals = new double[local_nnz];
    HPC_list_of_inds = new int   [local_nnz];

    double * curvalptr = HPC_list_of_vals;
    int * curindptr = HPC_list_of_inds;

    long long nnzglobal = 0;
    int iz;
    int iy;
    int ix;
    for (iz=0; iz<nz; iz++) {
        for (iy=0; iy<ny; iy++) {
            for (ix=0; ix<nx; ix++) {
                int curlocalrow = iz*nx*ny+iy*nx+ix;
                int currow = start_row+iz*nx*ny+iy*nx+ix;
                int nnzrow = 0;
                HPC_ptr_to_vals_in_row[curlocalrow] = curvalptr;
                HPC_ptr_to_inds_in_row[curlocalrow] = curindptr;
                int sx, sy, sz;
                for (sz=-1; sz<=1; sz++) {
                    for (sy=-1; sy<=1; sy++) {
                        for (sx=-1; sx<=1; sx++) {
                            int curcol = currow+sz*nx*ny+sy*nx+sx;
                            if ((ix+sx>=0) && (ix+sx<nx) && (iy+sy>=0) && (iy+sy<ny) && (curcol>=0 && curcol<total_nrow)) {
                                if (!use_7pt_stencil || (sz*sz+sy*sy+sx*sx<=1)) { // This logic will skip over point that are not part of a 7-pt stencil
                                    if (curcol==currow) {
                                        HPC_ptr_to_diags[curlocalrow] = curvalptr;
                                        *curvalptr++ = 27.0;
                                    }
                                    else {
                                        *curvalptr++ = -1.0;
                                    }
                                    *curindptr++ = curcol;
                                    nnzrow++;
                                } 
                            }
                        } // end sx loop
                    } // end sy loop
                } // end sz loop
                HPC_nnz_in_row[curlocalrow] = nnzrow;
                nnzglobal += nnzrow;
                (*x)[curlocalrow] = 0.0;
                (*b)[curlocalrow] = 27.0 - ((double) (nnzrow-1));
                (*xexact)[curlocalrow] = 1.0;
            } // end ix loop
        } // end iy loop
    } // end iz loop  
    HPC_start_row = start_row ; 
    HPC_stop_row = stop_row;
    HPC_total_nrow = total_nrow;
    HPC_total_nnz = total_nnz;
    HPC_local_nrow = local_nrow;
    HPC_local_ncol = local_nrow;
    HPC_local_nnz = local_nnz;
    return;
}

int HPCCG( double *  b, double *  x,
         int max_iter,  double tolerance, int &niters, double & normr,
        double * times)
{
    double t0 = 0.0;
    double t1 = 0.0;
    double t2 = 0.0;
    double t3 = 0.0;
    double t4 = 0.0;
    int nrow = HPC_local_nrow;
    int ncol = HPC_local_ncol;
    double * r = new double [nrow];
    double * p = new double [ncol]; // In parallel case, A is rectangular
    double * Ap = new double [nrow];

    normr = 0.0;
    double rtrans = 0.0;
    double oldrtrans = 0.0;

    int rank = 0; // Serial case (not using MPI)

    int print_freq = max_iter/10; 
    if (print_freq>50) print_freq=50;
    if (print_freq<1)  print_freq=1;

    // p is of length ncols, copy x to p for sparse MV operation
    waxpby(nrow, 1.0, x, 0.0, x, p); 
    HPC_sparsemv( p, Ap); 
    waxpby(nrow, 1.0, b, -1.0, Ap, r);
    ddot(nrow, r, r, &rtrans, t4);
    normr = sqrt(rtrans);
    int k;
    for(k=1; k<max_iter && normr > tolerance; k++ )
    {
        if (k == 1)
        {
            waxpby(nrow, 1.0, r, 0.0, r, p);
        }
        else
        {
            oldrtrans = rtrans;
            ddot (nrow, r, r, &rtrans, t4); 
            double beta = rtrans/oldrtrans;
            waxpby (nrow, 1.0, r, beta, p, p); 
        }
        normr = sqrt(rtrans);
        if (rank==0 && (k%print_freq == 0 || k+1 == max_iter))
            std::cout << "Iteration = "<< k << "   Residual = "<< normr << std::endl;
        HPC_sparsemv(p, Ap); // 2*nnz ops
        double alpha = 0.0;
        ddot(nrow, p, Ap, &alpha, t4);
        alpha = rtrans/alpha;
        waxpby(nrow, 1.0, x, alpha, p, x);// 2*nrow ops
        waxpby(nrow, 1.0, r, -alpha, Ap, r);  
        niters = k;
    }

    delete [] p;
    delete [] Ap;
    delete [] r;
    return(0);
}

int main(int argc, char *argv[])
{
    int niters = 0;
    double normr = 0.0;
    int max_iter = 150;
    double tolerance = 0.0; 
    double *x;
    double *b;
    double *xexact;
    double norm;
    double d;
    int ierr = 0;
    int i, j;
    int ione = 1;
    double times[7];
    double t6 = 0.0;
    int nx,ny,nz;
    int size = 1; // Serial case (not using MPI)
    int rank = 0; 
    char *quality;
    nx = atoi(argv[1]);
    ny = atoi(argv[2]);
    nz = atoi(argv[3]);
    quality = argv[4];
    generate_matrix(nx, ny, nz, &x, &b, &xexact);
    startMeasure();
    ierr = HPCCG(  b, x, max_iter, tolerance, niters, normr, times);
    stopMeasure();
    size_t val = nx*ny*nz;
    writeData(x,val, DOUBLE,quality);   
    return 0 ;
}
