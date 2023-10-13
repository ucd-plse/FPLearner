#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <cmath>


#define DOUBLE 0
#define FLOAT 1
#define INT 2
#define LONG 3
void stopMeasure();
void startMeasure();
void writeData(double* ptr, size_t size, int type, char *name);
void writeData(float* ptr, size_t size, int type, char *name);
void writeData(int* ptr, size_t size, int type, char *name);

void writeData(double* ptr, size_t size, int type, const char *const_name);
void writeData(float* ptr, size_t size, int type, const char *const_name);
void writeData(int* ptr, size_t size, int type, const char *const_name);

void writeDataSize(size_t size,FILE *fptr);   
void writeDataType(int type,FILE *fptr);   

void writeDataValue(double val, int type, FILE *fd);
void writeDataValue(float val, int type, FILE *fd);
void writeDataValue(int val, int type, FILE *fd);

void MP_memcpy(float *dst, double *src, size_t elements);
void MP_memcpy(double *dst, float *src, size_t elements);
void MP_memcpy(float *dst, float *src, size_t elements);
void MP_memcpy(double *dst, double *src, size_t elements);

float *  MP_Malloc(size_t size, float *);
double *  MP_Malloc(size_t size, double *);
float*  MP_Malloc(int size, float *);
double*  MP_Malloc(int size, double *);



int isInteger(char *str){
    if (*str == '\0'){
        return 0;
    }
    for(; *str != '\0'; str++){
        if (*str < 48 || *str > 57){	// digit characters (need to include . if checking for float)
            return 0;
        }
    }
    return 1;
}


#define NUMBER_PAR_PER_BOX 100							
#define NUMBER_THREADS 128

#define DOT(A,B) ((A.x)*(B.x)+(A.y)*(B.y)+(A.z)*(B.z))	// STABLE


typedef struct nei_str
{
    int x, y, z;
    int number;
    long offset;

} nei_str;

typedef struct box_str
{
    int x, y, z;
    int number;
    long offset;
    int nn;
    nei_str nei[26];
} box_str;

double par_alpha;

void initAll(box_str* box_cpu, int boxes1d_arg);

int cur_arg;
int arch_arg;
int cores_arg;
int boxes1d_arg;
long number_boxes;
long box_mem;
size_t space_elem;
size_t space_mem2;


void  kernel_cpu( 
        box_str* box,
        double* rvx,double* rvy,double* rvz,double* rvv,
        double* qv,
        double* fvx,double* fvy,double* fvz,double* fvv)
{

    double alpha;
    double a2;
    int i, j, k, l;
    long first_i;
    double* rAx;
    double* rAy;
    double* rAz;
    double* rAv;
    double *fAx;
    double *fAy;
    double *fAz;
    double *fAv;

    // neighbor box
    int pointer;
    long first_j; 

    double* rBx;
    double* rBy;
    double* rBz;
    double* rBv;

    double* qB;

    // common
    double r2; 
    double u2;
    double fs;
    double vij;
    double fxij;
    double fyij;
    double fzij;
    double dx;
    double dy;
    double dz;


    alpha = par_alpha;
    a2 = 2.0*alpha*alpha;

        for(l=0; l<number_boxes; l=l+1){
            first_i = box[l].offset;
            rAx = &rvx[first_i];
            rAy = &rvy[first_i];
            rAz = &rvz[first_i];
            rAv = &rvv[first_i];

            fAx = &fvx[first_i];
            fAy = &fvy[first_i];
            fAz = &fvz[first_i];
            fAv = &fvv[first_i];
            for (k=0; k<(1+box[l].nn); k++) 
            {

                if(k==0){
                    pointer = l;
                }
                else{
                    pointer = box[l].nei[k-1].number; 
                }

                first_j = box[pointer].offset; 

                rBx = &rvx[first_j];
                rBy = &rvy[first_j];
                rBz = &rvz[first_j];
                rBv = &rvv[first_j];

                qB = &qv[first_j];

                for (i=0; i<NUMBER_PAR_PER_BOX; i=i+1){
                    for (j=0; j<NUMBER_PAR_PER_BOX; j=j+1){
                        double dot = ((rAx[i])*(rBx[j])+(rAy[i])*(rBy[j])+(rAz[i])*(rBz[j]));
                        r2 = rAv[i] + rBv[j] -dot; 
                        u2 = a2*r2;
                        vij= exp(-u2);
                        fs = 2.*vij;
                        dx = rAx[i]  - rBx[j]; 
                        dy = rAy[i]  - rBy[j]; 
                        dz = rAz[i]  - rBz[j]; 
                        fxij=fs*dx;
                        fyij=fs*dy;
                        fzij=fs*dz;

                        // forces
                        fAv[i] +=  qB[j]*vij;
                        fAx[i] +=  qB[j]*fxij;
                        fAy[i] +=  qB[j]*fyij;
                        fAz[i] +=  qB[j]*fzij;
                    } // for j
                } // for i
            } // for k
        } // for l
}

int  main( int argc, char *argv[])
{
    char *outputFile = NULL;
    int i, j, k, l, m, n;
    box_str* box_cpu;
    double* rv_cpux;
    double* rv_cpuy;
    double* rv_cpuz;
    double* rv_cpuv;
    double* qv_cpu;
    double* fv_cpux;
    double* fv_cpuy;
    double* fv_cpuz;
    double* fv_cpuv;
    boxes1d_arg = 1;
    int val = atoi(argv[1]);
    cores_arg=val; 
    boxes1d_arg = atoi(argv[2]);
    outputFile=argv[3];
    par_alpha = 0.5;
    number_boxes = boxes1d_arg * boxes1d_arg * boxes1d_arg;

    space_elem = number_boxes * NUMBER_PAR_PER_BOX;
    space_mem2 = space_elem * sizeof(double);
    box_mem = number_boxes * sizeof(box_str);
    box_cpu = (box_str*)malloc(box_mem);
    // initAll(box_cpu, boxes1d_arg); 
 
    srand(9);

    // input (distances)
    rv_cpux = MP_Malloc(space_elem, rv_cpux);
    rv_cpuy = MP_Malloc(space_elem, rv_cpuy);
    rv_cpuz = MP_Malloc(space_elem, rv_cpuz);
    rv_cpuv = MP_Malloc(space_elem, rv_cpuv);
    for(i=0; i<space_elem; i=i+1){
        rv_cpuv[i] = (rand()%10 + 1) / 10.0;
        rv_cpux[i] = (rand()%10 + 1) / 10.0;
        rv_cpuy[i] = (rand()%10 + 1) / 10.0;
        rv_cpuz[i] = (rand()%10 + 1) / 10.0;
    }
    // input (charge)
    qv_cpu = MP_Malloc(space_elem,qv_cpu);
    for(i=0; i<space_elem; i=i+1){
        qv_cpu[i] = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
    }

    fv_cpux = MP_Malloc(space_elem, fv_cpux);
    fv_cpuy = MP_Malloc(space_elem, fv_cpuy);
    fv_cpuz = MP_Malloc(space_elem, fv_cpuz);
    fv_cpuv = MP_Malloc(space_elem, fv_cpuv);

    for(i=0; i<space_elem; i=i+1){
        fv_cpux[i] = 0;
        fv_cpuy[i] = 0;
        fv_cpuz[i] = 0;
        fv_cpuv[i] = 0;
    }


    startMeasure();
    kernel_cpu(	
            box_cpu,
            rv_cpux,rv_cpuy,rv_cpuz,rv_cpuv,
            qv_cpu,
            fv_cpux,fv_cpuy,fv_cpuz,fv_cpuv
            );
    stopMeasure();
    if ( outputFile ){
        size_t size = space_elem *4;
        int type = DOUBLE;
        FILE *fptr = fopen(outputFile, "wb");
        writeDataSize(size,fptr);   
        writeDataType(type,fptr);   
        for(i=0; i<space_elem; i=i+1){
            writeDataValue(fv_cpuv[i], DOUBLE, fptr);
            writeDataValue(fv_cpux[i], DOUBLE, fptr);
            writeDataValue(fv_cpuy[i], DOUBLE, fptr);
            writeDataValue(fv_cpuz[i], DOUBLE, fptr);
        }
        fclose(fptr);
    }
    free(rv_cpuy);
    free(rv_cpux);
    free(rv_cpuz);
    free(rv_cpuv);

    free(qv_cpu);

    free(fv_cpuz);
    free(fv_cpuy);
    free(fv_cpux);
    free(fv_cpuv);
    free(box_cpu);
    return 0;
}
