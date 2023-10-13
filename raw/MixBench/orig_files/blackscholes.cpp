#include <stdio.h>
#include <stdlib.h>
#include <cmath>


#define DOUBLE 0
#define FLOAT 1
#define INT 2


void writeQualityFile(char *fileName, void *ptr, int type, size_t numElements);
void readData(FILE *fd, int **ptr,    size_t* numElements);
void readData(FILE *fd, float **ptr,  size_t* numElements);
void readData(FILE *fd, double **ptr, size_t* numElements);

void stopMeasure();
void startMeasure();

void writeData(double* ptr, size_t size, int type, char *name);
void writeData(float* ptr, size_t size, int type, char *name);
void writeData(int* ptr, size_t size, int type, char *name);

void writeData(double* ptr, size_t size, int type, const char *const_name);
void writeData(float* ptr, size_t size, int type, const char *const_name);
void writeData(int* ptr, size_t size, int type, const char *const_name);




double *prices;
size_t numOptions;

int    * otype;
double * sptprice;
double * strike;
double * rate;
double * volatility;
double * otime;
int numError = 0;
int nThreads;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Cumulative Normal Distribution Function
// See Hull, Section 11.8, P.243-244
//
//CONSTANTS
double inv_sqrt_2xPI=0.39894228040143270286f;
double zero = 0.0;
double half = 0.5;
double const1=0.2316419;
double one=1.0;
double const2=0.319381530;
double const3=0.356563782;
double const4=1.781477937;
double const5=1.821255978;
double const6=1.330274429;

double CNDF ( double InputX ) 
{
    int sign;

    double OutputX;
    double xInput;
    double xNPrimeofX;
    double expValues;
    double xK2;
    double xK2_2;
    double xK2_3;
    double xK2_4;
    double xK2_5;
    double xLocal;
    double xLocal_1;
    double xLocal_2;
    double xLocal_3;

    // Check for negative value of InputX
    if (InputX < zero) {
        InputX = -InputX;
        sign = 1;
    } else 
        sign = 0;

    xInput = InputX;

    // Compute NPrimeX term common to both four & six decimal accuracy calcs
    expValues = exp(-half * InputX * InputX);
    xNPrimeofX = expValues;
    xNPrimeofX = xNPrimeofX * inv_sqrt_2xPI;

    xK2 = const1* xInput;
    xK2 = one + xK2;
    xK2 = one / xK2;
    xK2_2 = xK2 * xK2;
    xK2_3 = xK2_2 * xK2;
    xK2_4 = xK2_3 * xK2;
    xK2_5 = xK2_4 * xK2;

    xLocal_1 = xK2 * const2;
    xLocal_2 = xK2_2 * (-const3);
    xLocal_3 = xK2_3 * const4;
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_4 * (-const5);
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_5 * const6;
    xLocal_2 = xLocal_2 + xLocal_3;

    xLocal_1 = xLocal_2 + xLocal_1;
    xLocal   = xLocal_1 * xNPrimeofX;
    xLocal   = one - xLocal;

    OutputX  = xLocal;

    if (sign) {
        OutputX = one - OutputX;
    }

    return OutputX;
} 

double BlkSchlsEqEuroNoDiv( double sptprice,
        double strike, double rate, double volatility,
        double time, int otype, float timet )
{
    double OptionPrice;

    // local private working variables for the calculation
    double xStockPrice;
    double xStrikePrice;
    double xRiskFreeRate;
    double xVolatility;
    double xTime;
    double xSqrtTime;

    double logValues;
    double xLogTerm;
    double xD1; 
    double xD2;
    double xPowerTerm;
    double xDen;
    double d1;
    double d2;
    double FutureValueX;
    double NofXd1;
    double NofXd2;
    double NegNofXd1;
    double NegNofXd2;    

    xStockPrice = sptprice;
    xStrikePrice = strike;
    xRiskFreeRate = rate;
    xVolatility = volatility;

    xTime = time;
    xSqrtTime = sqrt(xTime);

    logValues = log( sptprice / strike );

    xLogTerm = logValues;


    xPowerTerm = xVolatility * xVolatility;
    xPowerTerm = xPowerTerm * half;

    xD1 = xRiskFreeRate + xPowerTerm;
    xD1 = xD1 * xTime;
    xD1 = xD1 + xLogTerm;

    xDen = xVolatility * xSqrtTime;
    xD1 = xD1 / xDen;
    xD2 = xD1 -  xDen;

    d1 = xD1;
    d2 = xD2;

    NofXd1 = CNDF( d1 );
    NofXd2 = CNDF( d2 );

    FutureValueX = strike * ( exp( -(rate)*(time) ) );        
    if (otype == 0) {            
        OptionPrice = (sptprice * NofXd1) - (FutureValueX * NofXd2);
    } else { 
        NegNofXd1 = (one - NofXd1);
        NegNofXd2 = (one - NofXd2);
        OptionPrice = (FutureValueX * NegNofXd2) - (sptprice * NegNofXd1);
    }

    return OptionPrice;
}


int bs_thread(void *tid_ptr) {
    int i, j,k;
    double price;
    double priceDelta;
    int tid = *(int *)tid_ptr;
    int start = tid * (numOptions / nThreads);
    int end = start + (numOptions / nThreads);
    for (k =  0; k < 100; k++)
    for (i=start; i<end; i++) {
        /* Calling main function to calculate option value based on 
         * Black & Scholes's equation.
         */
        price = BlkSchlsEqEuroNoDiv( sptprice[i], strike[i],
                rate[i], volatility[i], otime[i], 
                otype[i], 0);
        prices[i] = price;

#ifdef ERR_CHK
        priceDelta = data[i].DGrefval - price;
        if( fabs(priceDelta) >= 1e-4 ){
            printf("Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
                    i, price, data[i].DGrefval, priceDelta);
            numError ++;
        }
#endif
    }

    return 0;
}

int main (int argc, char **argv)
{
    FILE *file;
    int i;
    int loopnum;
    int rv;
    struct timeval start, end; 

    // start timer. 


    printf("PARSEC Benchmark Suite\n");
    fflush(NULL);
    if (argc != 4)
    {
        printf("Usage:\n\t%s <nthreads> <inputFile> <outputFile>\n", argv[0]);
        exit(1);
    }
    nThreads = atoi(argv[1]);
    char *inputFile = argv[2];
    char *outputFile = argv[3];

    //Read input data from file
    file = fopen(inputFile, "rb");
    if(file == NULL) {
        printf("ERROR: Unable to open file `%s'.\n", inputFile);
        exit(1);
    }

    if(nThreads != 1) {
        printf("Error: <nthreads> must be 1 (serial version)\n");
        exit(1);
    }

#define PAD 256
#define LINESIZE 64
    readData(file,&otype, &numOptions);  
    readData(file,&sptprice, &numOptions);  
    readData(file,&strike, &numOptions);  
    readData(file,&rate, &numOptions);  
    readData(file,&volatility, &numOptions);  
    readData(file,&otime, &numOptions);  
    prices = (double*) malloc(sizeof(double)*numOptions);
    printf("Size of data: %d\n", numOptions );


    int tid=0;

    startMeasure();
    bs_thread(&tid);
    stopMeasure();

    //Write prices to output file
    writeData(prices, numOptions, DOUBLE, outputFile);
    free(sptprice);
    free(strike);
    free(rate);
    free(volatility);
    free(otime);
    free(otype);
    free(prices);

    return 0;
}

