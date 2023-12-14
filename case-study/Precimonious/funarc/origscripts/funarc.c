#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>


#define ITERS 10

double fun( double x ) {
  int k, n = 5;
  double t1 = 1.0L;
  double d1 = 1.0L;

  t1 = x;

  for( k = 1; k <= n; k++ )
  {
    d1 = 2.0 * d1;
    t1 = t1 + sin (d1 * x) / d1;
  }

  return t1;
}

int main() {

  /****** BEGIN PRECIMONIOUOS PREAMBLE ******/
  // variables for logging/checking
  double epsilon = 1.0e-4;
  int l;

  // variables for timing measurement
  clock_t start_time, end_time;  
  double cpu_time_used;

  start_time = clock();
  
  // dummy calls to alternative functions
  sqrtf(0);
  acosf(0);
  sinf(0);
  /****** END PRECIMONIOUOS PREAMBLE ******/
  
  int i, j, k, n = 1000000;
  double h;
  double t1;
  double t2;
  double dppi;
  double s1;
  
  for (l = 0; l < ITERS; l++) {
    t1 = -1.0;
    dppi = acos(t1);
    s1 = 0.0;
    t1 = 0.0;
    h = dppi / n;

    for( i = 1; i <= n; i++ ) {
      t2 = fun (i * h);
      s1 = s1 + sqrt (h*h + (t2 - t1)*(t2 - t1));
      t1 = t2;
    }
  }


  /***** BEGIN PRECIMONIOUS ACCURACY CHECKING AND LOGGING ******/

  // record in file verified, s1, err
  double zeta_verify_value = 5.7957763224130E+00;
  double err;
  bool verified;
  err = fabs(s1 - zeta_verify_value) / zeta_verify_value;
  if (err <= epsilon) {
      verified = true;
      printf(" VERIFICATION SUCCESSFUL\n");
      printf(" Zeta is    %20.13E\n", s1);
      printf(" Error is   %20.13E\n", err);
  } else {
      verified = false;
      printf(" VERIFICATION FAILED\n");
      printf(" Zeta                %20.13E\n", s1);
      printf(" The correct zeta is %20.13E\n", zeta_verify_value);
  }
  FILE *fp = fopen("./log.txt", "w");
  fputs(verified ? "true\n" : "false\n", fp);
  fprintf(fp, "%20.13E\n", s1);
  fprintf(fp, "%20.13E\n", err);

  // record time
  end_time = clock(); 
  cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
  FILE *fp_t = fopen("./time.txt", "w");
  fprintf(fp_t, "%f\n", cpu_time_used);


  /****** END PRECIMONIOUS ACCURACY CHECKING AND LOGGING ******/

  return 0;
}

