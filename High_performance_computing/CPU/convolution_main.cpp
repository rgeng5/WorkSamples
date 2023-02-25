#include <omp.h>
#include <stdlib.h>
#include <iostream>
#include "convolution.h"
#include <chrono>
#include <ratio>
#include <cmath>
using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[]){
    
    size_t n = atoi(argv[1]);
    int t = atoi(argv[2]);
    size_t m = 3;

    omp_set_num_threads(t);
    //printf("t: %d\n", t);

    //int n_threads=omp_get_max_threads();
    //printf("Number of threads: %d\n", n_threads);

    float *A = new float[n*n];
    float *B = new float[n*n];;
    float *C = new float[n*n];;

    //initialize image with random float numbers [-10.0,10.0]
    //printf ("A\n");
    const int RANGE = 1000;
    for (size_t i = 0; i < n; ++i){ //row
        for (size_t j = 0; j < n; ++j){ //column
            int randA = rand() % (RANGE + 1);
	        A[i*n+j] = randA / 50.0f-10.0f; //row-major filling
            //printf ("%f ", A[i*n+j]);
        }
        //printf ("\n");
    }

    //initialize mask with random float numbers [-1.0,1.0]
    //printf ("B\n");
    for (size_t i = 0; i < m; ++i){ //row
        for (size_t j = 0; j < m; ++j){ //column
            int randB = rand() % (RANGE + 1);
            B[i*m+j] = randB / 500.0f-1.0f; //row-major filling
            //printf ("%f ", B[i*m+j]);
        }
        //printf ("\n");
    }

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    start = high_resolution_clock::now();

    convolve(A, C, n, B, m);

    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    printf ("%f\n", C[0]);
    printf ("%f\n", C[n*n-1]);
    cout << duration_sec.count() << "\n";

    return 0;
}