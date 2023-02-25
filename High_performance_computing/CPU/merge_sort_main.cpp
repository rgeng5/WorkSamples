#include <omp.h>
#include <stdlib.h>
#include <iostream>
#include "msort.h"
#include <chrono>
#include <ratio>
#include <cmath>
using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[]){
    
    std::size_t n = atoi(argv[1]);
    int t = atoi(argv[2]);
    std::size_t ts = atoi(argv[3]);
    
    omp_set_num_threads(t);
    //printf("t: %d\n", t);

    int *A = new int[n];
    //initialize arrays with random int numbers [-1000,1000]
    const int RANGE = 2000;
    for (size_t i = 0; i < n; ++i){ 
            int randA = rand() % (RANGE + 1);
	        A[i] = randA - 1000; 
            //printf ("In %d\n", A[i]);
    }

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    start = high_resolution_clock::now();

    msort(A, n, ts);

    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    printf ("%d\n", A[0]);
    printf ("%d\n", A[n-1]);
    cout << duration_sec.count() << "\n";

    return 0;
}