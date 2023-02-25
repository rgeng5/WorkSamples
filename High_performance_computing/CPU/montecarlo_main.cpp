#include <omp.h>
#include <stdlib.h>
#include <iostream>
#include "montecarlo.h"
#include <chrono>
#include <ratio>
#include <cmath>
#include <random>
#include <vector>
using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[]){
    
    size_t n = atoi(argv[1]);
    int t = atoi(argv[2]);

    omp_set_num_threads(t);
    //printf("t: %d\n", t);

    //int n_threads=omp_get_max_threads();
    //printf("Number of threads: %d\n", n_threads);

    float *x = new float[n];
    float *y = new float[n];


    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    const float minval = -1.0, maxval = 1.0;
    std::uniform_real_distribution<float> dist(minval, maxval);
    //std::uniform_real_distribution<float> dist(minval,std::nextafter(maxval, std::numeric_limits<float>::max()));
    

    for (size_t i = 0; i < n; ++i){
        x[i] = dist(generator);
        y[i] = dist(generator);
        //printf ("%f ", x[i]);
    }
    const float r = 1.0;
    int count = 0;

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    start = high_resolution_clock::now();

    for (size_t j = 0; j < 10; ++j){
        count = montecarlo(n, x, y, r);
        //printf("count 2 %d\n", count);
    }

    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    float pi;
    pi = static_cast<float>(count)/static_cast<float>(n)*4;
    printf ("%f\n", pi);
    cout << duration_sec.count()/10 << "\n";

    return 0;
}