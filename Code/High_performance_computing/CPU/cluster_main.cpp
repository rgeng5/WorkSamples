#include <omp.h>
#include <stdlib.h>
#include <iostream>
#include "cluster.h"
#include <chrono>
#include <ratio>
#include <cmath>
#include <array>
#include <algorithm>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[]){
    
    const std::size_t n = atoi(argv[1]);
    const std::size_t t = atoi(argv[2]);
    
    omp_set_num_threads(t);
    //printf("t: %d\n", t);

    float * arr = new float[n];
    //float arr[8] = {0, 1, 3, 4, 6, 6, 7, 8};
    float * centers = new float[t];
    float * dists = new float[n];

    //initialize arr with random int numbers [0,n]
    //printf ("arr\n");
    const int RANGE = n;
    for (size_t i = 0; i < n; ++i){ 
            int randA = rand() % (RANGE + 1);
	        arr[i] = static_cast<float>(randA); 
            //printf ("%f ", arr[i]);
    }
    //printf ("\n");

    std::sort(arr, arr+n);

    //initialize centers and dists arrays
    for (size_t j = 1; j < t+1; ++j){ 
	        centers[j-1] = static_cast<float>((2*j-1)*n/(2*t)); 
            dists[j-1] = 0;
            //printf ("centers %f\n", centers[j-1]);
    }

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    start = high_resolution_clock::now();

    //for (size_t j = 0; j < 10; ++j){
        cluster(n, t, arr, centers, dists);
    //}

    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    //find the maximum distance
    float maximum;
    float pos = 0;
    maximum = dists[0];
    for(size_t k = 1; k < t; ++k) {
        //printf ("dists %f\n", dists[k-1]);
        if(dists[k]>maximum) {
            maximum = dists[k];
            pos = k;
        }
    }
    //printf ("dists %f\n", dists[t-1]);

    cout << maximum << "\n";
    cout << pos << "\n";
    cout << duration_sec.count() << "\n";

    return 0;
}