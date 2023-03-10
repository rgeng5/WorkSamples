#include <cstdlib>
#include <iostream>
#include "cluster.h"

void cluster(const size_t n, const size_t t, const float *arr,
             const float *centers, float *dists) {
#pragma omp parallel num_threads(t) 
  {
    unsigned int tid = omp_get_thread_num();
#pragma omp for reduction(+:dists[:t])
    for (size_t i = 0; i < n; i++) {
      dists[tid] += abs(arr[i] - centers[tid]);
      //printf ("i %zu tid %i dist %f \n", i, tid, dists[tid]);
    }
  }
}