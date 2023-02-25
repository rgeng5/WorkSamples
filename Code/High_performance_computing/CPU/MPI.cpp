#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <ratio>
#include <cmath>
#include <random>
#include <vector>
#include "mpi.h"
#include <stdio.h>
#include <string.h>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[]) {
    size_t n = atoi(argv[1]);
    float *x = new float[n];
    float *y = new float[n];
    double t0;
    double t1;
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    const float minval = -1.0, maxval = 1.0;
    std::uniform_real_distribution<float> dist(minval, maxval);
    
    //fill 2 buffer arrays with random float numbers from (-1.0,1.0)
    for (size_t i = 0; i < n; ++i){
        x[i] = dist(generator);
        y[i] = dist(generator);
    }

    int         my_rank;       /* rank of process      */
    int         p;             /* number of processes  */
    int         source;        /* rank of sender       */
    int         dest;          /* rank of receiver     */
    int         tag = 0;       /* tag for messages     */

    MPI_Status  status;        /* return status for receive  */
   
    MPI_Init(&argc, &argv); // Start up MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // Find out process rank   
    MPI_Comm_size(MPI_COMM_WORLD, &p); // Find out number of processes

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    if (my_rank == 0) {
        dest = 1;
        source = 1;
        /* start timing t0 */
        start = high_resolution_clock::now();
        MPI_Send(x, n, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
        MPI_Recv(y, n, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);
        /* end timing t0 */
        end = high_resolution_clock::now();
        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        t0 = duration_sec.count();
        MPI_Send(&t0, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
    } 
    else if (my_rank == 1){ 
        dest = 0;
        source = 0;
        /* start timing t1 */
        start = high_resolution_clock::now();
        MPI_Recv(x, n, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);
        MPI_Send(y, n, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
        /* end timing t1 */
        end = high_resolution_clock::now();
        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        t1 = duration_sec.count();
        MPI_Recv(&t0, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);

        cout << t0+t1 << "\n";
    }
   
    MPI_Finalize(); // Shut down MPI
    return 0;
} 
