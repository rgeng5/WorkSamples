#include <stdlib.h>
#include <iostream>
#include "optimize.h"
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

    vec *v = new vec(n);
    //float arr[8] = {0, 1, 3, 4, 6, 6, 7, 8};

    v->len=n;
    v->data = new data_t[n];
    data_t * dest = new data_t[1];

    //initialize arr with random int numbers [0,n]
    //printf ("data \n");
    const int RANGE = n;
    for (size_t i = 0; i < n; ++i){ 
            data_t randA = rand() % (RANGE + 1);
	        v->data[i] = static_cast<data_t>(randA); 
            //printf ("%d ", v->data[i]);
    }
    //printf ("\n");

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    start = high_resolution_clock::now();
    for (size_t j = 0; j < 10; ++j){
        optimize1(v, dest);
    }
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << dest[0] << "//from optimize1\n";
    cout << duration_sec.count()/10 << "//from optimize1\n";
    //cout << duration_sec.count()/10 << "\n";

    start = high_resolution_clock::now();
    for (size_t j = 0; j < 10; ++j){
        optimize2(v, dest);
    }
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << dest[0] << "//from optimize2\n";
    cout << duration_sec.count()/10 << "//from optimize2\n";
    //cout << duration_sec.count()/10 << "\n";

    start = high_resolution_clock::now();
    for (size_t j = 0; j < 10; ++j){
        optimize3(v, dest);
    }
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << dest[0] << "//from optimize3\n";
    cout << duration_sec.count()/10 << "//from optimize3\n";
    //cout << duration_sec.count()/10 << "\n";

    start = high_resolution_clock::now();
    for (size_t j = 0; j < 10; ++j){
        optimize4(v, dest);
    }
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << dest[0] << "//from optimize4\n";
    cout << duration_sec.count()/10 << "//from optimize4\n";
    //cout << duration_sec.count()/10 << "\n";

    start = high_resolution_clock::now();
    for (size_t j = 0; j < 10; ++j){
        optimize5(v, dest);
    }
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << dest[0] << "//from optimize5\n";
    cout << duration_sec.count()/10 << "//from optimize5\n";
    //cout << duration_sec.count()/10 << "\n";

    return 0;
}