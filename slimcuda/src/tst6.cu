#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdint.h>

static const int NUM_BLOCKS = 1;
static const int NUM_THREADS = 33;
static const int NUM_TOTAL = NUM_BLOCKS * NUM_THREADS;

__device__ void tstfun1(int a1, int a2);
__device__ void tstfun2(int a1, int a2);

__global__ void tstfun(int a1, int a2)
{
    tstfun1(a1, a2);
}


__device__ void tstfun1(int a1, int a2)
{
    if(a1 == 0x12345)
       tstfun2(a1, a2 - 1);
    printf("TF1\n");
}

__device__ void tstfun2(int a1, int a2)
{
    if(a2 == 0x12346)
            tstfun1(a1 -1, a2);
    printf("TF2\n");
}

///  host code
    
int main(int argc, char* argv[]) 
{
    // Launch the kernel.
    unsigned int* dev_data;
    checkCudaErrors(cudaMalloc(&dev_data, sizeof(unsigned int) * NUM_TOTAL));
    checkCudaErrors(cudaMemset(dev_data, 0, sizeof(unsigned int) * NUM_TOTAL));
    unsigned int* host_data = (unsigned int*)malloc(sizeof(unsigned int) * NUM_TOTAL);
    
    tstfun<<<NUM_BLOCKS, NUM_THREADS>>>(random(), random());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(host_data, dev_data, sizeof(unsigned int) * NUM_TOTAL, cudaMemcpyDeviceToHost));
    printf("Success.\n");

    return 0;
}

