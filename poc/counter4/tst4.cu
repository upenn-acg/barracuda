#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdint.h>

static const int NUM_BLOCKS = 2;
static const int NUM_THREADS = 2;
static const int NUM_TOTAL = NUM_BLOCKS * NUM_THREADS;

__global__ void tstfun(volatile unsigned int* data, int repeats) 
{
    int id = ((blockDim.x * blockIdx.x) + threadIdx.x);
    for(int i = 0; i < repeats; ++i )
    {
        data[id] += 4;
        if(i % 2 == 0)
            data[id] += 1;
    }
}

///  host code
    
int main(int argc, char* argv[]) 
{
    // Launch the kernel.
    unsigned int* dev_data;
    checkCudaErrors(cudaMalloc(&dev_data, sizeof(unsigned int) * NUM_TOTAL));
    checkCudaErrors(cudaMemset(dev_data, 0, sizeof(unsigned int) * NUM_TOTAL));
    unsigned int* host_data = (unsigned int*)malloc(sizeof(unsigned int) * NUM_TOTAL);
    
    tstfun<<<NUM_BLOCKS, NUM_THREADS>>>(dev_data, 5);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(host_data, dev_data, sizeof(unsigned int) * NUM_TOTAL, cudaMemcpyDeviceToHost));
    printf("Success.\n");

    return 0;
}

