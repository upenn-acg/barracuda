#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdint.h>

static const int NUM_BLOCKS = 2;
static const int NUM_THREADS = 2;
static const int NUM_TOTAL = NUM_BLOCKS * NUM_THREADS;

static const int NUM_VALUES = 30;

__global__ void tstfun(volatile unsigned int* data) 
{
    int id = ((blockDim.x * blockIdx.x) + threadIdx.x);
    for(int i = 0; i < NUM_VALUES; ++i )
    {
        data[id] += 4;
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
    
    tstfun<<<NUM_BLOCKS, NUM_THREADS>>>(dev_data);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(host_data, dev_data, sizeof(unsigned int) * NUM_TOTAL, cudaMemcpyDeviceToHost));
    for(int i = 0; i < NUM_TOTAL; ++ i)
    {
        if(host_data[i] != NUM_VALUES*4)
        {
            fprintf(stderr, "At index: %i: %i\n", i, host_data[i]);
        }
    }
    printf("Success.\n");

    return 0;
}

