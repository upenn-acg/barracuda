#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdint.h>

static const int NUM_BLOCKS = 1;
static const int NUM_THREADS = 33;
static const int NUM_TOTAL = NUM_BLOCKS * NUM_THREADS;

__device__ void incx(volatile unsigned int* data, int idx1, int idx2)
{
    if(blockIdx.x != 0)
        return;
    if(threadIdx.x == idx1)
        data[0] = 1;
    else if(threadIdx.x == idx2)
    {
        data[1] = 1;
        data[1] = 2;
        data[1] = 3;
        data[1] = 4;
        data[1] = 5;
    }
}

__global__ void tstfun(volatile unsigned int* data, int repeats, bool sync) 
{
    incx(data, 0, 32); 
    if(sync)
        __syncthreads();
    incx(data, 32, 0); 
    if(sync)
        __syncthreads();
    incx(data, 0, 32); 
    if(sync)
        __syncthreads();
    incx(data, 32, 0); 
    if(sync)
        __syncthreads();
}

///  host code
    
int main(int argc, char* argv[]) 
{
    bool sync = false;
    if(argc > 1)
    {
        sync = strcasecmp(argv[1], "sync") == 0;
        if(argc != 2 || !sync)
        {
            fprintf(stderr, "Syntax error: %s [sync]\n", argv[0]);
            return 1;
        }
    }
    printf("tst5: sync=%i\n", sync);

    // Launch the kernel.
    unsigned int* dev_data;
    checkCudaErrors(cudaMalloc(&dev_data, sizeof(unsigned int) * NUM_TOTAL));
    checkCudaErrors(cudaMemset(dev_data, 0, sizeof(unsigned int) * NUM_TOTAL));
    unsigned int* host_data = (unsigned int*)malloc(sizeof(unsigned int) * NUM_TOTAL);
    
    tstfun<<<NUM_BLOCKS, NUM_THREADS>>>(dev_data, 5, sync);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(host_data, dev_data, sizeof(unsigned int) * NUM_TOTAL, cudaMemcpyDeviceToHost));
    printf("Done.\n");

    return 0;
}

