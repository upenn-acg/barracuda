#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdint.h>

static const int NUM_BLOCKS = 1;
static const int NUM_THREADS = 1;
static const int NUM_TOTAL = NUM_BLOCKS * NUM_THREADS;

__global__ void tstfun(volatile unsigned int* data, int repeats) 
{
    int id = ((blockDim.x * blockIdx.x) + threadIdx.x);
    data[id] = 0;
    int x1 = 0, x2 = 0, x3 = 0;

    x1 = data[id];
    atomicAdd((unsigned int*)&data[id], 1);
    __syncthreads();
    __threadfence();
    atomicAdd((unsigned int*)&data[id], 1);
    __syncthreads();
    __threadfence();
    atomicAdd((unsigned int*)&data[id], 1);
    __syncthreads();
    __threadfence();
    data[id] = 0;
    __syncthreads();
    data[id] = 0;
    __threadfence();
    __syncthreads();
    __threadfence();
    x2 = data[id];
    __syncthreads();
    __threadfence();
    x3 = data[id];
    __syncthreads();
    __syncthreads();
    __threadfence();
    atomicExch((unsigned int*)&data[id], 1);
    __syncthreads();
    __threadfence();
    atomicExch((unsigned int*)&data[id], 1);
    __syncthreads();
    __threadfence();
    atomicCAS((unsigned int*)&data[id], 0, 1);
    __syncthreads();
    __threadfence();
    atomicCAS((unsigned int*)&data[id], 0, 1);

    if(x1 == 0x112233 || x2 == 0x23929303 || x3 == 0x84938493)
        printf("BLA!");
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

