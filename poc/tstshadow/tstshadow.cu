#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

static const int NUM_BLOCKS = 64;
static const int NUM_THREADS = 1024;
static const int NUM_LOOPS = 5;
static const int TOTAL = NUM_BLOCKS * NUM_THREADS * NUM_LOOPS;

__device__ unsigned int counter = 0;

__global__ void tst_shadow(unsigned int* numbers)
{
    for(int i = 0; i < NUM_LOOPS; ++ i)
    {
        unsigned int val = atomicAdd((unsigned int*)&counter, 1);
        if(val >= TOTAL)
            val = TOTAL;
        atomicAdd(&numbers[val], 1);
    }
}

int main(int argc, char* argv[]) {

    // Copy input data to device.
    unsigned int* numbers;
    checkCudaErrors(cudaMalloc(&numbers, (1 + TOTAL) * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(numbers, 0 ,(1 + TOTAL) * sizeof(unsigned int)));

    // Launch the kernel.
    tst_shadow<<<NUM_BLOCKS, NUM_THREADS>>>(numbers);

    // Copy output data to host.
    checkCudaErrors(cudaDeviceSynchronize());
    unsigned int* lnumbers = (unsigned int*)malloc((1 + TOTAL) * sizeof(unsigned int));
    checkCudaErrors(cudaMemcpy(lnumbers, numbers, (1 + TOTAL) * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < TOTAL; ++i) {
        if(lnumbers[i] != 1)
        {
            fprintf(stderr, "Error at %i\n", i);
        }
    }
    if(lnumbers[TOTAL] != 0)
    {
        fprintf(stderr, "Overflows: %i\n", lnumbers[TOTAL]);
    }

    cudaDeviceReset();
    return 0;
}

