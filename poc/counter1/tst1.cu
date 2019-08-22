#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdint.h>


__global__ void tstfun(int *src, int* dst, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i] = src[i] * m;
}

///  host code
    
int main(int argc, char* argv[]) 
{
    int *dst, *src;
    int *dev_dst, *dev_src;

    int num_blocks = 1;
    int num_threads = 1;

    if(argc > 1)
    {
        num_threads = atoi(argv[1]);
        if(argc > 2)
            num_blocks = atoi(argv[2]);
    }
    int num_total = num_threads * num_blocks;           
    printf("Threads: %i Blocks: %i = %i\n", num_threads, num_blocks, num_total);

    dst = new int[num_total];
    src = new int[num_total];
    for(int i = 0; i < num_total; ++ i)
    {
        dst[i] = 0;
        src[i] = i + 10;
    }

    checkCudaErrors(cudaMalloc(&dev_src, sizeof(int) * num_total));
    checkCudaErrors(cudaMemcpy(dev_src, src, sizeof(int) * num_total, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&dev_dst, sizeof(int) * num_total));
    checkCudaErrors(cudaMemset(dev_dst, 0, sizeof(int) * num_total));

    const int m = 5;
    tstfun<<<num_blocks, num_threads>>>(dev_src, dev_dst, m);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(dst, dev_dst, sizeof(int) * num_total, cudaMemcpyDeviceToHost));

    for(int i = 0; i < num_total; ++ i)
    {
        if(dst[i] != src[i] * m)
        {
            fprintf(stderr, "At index: %i: %i\n", i, dst[i]);
        }
    }
    printf("Success (%i*%i=%i).\n", num_blocks, num_total, num_total);


    return 0;
}

