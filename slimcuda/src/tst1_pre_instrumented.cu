#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdint.h>

#define NO_STUB_MAIN
#include "ptx_stub.cu"

__device__ __attribute__((noinline)) void store(uint64_t tid, int* where, int what)
{
    *where = what;
    STORE_OP_FUNCTION_NAME(tid, (uintptr_t)where, OP_STORE, 3);
    
    printf("Storing at %p from %p\n",  where, tid);
}

__device__ int zglobal[32];

__global__ void tstfun(uint32_t sid, int *src, int* dst, const int m)
{
    uint64_t tid = GETTID_FUNCTION_NAME(sid);
    STORE_OP_FUNCTION_NAME(tid, 0, OP_START_KERNEL, 4);

    __shared__ int zshared[32];
    int p;
    int* pp = &p;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int val = src[i];
    STORE_OP_FUNCTION_NAME(tid, (uintptr_t)&src[i], OP_LOAD, 5);
    printf("Tst1: &src[%i]=%p &dst[%i]=%p m=%i\n", i, &src[i], i, &dst[i], m);
//    printf("Tst1: &src[%i]=%p &dst[%i]=%p m=%i zhared[i]=%p zglobal[i]=%p, pp=%p\n", i, &src[i], i, &dst[i], m, &zshared[i], &zglobal[i], pp);
    store(tid, &dst[i], val * m);
    store(tid, &zshared[i], val * m);
    store(tid, &zglobal[i], val * m);
    store(tid, pp, val  * m);
    dst[i] = val * m;
    STORE_OP_FUNCTION_NAME(tid, (uintptr_t)&dst[i], OP_STORE, 6);
    zshared[i] = val * m;
    STORE_OP_FUNCTION_NAME(tid, (uintptr_t)&zshared[i], OP_STORE, 7);
    zglobal[i] = val * m;
    STORE_OP_FUNCTION_NAME(tid, (uintptr_t)&zglobal[i], OP_STORE, 8);
    
    STORE_OP_FUNCTION_NAME(tid, 0, OP_END_KERNEL, 9);
}

///  host code
    
int main(int argc, char* argv[]) 
{
    int *dst, *src;
    int *dev_dst, *dev_src;

    int num_blocks = 2;
    int num_threads = 2;

    if(argc > 1)
    {
        num_threads = atoi(argv[1]);
        if(argc > 2)
            num_blocks = atoi(argv[2]);
    }
    int num_total = num_threads * num_blocks;           
    printf("Tst1: threads=%i blocks:=%i total=%i\n", num_threads, num_blocks, num_total);

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
    tstfun<<<num_blocks, num_threads>>>(0, dev_src, dev_dst, m);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(dst, dev_dst, sizeof(int) * num_total, cudaMemcpyDeviceToHost));

    for(int i = 0; i < num_total; ++ i)
    {
        if(dst[i] != src[i] * m)
        {
            fprintf(stderr, "Tst1: Error At index: %i: %i\n", i, dst[i]);
            return -1;
        }
    }
    printf("Tst1: Success (%i*%i=%i).\n", num_blocks, num_total, num_total);
    printf("Tst1: no hazards expected.\n");


    stub_force_linking();

    return 0;
}

