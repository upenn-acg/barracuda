#define CUDA
#include <cuda_runtime.h>
#include "devarea.hpp"
#include "protocol.hpp"
#include "debug.h"
#include "ptx_stub.h"

__device__ DeviceArea DEVICE_AREA_GLOBAL_NAME;

extern "C" __global__ void INIT_FUNCTION_NAME()
{
    DEVICE_AREA_GLOBAL_NAME.init();
    __threadfence_system();
}

extern "C" __device__ __attribute__((noinline)) void STORE_OP_FUNCTION_NAME(bool op)
{
    DEVICE_AREA_GLOBAL_NAME.inc(blockIdx.x % 64, op);
}


extern "C" __global__ void force_function_linking(int count)
{
    STORE_OP_FUNCTION_NAME(true);
    STORE_OP_FUNCTION_NAME(false);
}


int main (int argc, char* argv[])
{
    uint64_t* x;
    if(0 != cudaMalloc(&x, sizeof(uint64_t)))
    {
        printf("Failed cudaMalloc().\n");
        return 1;
    }
    void* buf;
    int buf_size = 64 * 1000;
    if(0 != cudaMalloc(&buf, buf_size))
    {
        printf("Failed cudaMalloc().\n");
        return 1;
    }
    DeviceArea devarea;

    INIT_FUNCTION_NAME<<<1,1>>>();
    int sync = cudaDeviceSynchronize();
    if(sync != 0)
    {
        printf("%s failed, err=%i\n", NAMEOF_INIT_FUNCTION_NAME, sync);
        return 2;
    }

    force_function_linking<<<1,1>>>(10);
    sync = cudaDeviceSynchronize();
    if(sync != 0)
    {
        printf("Link function failed, err=%i\n", sync);
        return 2;
    }
    int v = cudaMemcpyFromSymbol(&devarea, DEVICE_AREA_GLOBAL_NAME, sizeof(DeviceArea), 0, cudaMemcpyDeviceToHost);
    if(v!=0)
    {
        printf("Run function failed, err=%i\n", v);
    }
    printf("Value:% i\n", devarea.get_count(1));
    
    printf("PTX stubs tested OK!\n");
    return 0;
}
