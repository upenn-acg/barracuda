#include <cuda_runtime.h>
#include "devarea.hpp"
#include "protocol.hpp"
#include "debug.h"

__device__ DeviceArea __device_area;

extern "C" __global__ void INIT_FUNCTION_NAME(DeviceArea device_area)
{
    __device_area = device_area;

    for(int i = 0; i < __device_area.numq(); ++ i)
    {
        PCHeader* pcheader = __device_area.header(i);
        pcheader->read_head = 0;
        pcheader->write_head = 0;
        pcheader->tail = 0;
        DEBUGONLY(printf("PC Buffer %i initialized, at %p, size is: %i\n", i, pcheader, __device_area.qbuf_size());)
    }

    __threadfence_system();
}

extern "C" __device__ __attribute__((noinline)) uint64_t GETTID_FUNCTION_NAME(int streamid)
{
    return BUILD_ADDRESS(streamid,
                (blockIdx.x + blockIdx.y * gridDim.x
                    + gridDim.x * gridDim.y * blockIdx.z),
                    ((threadIdx.z * (blockDim.x * blockDim.y))
                    + (threadIdx.y * blockDim.x)
                    + threadIdx.x));
}

extern "C" __global__ void stub_gettid_kernel(uint64_t* tid)
{
    *tid = GETTID_FUNCTION_NAME(0x1234);
}


int main (int argc, char* argv[])
{
    DeviceArea devarea;
    INIT_FUNCTION_NAME<<<1,1>>>(devarea);
    stub_gettid_kernel<<<1,1>>>(NULL);
    return 0;
}
