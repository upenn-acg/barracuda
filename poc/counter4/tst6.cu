#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdint.h>
#include "ptx_stub.h"
#include "devarea.hpp"
#include "protocol.hpp"
#include "debug.h"

static const int NUM_BLOCKS = 1;
static const int NUM_THREADS = 1;
static const int NUM_TOTAL = NUM_BLOCKS * NUM_THREADS;

__device__ DeviceArea DEVICE_AREA_GLOBAL_NAME;

#define PTRTO(x) (uint64_t)(uintptr_t)(void*)x

extern "C" __global__ void INIT_FUNCTION_NAME(DeviceArea device_area)
{
    memcpy(&DEVICE_AREA_GLOBAL_NAME, &device_area, sizeof(device_area));

    for(int i = 0; i < DEVICE_AREA_GLOBAL_NAME.numq(); ++ i)
    {
        PCHeader* pcheader = DEVICE_AREA_GLOBAL_NAME.header(i);
        pcheader->read_head = 0;
        pcheader->write_head = 0;
        pcheader->tail = 0;
        DEBUGONLY(printf("PC Buffer %i initialized, at %p, size is: %i\n", i, pcheader, DEVICE_AREA_GLOBAL_NAME.qbuf_size());)
    }

    __threadfence_system();
}

static __device__ FINLINE unsigned int __ptx_laneid()
{
    unsigned int value;
    asm volatile("mov.u32 %0, %%laneid;"  : "=r"(value));
    return value;
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


extern "C" __device__ __attribute__((noinline)) void STORE_OP_FUNCTION_NAME(const uint64_t tid, const uint64_t address, const uint32_t op, const uint32_t loc_id)
{
    const unsigned int active = __ballot(1);
    const unsigned int myidx = __ptx_laneid();
    const unsigned int ldridx = __ffs(active) - 1;
    const int qid = blockIdx.x % DEVICE_AREA_GLOBAL_NAME.numq(); // XXX: change to SM number
    const int size = DEVICE_AREA_GLOBAL_NAME.qbuf_size();
    int pos = 0;
    PCHeader* pcheader = DEVICE_AREA_GLOBAL_NAME.header(qid);
    PCRecord* pcstart = DeviceArea::start(pcheader);
    PCRecord* record = NULL;

    if(ldridx == myidx)
    {
        volatile unsigned int* tail = (volatile unsigned int*)&pcheader->tail;
        pos = atomicAdd(&pcheader->write_head, 1);
        while((pos - *tail) >= size)
            __threadfence_system();
    }
    pos = __shfl(pos, ldridx);
    record = pcstart + (pos % size);
    DEBUGONLY(printf("bi=%i ti=%i myidx=%i ldridx=%i pos=%i record=%p ra=%p\n", blockIdx.x, threadIdx.x, myidx, ldridx, pos, record, &record->address[myidx]);)
    record->address[myidx] = address;

    if(ldridx == myidx)
    {
        record->tid = tid;
        record->active = active;
        record->op_state= (__isGlobal((void*)address) ? GLOBAL_FLAG : 0) | op;
        record->loc_id = loc_id;
        while(atomicCAS(&pcheader->read_head, pos, pos + 1) != pos)
            __threadfence();
    }
    __threadfence_system();
}


extern "C" __global__ void force_function_linking(uint64_t* tid)
{
    *tid = GETTID_FUNCTION_NAME(0x1234);
    STORE_OP_FUNCTION_NAME(*tid, NULL, OP_SYNCTHREADS, 1);
}


int bla()
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
    DeviceArea devarea(buf, buf_size, 2);

    INIT_FUNCTION_NAME<<<1,1>>>(devarea);
    int sync = cudaDeviceSynchronize();
    if(sync != 0)
    {
        printf("%s failed, err=%i\n", NAMEOF_INIT_FUNCTION_NAME, sync);
        return 2;
    }

    force_function_linking<<<1,1>>>(x);
    sync = cudaDeviceSynchronize();
    if(sync != 0)
    {
        printf("Link function failed, err=%i\n", sync);
        return 2;
    }
    printf("PTX stubs tested OK!\n");
    return 0;
}

__global__ void tstfun(volatile unsigned int* data, int repeats) 
{
    uint64_t tid = GETTID_FUNCTION_NAME(0);
    int id = ((blockDim.x * blockIdx.x) + threadIdx.x);
    data[id] = 0;
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 2, 2);
    int x1 = 0, x2 = 0, x3 = 0;

    x1 = data[id];
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 1, 3);
    atomicAdd((unsigned int*)&data[id], 1);
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 3, 4);
    __syncthreads();
    STORE_OP_FUNCTION_NAME(tid, 0, 4, 3);
    __threadfence();
    atomicAdd((unsigned int*)&data[id], 1);
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 54, 5);
    __syncthreads();
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 4, 6);
    __threadfence();
    atomicAdd((unsigned int*)&data[id], 1);
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 54, 7);
    __syncthreads();
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 4, 8);
    __threadfence();
    data[id] = 0;
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 34, 8);
    __syncthreads();
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 4, 9);
    data[id] = 0;
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 2, 10);
    __threadfence();
    __syncthreads();
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 4, 11);
    __threadfence();
    x2 = data[id];
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 1, 12);
    __syncthreads();
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 4, 13);
    __threadfence();
    x3 = data[id];
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 1, 14);
    __syncthreads();
    __syncthreads();
    __threadfence();
    atomicExch((unsigned int*)&data[id], 1);
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 38, 15);
    __syncthreads();
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 4, 16);
    __threadfence();
    atomicExch((unsigned int*)&data[id], 1);
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 38, 17);
    __syncthreads();
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 4, 18);
    __threadfence();
    atomicCAS((unsigned int*)&data[id], 0, 1);
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 3, 19);
    __syncthreads();
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 4, 20);
    __threadfence();
    atomicCAS((unsigned int*)&data[id], 0, 1);
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 3, 21);

    if(x1 == 0x112233 || x2 == 0x23929303 || x3 == 0x84938493)
        printf("BLA!");
    STORE_OP_FUNCTION_NAME(tid, PTRTO(&data[id]), 5, 22);
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

    if(argc == 20920)
        bla();

    return 0;
}

