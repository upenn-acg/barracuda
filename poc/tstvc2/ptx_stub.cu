#include <cuda_runtime.h>
#include "devarea.hpp"
#include "protocol.hpp"
#include "debug.h"
#include "ptx_stub.h"

__device__ DeviceArea DEVICE_AREA_GLOBAL_NAME;

extern "C" __global__ void INIT_FUNCTION_NAME(DeviceArea device_area)
{
    memcpy(&DEVICE_AREA_GLOBAL_NAME, &device_area, sizeof(device_area));

    for(int i = 0; i < DEVICE_AREA_GLOBAL_NAME.numq(); ++ i)
    {
        PCHeader* pcheader = DEVICE_AREA_GLOBAL_NAME.header(i);
        pcheader->read_head = 0;
        pcheader->write_head = 0;
        pcheader->tail = 0;
        SLIM_VERBOSEONLY(printf("PC Buffer %i initialized, at %p, size is: %i\n", i, pcheader, DEVICE_AREA_GLOBAL_NAME.qbuf_size());)
    }

    __threadfence_system();
}

static __device__ FINLINE unsigned int __ptx_laneid()
{
	return threadIdx.x % (1 << WARP_SHIFT);
}

extern "C" __device__ __attribute__((noinline)) uint64_t GETTID_FUNCTION_NAME(int streamid)
{
	uint64_t block = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	uint64_t thread = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    return build_tid(streamid, block, thread);
}


static __device__ inline uint64_t get_normalized_ptr(uint64_t address)
{
    if(address == 0)
        return 0;
    uint64_t mask = 4;
    uint64_t outaddr = 0;
    asm volatile ("{ \n\t"
                  "    .reg .pred pg;\n\t"
                  "    .reg .pred ps; \n\t"
                  "    isspacep.global pg, %2; \n\t"
                  "    isspacep.shared ps, %2; \n\t"
                  "    @pg cvta.to.global.u64 %0, %2; \n\t"                     
                  "    @pg mov.u64 %1, 1; \n\t"
                  "    @ps cvta.to.shared.u64 %0, %2; \n\t"                     
                  "    @ps mov.u64 %1, 2; \n\t"
                  "} \n\t" : "=l"(outaddr) , "=l"(mask) : "l"(address));
    return outaddr | (mask << 62);
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
    SLIM_VERBOSEONLY(printf("bi=%i ti=%i myidx=%i ldridx=%i pos=%i record=%p ra=%p\n", blockIdx.x, threadIdx.x, myidx, ldridx, pos, record, &record->address[myidx]);)
    record->address[myidx] = get_normalized_ptr(address);

    if(ldridx == myidx)
    {
        record->wid = tid;
        record->active = active;
        record->op = op;
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
