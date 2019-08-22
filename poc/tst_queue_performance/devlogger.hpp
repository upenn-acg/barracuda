#pragma once

__device__ void __store_op(uint64_t tid, const void* address, unsigned char op)
{
    unsigned int active = __ballot(1);
    unsigned int myidx = threadIdx.x % WARP_SIZE; 
    unsigned int ldridx = __ffs(active) - 1;
    int pos = 0;
    const int tbid = (blockIdx.x + pcqnum - 1) % pcqnum;
    const int size = pcqbuffer_size;
    PCHeader* pcheader = PCHEADER(pcqheader, tbid, pcqbuffer_size);
    PCRecord* pcstart = PCSTART(pcheader);
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
    record->address[myidx] = (slimptr_t)(uintptr_t)address;

    if(ldridx == myidx)
    {
        record->tid = tid;
        record->active = active;
        record->op = op;
        while(atomicCAS(&pcheader->read_head, pos, pos + 1) != pos)
            __threadfence_system();
    }
}

