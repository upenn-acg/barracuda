#pragma once

#include "protocol.hpp"
#include "compiler.hpp"
#include <cuda.h>

struct entry_t
{
    int count;
    int ins;
    int padding[2];
};

class DeviceArea
{
public:
    enum { HEADER_SIZE = 64 }; // max(sizeof(PCHeader), alignment(PCRecord));

    BFUNC FINLINE DeviceArea()
    {
    }

    BFUNC FINLINE ~DeviceArea()
    {
    }

    BFUNC FINLINE void init()
    {
        for(int i = 0; i < 64; ++ i)
        {
            entries[i].count = 0;
            entries[i].ins = 0;
        }
    }

#ifdef CUDA
    __device__ FINLINE void inc(int q, bool op)
    {
       atomicAdd(&entries[q].count, 1);
       if(op)
           atomicAdd(&entries[q].ins, 1);
    }
#endif
    BFUNC FINLINE int get_count(int q)
    {
        return entries[q].count;
    }
    BFUNC FINLINE int get_ins(int q)
    {
        return entries[q].ins;
    }


private:
    entry_t entries[64];    
};

