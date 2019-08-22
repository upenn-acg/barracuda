#pragma once

#include <map>

class SlimFast
{
public:
    static inline SlimFast& instance()
    {
        static SlimFast inst;
        return inst;
    }

    static inline void process_records(PCRecord* records, int count)
    {
        PCRecord tmp;
        for(;count > 0; -- count, ++ records)
        {
            memcpy(&tmp, records, sizeof(*records));
        }
    }

    static void print()
    {
    }

    // XXX: add host side operations
   
    static void on_stream_created(unsigned short stream_id)
    {
    }

    static void on_stream_destroyed(unsigned short stream_id)
    {
    }

    static void on_kernel_launched(unsigned short stream_id, dim3 grid_dim)
    {
    }

    static void on_kernel_done(unsigned short stream_id)
    {
    }


private:
    SlimFast()
    {
    }

    ~SlimFast()
    {
    }

};
