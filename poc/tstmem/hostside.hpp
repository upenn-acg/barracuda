#pragma once

#include <map>
#include "env.hpp"

class HostSide
{
public:
    static inline HostSide& instance()
    {
        static HostSide inst;
        return inst;
    }

    static inline void process_records(PCRecord* records, int count)
    {
        auto& inst = instance();
        for(;count > 0; -- count, ++ records)
        {
            inst.process_record(records);
        }
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
    HostSide()
    {
        _gpu_only = Env::gpu_only();
    }

    ~HostSide()
    {
    }

    void process_record(PCRecord* record)
    {
        if(_gpu_only)
            return;
        char buffer[1024];
        int osl = record->op_state;
        int op = osl & 0xFF;
        int global = (osl >> 8) & 0xFF; 
        int bufpos = sprintf(buffer, "TRACE,%lx,%08x,%i,%i,%i", record->tid, record->active, op, global, record->loc_id);
        for(int j = 0; j < WARP_SIZE; ++ j)
        {
             
            if(record->active & (1 << j))
                bufpos += sprintf(&buffer[bufpos], ",%lx", record->address[j]);
        }
        puts(buffer);
    }

    bool _gpu_only;
};
