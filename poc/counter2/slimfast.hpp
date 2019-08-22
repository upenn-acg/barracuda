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
        for(;count > 0; -- count, ++ records)
        {
            bool first = true;
            for(int j = 0; j < WARP_SIZE; ++ j)
            {
                 
                if(records->active & (1 << j))
                {
                    printf("TRACE,%i,%lx,%i,%08x,%i,%i,%lx\n", first ? 1 : 0,
                            records->tid, j, records->active, records->op, records->predicated, records->address[j]);
                    first = false;
                }
            }
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
