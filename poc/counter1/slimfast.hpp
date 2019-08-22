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
//            printf("Processing %i records.\n", count) ;
        instance().process_records_imp(records, count);
    }

    static void print()
    {
        auto& inst = instance();
        for(int i = 0; i < PROTO_OP_MAX; ++ i)
            printf("STAT %i %i\n", i, inst._counters[i]);
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
    typedef uint64_t tid_t;
    typedef uint64_t blk_t;
    typedef void (SlimFast::*handler_t)(tid_t tid, const void* address);

    SlimFast()
    {
    }

    ~SlimFast()
    {
    }

    
    void process_records_imp(PCRecord* record, int count)
    {
        for(int i = 0; i < count; ++ i, ++ record)
        {
            __sync_fetch_and_add(&_counters[record->op], 1);
        }
    }
          
    static inline tid_t get_tid(PCRecord* r, int idx)
    {
        return r->tid + idx;
    }

    static inline tid_t get_tid_t(unsigned short stream_id, unsigned short block_id, unsigned int thread_id)
    {
        tid_t val = stream_id;
        val = val << 16;
        val |= block_id;
        val = val << 32;
        val |= thread_id;
        return val;
    }

    int _counters[PROTO_OP_MAX];
};
