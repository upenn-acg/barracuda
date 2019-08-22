#pragma once

#include "CV.hpp"
#include "threadblocks.hpp"
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
        instance().process_records_imp(records, count);
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
    typedef void (SlimFast::*handler_t)(tid_t tid, void* address);
    handler_t _handlers[PROTO_OP_MAX];

    SlimFast()
    {
        _handlers[OP_UNKNOWN] = &SlimFast::handle_unknown;
        _handlers[OP_READ] = &SlimFast::handle_read;
        _handlers[OP_WRITE] = &SlimFast::handle_write;
        _handlers[OP_ATOMIC_WRITE] = &SlimFast::handle_atomic_write;
        _handlers[OP_ATOMIC_READWRITE] = &SlimFast::handle_atomic_readwrite;
        _handlers[OP_SYNCTHREADS] = &SlimFast::handle_syncthreads;
    }

    ~SlimFast()
    {
    }


    void process_records_imp(PCRecord* record, int count)
    {
        for(int i = 0; i < count; ++ i, ++ record)
        {
            printf("Processing %i:%i:%i active=0x%x op=%i\n", 
                    record->stream_id, record->block_id, record->warp_id, 
                    record->active, record->op);
           
            handler_t handler = _handlers[record->op]; 
            for(int j = 0; j < WARP_SIZE; ++ j)
            {
                if(record->active & (1 << j))
                {
                    (this->*handler)(get_tid(record, j), record->address[j]);
                }
            }
    
        }
    }
          
    static inline tid_t get_tid(PCRecord* r, int idx)
    {
        return CV::get_tid_t(r->stream_id, r->block_id, r->warp_id * WARP_SIZE + idx);
    }

    void handle_unknown(tid_t tid, void* addresS)
    {
        fprintf(stderr, "Unknown operation!\n");
        exit(-1);
    }
    void handle_read(tid_t tid, void* address)
    {
        printf("Read from 0x%x: %p\n", tid, address);
    }
    void handle_write(tid_t tid, void* address)
    {
        printf("Write to 0x%x: %p\n", tid, address);
    }
    void handle_atomic_write(tid_t tid, void* address)
    {
        printf("Atomic write to 0x%x: %p\n", tid, address);
    }
    void handle_atomic_readwrite(tid_t tid, void* address)
    {
        printf("Atomic readwrite to/from 0x%x: %p\n", tid, address);
    }
    void handle_syncthreads(tid_t tid, void* address)
    {
        printf("Syncthreads()\n", tid);
    }

    std::map<uint64_t, unsigned int> _thread_lc;
};
