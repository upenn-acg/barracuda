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
            printf("Processing %i records.\n", count) ;
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
    typedef uint64_t tid_t;
    typedef uint64_t blk_t;
    typedef void (SlimFast::*handler_t)(tid_t tid, const void* address);
    handler_t _handlers[PROTO_OP_MAX];

    SlimFast()
    {
        _handlers[OP_UNKNOWN] = &SlimFast::handle_unknown;
        _handlers[OP_LOAD] = &SlimFast::handle_read;
        _handlers[OP_STORE] = &SlimFast::handle_write;
        _handlers[OP_ATOMIC] = &SlimFast::handle_atomic;
        _handlers[OP_SYNCTHREADS] = &SlimFast::handle_syncthreads;
        _handlers[OP_FENCE_CTA] = &SlimFast::handle_fence_cta;
        _handlers[OP_FENCE_GL] = &SlimFast::handle_fence_gl;
        _handlers[OP_FENCE_SYS] = &SlimFast::handle_fence_sys;
    }

    ~SlimFast()
    {
    }


    void process_records_imp(PCRecord* record, int count)
    {
        for(int i = 0; i < count; ++ i, ++ record)
        {
           
            handler_t handler = _handlers[record->op]; 
            for(int j = 0; j < WARP_SIZE; ++ j)
            {
                if(record->active & (1 << j))
                {
                    printf("Processing %lx active=0x%x op=%i loc=%i WID:%i adddr=%p\n", 
                        record->tid, record->active, record->op, record->loc_id, j, record->address[j]);
                    (this->*handler)(get_tid(record, j), (void*)(uintptr_t)record->address[j]);
                }
            }
    
        }
    }
          
    static inline tid_t get_tid(PCRecord* r, int idx)
    {
        return r->tid + idx;
    }

    void handle_unknown(tid_t tid, const void* addresS)
    {
        fprintf(stderr, "Unknown operation!\n");
        exit(-1);
    }
    void handle_read(tid_t tid, const void* address)
    {
        DEBUGONLY(printf("Read from 0x%x: %p\n", tid, address);)
    }
    void handle_write(tid_t tid, const void* address)
    {
        DEBUGONLY(printf("Write to 0x%x: %p\n", tid, address);)
    }
    void handle_atomic(tid_t tid, const void* address)
    {
        DEBUGONLY(printf("Atomic write to 0x%x: %p\n", tid, address);)
    }
    void handle_syncthreads(tid_t tid, const void* address)
    {
        DEBUGONLY(printf("Syncthreads()\n", tid);)
    }
    void handle_fence_cta(tid_t tid, const void* address)
    {
        DEBUGONLY(printf("fence_cta()\n", tid);)
    }
    void handle_fence_gl(tid_t tid, const void* address)
    {
        DEBUGONLY(printf("fence_gl()\n", tid);)
    }
    void handle_fence_sys(tid_t tid, const void* address)
    {
        DEBUGONLY(printf("fence_sys()\n", tid);)
    }
    static inline blk_t get_blk_t(unsigned short stream_id, unsigned short block_id)
    {
        blk_t val = stream_id;
        val = val << 16;
        val |= block_id;
        return val;
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


    std::map<uint64_t, unsigned int> _thread_lc;
};
