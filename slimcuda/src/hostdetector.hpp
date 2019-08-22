#pragma once

#include <unordered_map>
#include <tbb/concurrent_unordered_map.h>
#include <mutex>
#include <pthread.h>
#include <stdio.h>
#include "warp_cv.hpp"
#include "mem_cv.hpp"
#include "debug.h"
#include "sync_vcs.hpp"
#include "protocol.hpp"
#include "memloc.hpp"
#include "thread_map.hpp"
#include <atomic>

class HostDetector
{
public:
    enum {
        LOCALS_CAPACITY = 65536
    };

    enum {
        PT_ALIGNMENT_SHIFT = 20
    };
    enum { PT_ALIGNMENT = (1 << PT_ALIGNMENT_SHIFT) };
    enum { CUDA_ARG_SIZE = 8 };

    static HostDetector INSTANCE;

    void on_init(int numq)
    {
        _print_hazards = !Env::supress_prints();
        _shared_pt = nullptr;
        _break_count = 0;
        _kernel_id_counter = 0;
        pthread_cond_init(&_kernel_end_cond, nullptr);
        pthread_mutex_init(&_kernel_end_mutex, nullptr);

        _thread_count = numq;
        if(0 != pthread_barrier_init(&_barrier, nullptr, _thread_count + 1))
        {
            fprintf(stderr, "Failed creating pthread barrier.\n");
            exit(-1);
        }

        size_t free = 0, total = 0;
        VERIFY(ORIG(cudaMemGetInfo)(&free, &total));
        uintptr_t ptr;
        VERIFY(ORIG(cudaMalloc)((void**)&ptr, free));
        VERIFY(ORIG(cudaFree)((void*)ptr));
        uintptr_t used = total - free;
    
        _min_global_address = ((ptr > used) ? (ptr - used) : 0) & ~(PT_ALIGNMENT - 1);
        _max_global_address = (ptr + total + PT_ALIGNMENT - 1) & ~(PT_ALIGNMENT - 1);
        //printf("Globals at %lx - %lx\n", _min_global_address, _max_global_address);        
        
        _global_pt_entries = (_max_global_address - _min_global_address) >> PT_ALIGNMENT_SHIFT;
        _global_pt = new memloc*[_global_pt_entries];
        memset(_global_pt, 0, sizeof(memloc*) * _global_pt_entries);
    }

    static void disable()
    {
        INSTANCE._disabled = true;
    }

    static inline void check_for_pause()
    {
/*        if(INSTANCE._should_break)
        {
            int new_brk = __sync_add_and_fetch(&INSTANCE._break_count, 1);
            SLIM_VERBOSEONLY(printf("check_for_pause(): breaking, new_brk=%i\n", new_brk));
            pthread_barrier_wait(&INSTANCE._barrier);
        }
*/
    }

    static inline void process_records(PCRecord* records, int count)
    {
        INSTANCE.process_records_imp(records, count);
    }

    // XXX: add host side operations
    static void on_cuda_malloc(void* what, int size)
    {
    }

    static void on_stream_created(unsigned short stream_id)
    {
        SLIM_DPRINTF(VL_QUIET, "Not implemented yet!");
        SLIM_ASSERT(false);
    }

    static void on_stream_destroyed(unsigned short stream_id)
    {
        SLIM_DPRINTF(VL_QUIET, "Not implemented yet!");
        SLIM_ASSERT(false);
    }

    static inline int on_kernel_launched(unsigned short stream_id, dim3 grid_dim, dim3 block_dim, size_t shared_mem)
    {
        return INSTANCE.on_kernel_launched_imp(stream_id, grid_dim, block_dim, shared_mem);
    }

    int on_kernel_launched_imp(unsigned short stream_id, dim3 grid_dim, dim3 block_dim, size_t shared_mem)
    {
        if(_disabled)
            return 0;
        int kernel_id = _kernel_id_counter ++;

        if(kernel_id != 0)
            wait_for_kernels_to_complete();  

        unsigned int block_count = grid_dim.x * grid_dim.y * grid_dim.z;
        unsigned int block_size = block_dim.z * block_dim.y * block_dim.x;

        // assume single stream
        update_rss();
        reset_global_memory();
        allocate_shared_pts(block_count, shared_mem);
        _ended_kernel_count = 0;
        _last_kernel_count = block_count * block_size;
//        suspend_all_threads();
        INSTANCE._stm.reset(0 /*XXX:singlestream*/, block_count, block_size); // XXXX lnu
//        resume_all_threads();

        return kernel_id;

    }

    static void print_memory()
    {
        update_rss();
        printf("#MAXRSS: %lu\n", _max_rss);
    }

    static void print_hazards()
    {
        const char* names[] = { "to", "ww", "wa", "aw", "rw", "wr" };
        printf("Hazards:\n");
        for (int i = 0; i < MAX_HAZARDS; ++i)
        {
            fprintf(stderr, "HAZARDS %s %s %li %i\n", "me", names[i], INSTANCE._hazard_loc_count[i].size(), INSTANCE._hazard_count[i]);
        }
    }

    static void update_rss()
    {
        FILE* file = fopen("/proc/self/statm", "r");
        if(file != NULL)
        {
            long tmp, rss;
            fscanf(file, "%li %li ", &tmp, &rss);
            rss = rss * 4096;
            if(rss > _max_rss)
                _max_rss = rss;
            fclose(file);
        }
    }

    static long _max_rss;

private:
    static void wait_for_kernels_to_complete()
    {
        SLIM_DPRINTF(VL_INFO, "Waiting for previous kernel to terminate.");
        pthread_mutex_lock(&INSTANCE._kernel_end_mutex);
        pthread_cond_wait(&INSTANCE._kernel_end_cond, &INSTANCE._kernel_end_mutex);
        pthread_mutex_unlock(&INSTANCE._kernel_end_mutex);
        SLIM_DPRINTF(VL_INFO, "Previous kernel terminated.");
    }

    static void suspend_all_threads()
    { 
         INSTANCE._should_break = true;
         while(INSTANCE._break_count < INSTANCE._thread_count)
             __sync_synchronize();
    }

    static void resume_all_threads()
    {
        INSTANCE._should_break = false;
        INSTANCE._break_count = 0;
        __sync_synchronize();
        pthread_barrier_wait(&INSTANCE._barrier);
    }

private:
    typedef uint64_t tid_t;
    enum Hazard {
        HAZARD_NONE, HAZARD_WW, HAZARD_WA, HAZARD_AW, HAZARD_RW, HAZARD_WR, MAX_HAZARDS
    };
    typedef void (HostDetector::*handler_t)(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae, uint32_t* bcv);
    handler_t _handlers[PROTO_OP_MAX];


    HostDetector() : _global_pt(nullptr) /*,_locals(LOCALS_CAPACITY),*/ 
    {
        _disabled = false;
        SLIM_ASSERT(sizeof(memloc) == 32);

        for(int i = 0; i < PROTO_OP_MAX; ++ i)
            _handlers[i] = &HostDetector::handle_unknown;
        _handlers[OP_LOAD] = &HostDetector::handle_load;
        _handlers[OP_STORE] = &HostDetector::handle_store;
        _handlers[OP_ATOMIC] = &HostDetector::handle_atomic;
        _handlers[OP_SYNCTHREADS] = &HostDetector::handle_syncthreads;
        _handlers[OP_ACQ_CTA] = &HostDetector::handle_acq_cta;
        _handlers[OP_REL_CTA] = &HostDetector::handle_rel_cta;
        _handlers[OP_ACQ_GL] = &HostDetector::handle_acq_gl;
        _handlers[OP_REL_GL] = &HostDetector::handle_rel_gl;
        _handlers[OP_ACQ_SYS] = &HostDetector::handle_acq_sys;
        _handlers[OP_REL_SYS] = &HostDetector::handle_rel_sys;
        _handlers[OP_CONVERGE] = &HostDetector::handle_converge;
        _handlers[OP_START_KERNEL] = &HostDetector::handle_start_kernel;
        _handlers[OP_END_KERNEL] = &HostDetector::handle_end_kernel;

        memset(_hazard_count, 0, sizeof(_hazard_count));
    }

    ~HostDetector()
    {
        if (_global_pt != nullptr)
        {
            reset_global_memory();
            delete[] _global_pt;
        }
    }

    void process_records_imp(PCRecord* record, int count)
    {
        for(int i = 0; i < count; ++ i, ++ record)
        {
            check_for_pause();

            auto& wcv = _stm.cv(record->wid);
            if (record->op == OP_START_KERNEL)
            {
                SLIM_DPRINTF(VL_INFO, "Starting record for warp %li, active mask %x at %p", record->wid, record->active, &wcv);
                wcv.init(record->active);
                continue;
            }
            //increment the epoch here for implicit warp-level synchronization 
            auto* ae = wcv.step(record->active);
            uint32_t* bcv = _stm.bid(record->wid);
            SLIM_DPRINTF(VL_DEBUG, "BCV of %lx (%p) = %u", record->wid, bcv, *bcv);
            if (*bcv < ae->epoch)
            {
                *bcv = ae->epoch;
                SLIM_DPRINTF(VL_DEBUG, "New BCV of %lx (%p) = %u", record->wid, bcv, *bcv);
            }
            auto handler = _handlers[record->op];
            uint32_t mask = 1;
            for (int j = 0; j < WARP_SIZE; ++j, mask <<= 1)
            {
                //uint64_t wid2 = tid_to_wid( get_tid( record, j));
                if(record->active & mask)
                {
                    tid_t tid = get_tid(record, j);
                    void* address = (void*)(uintptr_t)record->address[j];
                    SLIM_DPRINTF(VL_INFO, "OP %s at tid=0x%lx addr=%p loc=%i", get_op_name(record->op), tid, address, record->loc_id);

                    (this->*handler)(record->loc_id, tid, address, ae, bcv);
                }
            }
    
        }
    }
    
    const char* get_op_name(int op)
    {
        switch(op)
        {
            case OP_LOAD: return "load";
            case OP_STORE: return "store";
            case OP_ATOMIC: return "atomic";
            case OP_SYNCTHREADS: return "synchtreads";
            case OP_ACQ_CTA: return "acq_cta";
            case OP_REL_CTA: return "rel_cta";
            case OP_ACQ_GL: return "acq_gl";
            case OP_REL_GL: return "rel_gl";
            case OP_ACQ_SYS: return "acq_sys";
            case OP_REL_SYS: return "rel_sys";
            case OP_CONVERGE: return "convergence";
            case OP_START_KERNEL: return "start_kernel";
            case OP_END_KERNEL: return "end_kernel";
        }
        return "UNKNOWN";
    } 

    inline void report_hazard(memloc* loc, uint32_t loc_id, Hazard hazard, active_epoch::TestResult tr)
    {
        static const char* names[] = { "to", "ww", "wa", "aw", "rw", "wr" };
        static const char* types[] = { "ok" ,"general", "block", "warp", "divergence" };

        log_hazard(loc_id, hazard);
        if(_print_hazards)
            printf("HAZARD! %s %i hzrd=%s nme=%s shrd=%i atm=%i lck=%i rdshrd=%i\n", "me", loc_id, names[hazard], types[tr], loc->is_shared, loc->is_atomic, loc->is_lock, loc->is_readshared);
    }

    void log_hazard(uint32_t loc_id, Hazard hazard)
    {
        auto vt = tbb::concurrent_unordered_map<uint32_t, uint32_t>::value_type(loc_id, 1);
        auto pos = _hazard_loc_count[hazard].insert(vt);
        if (!pos.second)
        {
            uint32_t* data = &pos.first->second;
            __sync_fetch_and_add(data, 1);
        }
        _hazard_count[hazard] += 1;

        pos = _hazard_loc_count[HAZARD_NONE].insert(vt);
        if (!pos.second)
        {
            uint32_t* data = &pos.first->second;
            __sync_fetch_and_add(data, 1);
        }
        _hazard_count[HAZARD_NONE] += 1;
    }

    typedef uint64_t bid_t;/// XXXX: temp
    static inline tid_t get_tid(PCRecord* r, int idx)
    {
        uint32_t w = r->wid & ((1 << WARP_SHIFT) - 1);
        uint64_t h = r->wid >> BLOCK_SHIFT ;
        return h << BLOCK_SHIFT | w << WARP_SHIFT | idx;
        // bug: sid bid wid(16b) -> sid bid wid(16b) idx(5b)
        // return r->wid << WARP_SHIFT | idx;
    }
    static inline bid_t tid_to_bid(tid_t t )
    {
        return (bid_t)(t >> BLOCK_SHIFT);
    }

    void handle_unknown(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae, uint32_t *bcv)
    {
        fprintf(stderr, "ERROR!: Unknown operation!\n");
        exit(-1);
    }
    void handle_load(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae, uint32_t *bcv)
    {
        memloc* loc = get_memloc(tid, address);
        if (loc == nullptr)
            return;

        guard g(loc);

        const int my_epoch = ae->epoch;

        if (loc->is_readshared)
        {
            mem_cv* cv = loc->read_cv;
            auto it = cv->find(tid);
            if (it != cv->end() && it->second == my_epoch)
                return;
        }
        else
        {
            if (loc->read_tid == tid && loc->read_epoch == my_epoch)
                return;
        }

        if (loc->write_epoch != 0)
        {
            auto tr = ae->test_epoch(tid, loc->write_tid, loc->write_epoch);
            if (tr != active_epoch::EARLIER)
                report_hazard(loc, loc_id, HAZARD_WR, tr);
        }
        set_read(loc, my_epoch, tid);
    }

    void handle_store(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae, uint32_t *bcv)
    {
        memloc* loc = get_memloc(tid, address);
        if (loc == nullptr)
            return;

        guard g(loc);
        
        const int my_epoch = ae->epoch;
        if (loc->write_tid == tid && loc->write_epoch == my_epoch)
            return;

        if (loc->write_epoch != 0)
        {
            if (loc->is_atomic)
            {
                loc->is_atomic = false;
                report_hazard(loc, loc_id, HAZARD_AW, ae->get_fail_type(tid, loc->write_tid));
            }
            else
            {
                auto tr = ae->test_epoch(tid, loc->write_tid, loc->write_epoch);
                if (tr != active_epoch::EARLIER)
                    report_hazard(loc, loc_id, HAZARD_WW, tr);
            }
        }

        if (loc->is_readshared)
        {
            mem_cv* cv = loc->read_cv;
            for(auto it = cv->begin(); it != cv->end(); ++ it)
            {
                if (tid != it->first)
                {
                    auto tr = ae->test_epoch(tid, it->first, it->second);
                    if (tr != active_epoch::EARLIER)
                        report_hazard(loc, loc_id, HAZARD_RW, tr);
                }
            }
        }
        else
        {
            if (loc->read_epoch != 0 && tid != loc->read_tid)
            {
                auto tr = ae->test_epoch(tid, loc->read_tid, loc->read_epoch);
                if (tr != active_epoch::EARLIER)
                    report_hazard(loc, loc_id, HAZARD_RW, tr);
            }
        }

        reset_read(loc, my_epoch, tid);
        set_write(loc, my_epoch, tid);
    }
    void handle_atomic(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae, uint32_t *bcv)
    {
        memloc* loc = get_memloc(tid, address);
        if (loc == nullptr)
            return;
        guard g(loc);

        const int my_epoch = ae->epoch;
        if (loc->write_tid == tid && loc->write_epoch == my_epoch)
            return;

        if (loc->write_epoch != 0)
        {
            if (!loc->is_atomic)
            {
                auto tr = ae->test_epoch(tid, loc->write_tid, loc->write_epoch);
                if (tr != active_epoch::EARLIER)
                    report_hazard(loc, loc_id, HAZARD_WA, tr);
            }
        }

        if (loc->is_readshared)
        {
            mem_cv* cv = loc->read_cv;
            for(auto it = cv->begin(); it != cv->end(); ++ it)
            {
                if (tid != it->first)
                {
                    auto tr = ae->test_epoch(tid, it->first, it-> second);
                    if (tr != active_epoch::EARLIER)
                        report_hazard(loc, loc_id, HAZARD_RW, tr);
                }
            }
        }
        else
        {
            if (loc->read_epoch != 0 && tid != loc->read_tid)
            {
                auto tr = ae->test_epoch(tid, loc->read_tid, loc->read_epoch);
                if (tr != active_epoch::EARLIER)
                    report_hazard(loc, loc_id, HAZARD_RW, tr);
            }
        }

        reset_read(loc, my_epoch, tid);
        set_write(loc, my_epoch, tid);
        loc-> is_atomic = true;
    }

    void handle_syncthreads(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae, uint32_t *bcv)
    {
        SLIM_ASSERT(ae->epoch <= *bcv);
        ae->epoch.max_of(*bcv);
        ae->cv.block_max_of(tid, *bcv);
        *bcv = ae->epoch;
        SLIM_DPRINTF(VL_DEBUG, "Setting BCV of %lx (%p) = %u", tid, bcv, *bcv);
    }
    void handle_acq_cta(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae, uint32_t *bcv)
    {
        bid_t blk = tid_to_bid( tid );
        thread_cv blk_cv(_svcs.get_block_vc( const_cast<void *>(address), blk));
        SLIM_ASSERT( ae->epoch >= blk_cv.get_epoch(tid, tid) );
        ae->cv.max_of( blk_cv );
    }
    void handle_rel_cta(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae, uint32_t *bcv)
    {
        bid_t blk = tid_to_bid( tid );
        thread_cv local_cv(thread_cv(tid, ae->epoch));
        local_cv.max_of(ae->cv);
        _svcs.max_pb_vc(const_cast<void *>(address), blk, local_cv );
        ae->epoch.inc();
    }
    void handle_acq_gl(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae, uint32_t *bcv)
    {
        thread_cv all_cv = _svcs.get_global_max_vc( const_cast<void *>(address) );
        SLIM_ASSERT( ae-> epoch >= all_cv.get_epoch(tid, tid ));
        ae->cv.max_of( all_cv );
    }
    void handle_rel_gl(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae, uint32_t *bcv)
    {
        thread_cv local_cv= thread_cv(tid, ae->epoch );
        local_cv.max_of( ae->cv );
        _svcs.max_gl_vc(const_cast<void *>(address), local_cv);
        ae->epoch.inc(); 
    }

    void handle_acq_sys(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae, uint32_t *bcv)
    {
        handle_acq_gl(loc_id, tid, address, ae, bcv );
    }

    void handle_rel_sys(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae, uint32_t *bcv)
    {
        handle_rel_gl(loc_id, tid, address, ae, bcv );
    }

    void handle_converge(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae, uint32_t *bcv)
    {
    }

    void handle_start_kernel(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae, uint32_t *bcv)
    {
    }

    void handle_end_kernel(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae, uint32_t *bcv)
    {
        ++ _ended_kernel_count;
        //printf("called end_kernel: %d, %d\n", _ended_kernel_count.load(), _last_kernel_count.load());
        if(_ended_kernel_count.load() == _last_kernel_count.load())
        {
             pthread_mutex_lock(&_kernel_end_mutex);
             pthread_cond_signal(&_kernel_end_cond);
             pthread_mutex_unlock(&_kernel_end_mutex);
        }
    }

private:

private:
    void allocate_shared_pts(int block_count, int shmem_size)
    {
        if(_shared_pt != nullptr)
        {
            for(int i = 0; i < _shared_pt_entries; ++ i)
                if(_shared_pt[i] != nullptr) {
                    free_memloc(_shared_pt[i], _kernel_shmem_size);
                    _shared_pt[i] = nullptr;
                 }
            delete[] _shared_pt;
        }
        //_kernel_shmem_size = (1ULL << 32) >> __builtin_clz(shmem_size - 1);
        _kernel_shmem_size = 48 * 1024;
        _shared_pt_entries = block_count;
        _shared_pt = new memloc*[block_count];
        memset(_shared_pt, 0, sizeof(memloc*)*block_count);
    }

    void reset_global_memory()
    {
        for(int i = 0; i < _global_pt_entries; ++ i)
        {
            // check & update should be atomic 
            std::lock_guard<std::mutex> guard(_global_lock);
            if(_global_pt[i] != nullptr) {
                free_memloc(_global_pt[i], 1 << PT_ALIGNMENT_SHIFT);
            }
            _global_pt[i] = nullptr;
        }
    }

    static void free_memloc(memloc* entries, unsigned int count)
    {
        for(int i = 0 ;i < count; ++ i)
            free_memloc(entries[i]);
        // we also needs to free the the array of memlocs !
        delete[] entries;
    }

    static inline void free_memloc(memloc& loc)
    {
        if(loc.is_readshared)
        {
           delete loc.read_cv; 
        }
    }

    inline memloc* get_memloc(uint64_t tid, const void* address)
    {
        uint64_t addrptr = (uint64_t)(uintptr_t)address;
        uint32_t type = (uint32_t)(addrptr >> MEMORY_TYPE_SHIFT);
        addrptr =  addrptr & ~(3ULL << MEMORY_TYPE_SHIFT);
        //printf("Access type %x to 0x%lx\n", type, addrptr);
        switch(type)
        {
        case MEMORY_TYPE_GLOBAL: return get_memloc_global(tid, addrptr); 
        case MEMORY_TYPE_SHARED: return get_memloc_shared(tid, addrptr); 
        case 0: return nullptr;  // local memory
        }
        SLIM_ASSERT(false); 
        return nullptr;
    }

    inline memloc* get_memloc_shared(uint64_t tid, uintptr_t address)
    {
        //printf("Getting shared for %lx\n", address);
        uint32_t block = tid >> BLOCK_SHIFT; // XXX
#ifdef SLIM_DEBUG
        uint32_t realblock = address >> 32;
        SLIM_ASSERT(block == realblock);
        SLIM_ASSERT(address < _kernel_shmem_size);
#endif
        if(block >= _shared_pt_entries)
        {
            printf("***** Invalid block value %i, %i\n", block, address >>32);
            block = address >> 32;
            if(block >= _shared_pt_entries)
            {
                printf("*++*++* Invalid block value %i\n", block);
                exit(1);
            }
        }
        memloc* segptr = _shared_pt[block];
        address = address & (_kernel_shmem_size - 1);
        // intra-block is always in same thread on host side
        if (segptr == nullptr)
            _shared_pt[block] = segptr = memloc::alloc_memlocs(_kernel_shmem_size, true);

        return &segptr[address];
    }

    inline memloc* get_memloc_global(uint64_t tid, uintptr_t address)
    {
        //printf("Getting global for %lx\n", address);
        const uint32_t seg = (uint32_t)(address - _min_global_address) >> PT_ALIGNMENT_SHIFT;
        const uint32_t ofs = (uint32_t)(address - _min_global_address) & (PT_ALIGNMENT - 1);
        SLIM_ASSERT(seg < _global_pt_entries);

        if (address < _min_global_address || address >= _max_global_address)
            return nullptr;
        // to avoid data race on _global_pt[seg]
        memloc* segptr = read_memloc_safe(seg);
        //memloc* segptr = _global_pt[seg];
        if (segptr == nullptr)
            segptr = alloc_memloc_segment(seg);
        //printf("Getting global %lx from %x:%x\n", address, seg, ofs);
        return &segptr[ofs];
    }
    memloc* read_memloc_safe(uint32_t seg) {
        std::lock_guard<std::mutex> guard(_global_lock);
        return _global_pt[seg];
    }
    memloc* alloc_memloc_segment(uint32_t seg)
    {
        std::lock_guard<std::mutex> guard(_global_lock);
        if (_global_pt[seg] == nullptr)
        {
            //printf("Allocating global seg: %i\n", seg);
            const int entries = 1 << PT_ALIGNMENT_SHIFT;
            _global_pt[seg] = memloc::alloc_memlocs(entries, false);
        }
        return _global_pt[seg];
    }

    void destroy_memlocs(memloc* locs)
    {
        // XXX TODO    
    }

    FINLINE void reset_read(memloc* loc, uint32_t epoch, tid_t tid)
    {
        if (loc->is_readshared)
            delete loc->read_cv;
        loc->read_tid = tid;
        loc->read_epoch = epoch;
        loc->is_readshared = false;
    }

    FINLINE void set_read(memloc* loc, uint32_t epoch, tid_t tid)
    {
        if (!loc->is_readshared)
        {
            if (loc->read_epoch == 0)
            {
                loc->read_epoch = epoch;
                loc->read_tid = tid;
                return;
            }
            
            if(loc->read_tid == tid)
            {
                loc->read_epoch = epoch;
                return;
            }
            auto cur_tid = loc->read_tid;
            auto cur_epoch = loc->read_epoch;
            loc->is_readshared = true;
            loc->read_cv = new mem_cv();
            loc->read_cv->insert(mem_cv::value_type(cur_tid, cur_epoch));
        }
        loc->read_cv->insert(mem_cv::value_type(tid, epoch));
    }

    FINLINE void set_write(memloc* loc, uint32_t epoch, tid_t tid)
    {
        loc->write_tid = tid;
        loc->write_epoch = epoch;
    }


private:
    // manage vcs for all synchronization locations
    sync_vcs _svcs; 
    int _kernel_id_counter;
    uintptr_t _min_global_address, _max_global_address;
    threads_map _stm;
    std::mutex _global_lock;
    unsigned int _global_pt_entries, _shared_pt_entries;
    memloc** _global_pt;
    memloc** _shared_pt;
    //tbb::concurrent_unordered_map<uintptr_t,memloc> _locals;
    tbb::concurrent_unordered_map<uint32_t, uint32_t> _hazard_loc_count[MAX_HAZARDS];
    int _hazard_count[MAX_HAZARDS];
    volatile bool _should_break;
    volatile int _break_count;
    int _thread_count;
    pthread_barrier_t _barrier;

    // these assume a single stream for know
    std::atomic<unsigned int> _last_kernel_count, _ended_kernel_count;
    size_t _kernel_shmem_size;
    pthread_cond_t _kernel_end_cond;
    pthread_mutex_t _kernel_end_mutex;
    bool _print_hazards;
    bool _disabled;
}; 

HostDetector HostDetector::INSTANCE;
long HostDetector::_max_rss = 0;

