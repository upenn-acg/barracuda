#pragma once

#include <unordered_map>
#include <tbb/concurrent_unordered_map.h>
#include <mutex>
#include "warp_cv.hpp"
#include "mem_cv.hpp"
#include "debug.h"

class HostDetector
{
public:
	enum {
		MAX_STREAMS = 1024,
		LOCALS_CAPACITY = 65536
	};

	enum {
		PT_ALIGNMENT_SHIFT = 20
	};
	enum { PT_ALIGNMENT = (1 << PT_ALIGNMENT_SHIFT) };
	enum { MAX_SHMEM_SIZE = 65536 };
	enum { MAX_BLOCKS = 4096 };

	static HostDetector INSTANCE;

	void on_init()
	{
		size_t free = 0, total = 0;
		VERIFY(ORIG(cudaMemGetInfo)(&free, &total));
		uintptr_t ptr;
		VERIFY(ORIG(cudaMalloc)((void**)&ptr, free));
		VERIFY(ORIG(cudaFree)((void*)ptr));
		uintptr_t used = total - free;
	
		_min_global_address = ((ptr > used) ? (ptr - used) : 0) & ~(PT_ALIGNMENT - 1);
		_max_global_address = (ptr + total + PT_ALIGNMENT - 1) & ~(PT_ALIGNMENT - 1);
				
		size_t entries = (_max_global_address - _min_global_address) >> PT_ALIGNMENT_SHIFT;
		_global_pt = new memloc*[entries];
		memset(_global_pt, 0, sizeof(memloc*) * entries);

		_shared_pt = new memloc*[MAX_BLOCKS];
		memset(_shared_pt, 0, sizeof(memloc*) * MAX_BLOCKS);
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
    }

    static void on_stream_destroyed(unsigned short stream_id)
    {
    }

    static void on_kernel_launched(unsigned short stream_id, dim3 grid_dim, dim3 block_dim)
    {
		INSTANCE._stm.reset(stream_id, grid_dim, block_dim);
    }

    static void on_kernel_done(unsigned short stream_id)
    {
    }

	static void print_hazards()
	{
		const char* names[] = { "to", "ww", "wa", "aw", "rw", "wr" };
		for (int i = 0; i < MAX_HAZARDS; ++i)
		{
			fprintf(stderr, "HAZARDS %s %s %li %i\n", "me", names[i], INSTANCE._hazard_loc_count[HAZARD_NONE].size(), INSTANCE._hazard_count[HAZARD_NONE]);
		}
	}

private:
    typedef uint64_t tid_t;
	enum Hazard {
		HAZARD_NONE, HAZARD_WW, HAZARD_WA, HAZARD_AW, HAZARD_RW, HAZARD_WR, MAX_HAZARDS
	};
	typedef void (HostDetector::*handler_t)(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae);
    handler_t _handlers[PROTO_OP_MAX];


	HostDetector() : _global_pt(nullptr) /*,_locals(LOCALS_CAPACITY),*/ 
    {
		SLIM_ASSERT(sizeof(memloc) == 32);

        for(int i = 0; i < PROTO_OP_MAX; ++ i)
            _handlers[i] = &HostDetector::handle_unknown;
        _handlers[OP_LOAD] = &HostDetector::handle_load;
        _handlers[OP_STORE] = &HostDetector::handle_store;
        _handlers[OP_ATOMIC] = &HostDetector::handle_atomic;
        _handlers[OP_ACQ_CTA] = &HostDetector::handle_acq_cta;
		_handlers[OP_REL_CTA] = &HostDetector::handle_rel_cta;
		_handlers[OP_ACQ_GL] = &HostDetector::handle_acq_gl;
		_handlers[OP_REL_GL] = &HostDetector::handle_rel_gl;
		_handlers[OP_ACQ_SYS] = &HostDetector::handle_acq_sys;
		_handlers[OP_REL_SYS] = &HostDetector::handle_rel_sys;
		_handlers[OP_CONVERGE] = &HostDetector::handle_converge;
		_handlers[OP_START_KERNEL] = &HostDetector::handle_start_kernel;

		memset(_hazard_count, 0, sizeof(_hazard_count));
    }

    ~HostDetector()
    {
		if (_global_pt != nullptr)
		{
			int entries = (int)((_max_global_address - _min_global_address) >> PT_ALIGNMENT_SHIFT);
			for (int i = 0; i < entries; ++i)
				destroy_memlocs(_global_pt[i]);
			delete[] _global_pt;
		}
    }

    void process_records_imp(PCRecord* record, int count)
    {
        for(int i = 0; i < count; ++ i, ++ record)
        {
			auto& wcv = _stm.cv(record->wid);
			if (record->op == OP_START_KERNEL)
			{
				wcv.init(record->active);
				continue;
			}

			auto* ae = wcv.step(record->active);
			uint32_t* bcv = _stm.bid(record->wid);
			if (*bcv < ae->epoch)
				*bcv = ae->epoch;
			auto handler = _handlers[record->op];
			uint32_t mask = 1;
			for (int j = 0; j < WARP_SIZE; ++j, mask <<= 1)
            {
                if(record->active & mask)
                {
                    //printf("Processing %lx active=0x%x op=%i loc=%i WID:%i adddr=%p\n", 
                    //    record->wid, record->active, record->op, record->loc_id, j, (void*)record->address[j]);

					if(record->op == OP_SYNCTHREADS)
						handle_syncthreads(record->loc_id, get_tid(record, j), (void*)(uintptr_t)record->address[j], ae, *bcv);
					else
						(this->*handler)(record->loc_id, get_tid(record, j), (void*)(uintptr_t)record->address[j], ae);
                }
            }
    
        }
    }
     
	void report_hazard(uint32_t loc_id, Hazard hazard, active_epoch::TestResult tr)
	{
		const char* names[] = { "to", "ww", "wa", "aw", "rw", "wr" };
		const char* types[] = { "ok" ,"general", "warp", "divergence" };
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

		printf("HAZARD %s %i %s %s\n", "me", loc_id, names[hazard], types[tr]);

	}

    static inline tid_t get_tid(PCRecord* r, int idx)
    {
        return r->wid << WARP_SHIFT | idx;
    }


	void handle_unknown(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae)
    {
        fprintf(stderr, "Unknown operation!\n");
        exit(-1);
    }
	void handle_load(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae)
    {
		SLIM_VERBOSEONLY(printf("Read from 0x%x: %p\n", tid, address);)

		memloc* loc = get_memloc(tid, false, address);
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
				report_hazard(loc_id, HAZARD_WR, tr);
		}
		set_read(loc, my_epoch, tid);
	}

	void handle_store(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae)
    {
		SLIM_VERBOSEONLY(printf("Write to 0x%x: %p\n", tid, address);)
		memloc* loc = get_memloc(tid, false, address);
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
				report_hazard(loc_id, HAZARD_AW, ae->get_fail_type(tid, loc->write_tid));
			}
			else
			{
				auto tr = ae->test_epoch(tid, loc->write_tid, loc->write_epoch);
				if (tr != active_epoch::EARLIER)
					report_hazard(loc_id, HAZARD_WW, tr);
			}
		}

		if (loc->is_readshared)
		{
			mem_cv* cv = loc->read_cv;
			for(auto it = cv->begin(); it != cv->end(); ++ it)
			{
				if (tid != it->first)
				{
					auto tr = ae->test_epoch(tid, loc->read_tid, loc->read_epoch);
					if (tr != active_epoch::EARLIER)
						report_hazard(loc_id, HAZARD_RW, tr);
				}
			}
		}
		else
		{
			if (loc->read_epoch != 0 && tid != loc->read_tid)
			{
				auto tr = ae->test_epoch(tid, loc->read_tid, loc->read_epoch);
				if (tr != active_epoch::EARLIER)
					report_hazard(loc_id, HAZARD_RW, tr);
			}
		}

		reset_read(loc, my_epoch, tid);
		set_write(loc, my_epoch, tid);
    }
	void handle_atomic(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae)
    {
        SLIM_VERBOSEONLY(printf("Atomic write to 0x%x: %p\n", tid, address);)
    }
	void handle_syncthreads(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae, uint32_t bcv)
    {
		SLIM_VERBOSEONLY(printf("Syncthreads()\n", tid);)
		SLIM_ASSERT(ae->epoch <= bcv);
		ae->epoch.max_of(bcv);
		ae->cv.block_max_of(tid, bcv);
    }
	void handle_acq_cta(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae)
    {
        SLIM_VERBOSEONLY(printf("acq_cta()\n", tid);)
    }
	void handle_rel_cta(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae)
	{
		SLIM_VERBOSEONLY(printf("rel_cta()\n", tid);)
	}
	void handle_acq_gl(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae)
	{
		SLIM_VERBOSEONLY(printf("acq_gl()\n", tid);)
	}
	void handle_rel_gl(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae)
	{
		SLIM_VERBOSEONLY(printf("rel_gl()\n", tid);)
	}
	void handle_acq_sys(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae)
	{
		SLIM_VERBOSEONLY(printf("acq_sys()\n", tid);)
	}
	void handle_rel_sys(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae)
	{
		SLIM_VERBOSEONLY(printf("rel_sysd()\n", tid);)
	}
	void handle_converge(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae)
	{
		SLIM_VERBOSEONLY(printf("converge()\n", tid);)
	}
	void handle_start_kernel(uint32_t loc_id, tid_t tid, const void* address, active_epoch* ae)
	{
		SLIM_VERBOSEONLY(printf("start_kernel()\n", tid);)
	}

private:
	class stream_threads_map
	{
	public: 
		stream_threads_map() : _cvs(nullptr)
		{
		}

		~stream_threads_map()
		{ 
			delete[] _cvs;
		}

		void reset(const dim3& grid_dim, const dim3& block_dim)
		{
			if(_cvs != nullptr)
				delete[] _cvs;
			_blk_size = block_dim.z * block_dim.y * block_dim.x;
			_blk_count = grid_dim.z * grid_dim.y * grid_dim.x;
			int amount = (_blk_size * _blk_count + (1 << WARP_SHIFT) - 1) >> WARP_SHIFT;
			SLIM_DEBUGONLY(_size = amount;)
			printf("Creating %i warp CVs\n", amount);
			_cvs = new warp_cv[amount];
			_bids = new uint32_t[_blk_count];
			memset(_bids, 0, sizeof(_bids[0]) * _blk_count);
		}

		inline warp_cv& cv(uint64_t wid)
		{
			int blk = (int)(wid >> BLOCK_SHIFT);
			wid = wid & ((1 << BLOCK_SHIFT) - 1);
			SLIM_ASSERT(wid < _blk_size);
			SLIM_ASSERT(blk < _blk_count);
			return _cvs[blk * _blk_size + wid];
		}

		inline uint32_t* bid(uint64_t wid)
		{
			int blk = wid >> BLOCK_SHIFT;
			SLIM_ASSERT(blk < _blk_count);
			return &_bids[blk];
		}

	private:
		SLIM_DEBUGONLY(int _size;)
		int _blk_size;
		int _blk_count;
		warp_cv *_cvs;
		uint32_t* _bids;
	};

	class threads_map
	{
	public:
		threads_map()
		{
		}

		~threads_map()
		{
		}

		void reset(int stream, const dim3& grid_dim, const dim3& block_dim)
		{
			SLIM_ASSERT(stream < MAX_STREAMS);
			_st[stream].reset(grid_dim, block_dim);
		}

		inline warp_cv& cv(uint64_t tid)
		{
			uint32_t stream = tid_to_stream(tid);
			uint64_t wid = tid_to_wid(tid);
			SLIM_ASSERT(stream < MAX_STREAMS);
			return _st[stream].cv(wid);
		}
		inline uint32_t* bid(uint64_t tid)
		{
			uint32_t stream = tid_to_stream(tid);
			uint64_t wid = tid_to_wid(tid);
			SLIM_ASSERT(stream < MAX_STREAMS);
			return _st[stream].bid(wid);
		}
	private:
		stream_threads_map _st[MAX_STREAMS];
	};

	struct memloc
	{
		union
		{
			uint64_t read_tid;
			mem_cv* read_cv;
		};
		uint64_t write_tid;
		uint32_t read_epoch;
		uint32_t write_epoch;
		volatile uint32_t lock;
		uint8_t is_atomic;
		uint8_t is_readshared;
		uint8_t is_lock;
		uint8_t is_shared;
	};

private:
	inline memloc* get_memloc(uint64_t tid, bool is_shared, const void* address)
	{
		uintptr_t addrptr = (uintptr_t)address;
		if (is_shared)
			return get_memloc_shared(tid, addrptr);
		return get_memloc_global(addrptr);
	}

	inline memloc* get_memloc_shared(uint64_t tid, uintptr_t address)
	{
		int block = tid >> 32; // XXX
		SLIM_ASSERT(address < MAX_SHMEM_SIZE);
		memloc* segptr = _shared_pt[block];
		// intra-block is always in same thread on host side
		if (segptr == NULL)
		{
			segptr = new memloc[MAX_SHMEM_SIZE];
			for (int i = 0; i < MAX_SHMEM_SIZE; ++i)
			{
				memset(&segptr[i], 0, sizeof(segptr[i]));
				segptr[i].is_shared = true;
			}
			_shared_pt[block] = segptr;
		}
		return &segptr[address];
	}

	inline memloc* get_memloc_global(uintptr_t address)
	{
		if (address < _min_global_address || address >= _max_global_address)
			return nullptr;

		const uint32_t seg = (uint32_t)(address - _min_global_address) >> PT_ALIGNMENT_SHIFT;
		const uint32_t ofs = (uint32_t)(address - _min_global_address) & (PT_ALIGNMENT - 1);
		memloc* segptr = _global_pt[seg];
		if (segptr == NULL)
			segptr = alloc_memloc(seg);
		return &segptr[ofs];
	}

	memloc* alloc_memloc(uint32_t seg)
	{
		std::lock_guard<std::mutex> guard(_global_lock);
		if (_global_pt[seg] == NULL)
		{
			const int entries = 1 << PT_ALIGNMENT_SHIFT;
			_global_pt[seg] = new memloc[entries];
			memset(_global_pt[seg], 0, sizeof(memloc) * entries);
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

	enum {
		UNLOCKED = 0, LOCKED = 1
	};

	class guard
	{
	public:
		guard(memloc* loc) : _loc(loc)
		{
			while (__sync_lock_test_and_set((uint32_t*)&loc->lock, LOCKED) != UNLOCKED)
			{
				volatile auto* lock = &loc->lock;
				while (*lock);
			}
		}

		~guard()
		{
			__sync_lock_release((uint32_t*)&_loc->lock);
		}
	private:
		memloc* _loc;
	};

private:
	uintptr_t _min_global_address, _max_global_address;
	threads_map _stm;
	std::mutex _global_lock;
	memloc** _global_pt;
	memloc** _shared_pt;
	//tbb::concurrent_unordered_map<uintptr_t,memloc> _locals;
	tbb::concurrent_unordered_map<uint32_t, uint32_t> _hazard_loc_count[MAX_HAZARDS];
	int _hazard_count[MAX_HAZARDS];
};

HostDetector HostDetector::INSTANCE;
