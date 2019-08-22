#pragma once

#include <stdint.h>
#include <algorithm>
#include "debug.h"

#define CHECK_AND_DELETE(ptr) do {\
 if (!ptr) { \
    delete ptr;\
    ptr = 0;\
 }}while(0)

typedef uint64_t tid_t;

struct thread_cv
{
public:

	thread_cv()
    {
        b.flags = MODE_BASIC; b.block = b.warp = 0; 
    }
	thread_cv(tid_t tid, uint32_t epoch)
	{
		total_level_cv * tcv = make_tcv();
		tcv -> block = 0;
		tcv -> cv_epochs[tid] = epoch; 
	}
    ~thread_cv()
    {
		warp_level_cv *wcv = nullptr;
		total_level_cv *tcv = nullptr;
        switch(b.flags)
        {
        case MODE_BASIC: 
            break;
        case MODE_WARP_CV:
			wcv = (warp_level_cv*)p.ptr;
            CHECK_AND_DELETE(wcv);
			break;
		case MODE_TOTAL_CV:
			tcv = (total_level_cv*)p.ptr;
            CHECK_AND_DELETE(tcv);
            break;
        }
    }

	void diverge_from(thread_cv& other_cv, uint32_t active_mask, uint32_t epoch)
	{
		release();

		// fast path
		if (other_cv.b.flags == MODE_BASIC && other_cv.b.warp == 0)
		{
			b.flags = MODE_BASIC;
			b.block = other_cv.b.block;
			b.warp = epoch;
			return;
		}

		// slow path
		uint32_t mask = 1;
		warp_level_cv *wcv = nullptr, *owcv = nullptr;
		total_level_cv *tcv = nullptr, *otcv = nullptr;
		// XXX: no need to free & release in many cases
		switch (other_cv.b.flags)
		{
		case MODE_BASIC:
			wcv = make_wcv();
			for (int j = 0; j < WARP_SIZE; ++j, mask <<= 1)
				wcv->epochs[j] = (active_mask & mask) ? epoch : other_cv.b.warp;
			wcv->block = other_cv.b.block;
			break;
		case MODE_WARP_CV:
			wcv = make_wcv();
			owcv = (warp_level_cv*)other_cv.p.ptr;
			wcv->block = owcv->block;
			for (int j = 0; j < WARP_SIZE; ++j, mask <<= 1)
				wcv->epochs[j] = (active_mask & mask) ? epoch : owcv->epochs[j];
			break;
		case MODE_TOTAL_CV:
			tcv = make_tcv();
			otcv = (total_level_cv*)other_cv.p.ptr;
			tcv->block = otcv->block;
			tcv->cv_epochs.insert(otcv->cv_epochs.begin(), otcv->cv_epochs.end());
			for (int j = 0; j < WARP_SIZE; ++j, mask <<= 1)
				tcv->epochs[j] = (active_mask & mask) ? epoch : otcv->epochs[j];
			break;
		}
	}
	void max_of(thread_cv& other)
	{
		warp_level_cv *wcv = nullptr, *owcv = nullptr;
		total_level_cv *tcv = nullptr, *otcv = nullptr;
		bool all_same = true;
		int last = 0;

		switch (other.b.flags)
		{
		case MODE_BASIC:
			switch (b.flags)
			{
			case MODE_BASIC:
				if (b.warp < other.b.warp)
					b.warp = other.b.warp;
				break;
			case MODE_WARP_CV:
				wcv = (warp_level_cv*)p.ptr;
				for (int i = 0; i < WARP_SIZE; ++i)
				{
					if (wcv->epochs[i] < other.b.warp)
						wcv->epochs[i] = other.b.flags;
					if (all_same)
					{
						if (i > 0)
						{
							if (last != wcv->epochs[i])
								all_same = false;
						}
						else
							last = wcv->epochs[0];
					}
				}
				if (wcv->block < b.block)
					wcv->block = b.block;
				if (all_same)
					make_basic(last);
				break;
			case MODE_TOTAL_CV:
				// XXX: add inflation
				tcv = (total_level_cv*)p.ptr;
				if (tcv->block < b.block)
					tcv->block = b.block;
				for (int i = 0; i < WARP_SIZE; ++i)
				{
					if (tcv->epochs[i] < other.b.warp)
						tcv->epochs[i] = other.b.flags; //TODO: Bug? 
				}
				break;
			}
			break; 

		case MODE_WARP_CV:
			owcv = (warp_level_cv*)other.p.ptr;
			switch (b.flags)
			{
			case MODE_BASIC:
				{
				basic cb = b;
				wcv = make_wcv();
				wcv->block = std::max(cb.block, owcv->block);
				for (int i = 0; i < WARP_SIZE; ++i)
				{
					wcv->epochs[i] = (owcv->epochs[i] < cb.warp) ? cb.warp : owcv->epochs[i];
					if (all_same)
					{
						if (i > 0)
						{
							if (last != wcv->epochs[i])
								all_same = false;
						}
						else
							last = wcv->epochs[0];
					}
				}
				if (all_same)
					make_basic(last);
				}
				break;
			case MODE_WARP_CV:
				wcv = (warp_level_cv*)p.ptr;
				wcv->block = std::max(wcv->block, owcv->block);
				for (int i = 0; i < WARP_SIZE; ++i)
				{
					if (wcv->epochs[i] < owcv->epochs[i])
						wcv->epochs[i] = owcv->epochs[i];
					if (all_same)
					{
						if (i > 0)
						{
							if (last != wcv->epochs[i])
								all_same = false;
						}
						else
							last = wcv->epochs[0];
					}
				}
				if (all_same)
					make_basic(last);
				break;
			case MODE_TOTAL_CV:
				tcv = (total_level_cv*)p.ptr;
				tcv->block = std::max(tcv->block, owcv->block);
				for (int i = 0; i < WARP_SIZE; ++i)
				{
					if (tcv->epochs[i] < owcv->epochs[i])
						tcv->epochs[i] = owcv->epochs[i];
				}
				break;
			}
			break; 

		case MODE_TOTAL_CV:
			otcv = (total_level_cv*)other.p.ptr;
			switch (b.flags)
			{
			case MODE_BASIC:
				{
				basic cb = b;
				tcv = make_tcv();
				tcv->block = std::max(cb.block, otcv->block);
				memcpy(tcv->epochs, otcv->epochs, sizeof(tcv->epochs));
				tcv->cv_epochs.insert(otcv->cv_epochs.begin(), otcv->cv_epochs.end());
				for (int i = 0; i < WARP_SIZE; ++i)
					tcv->epochs[i] = (otcv->epochs[i] < cb.warp) ? cb.warp : otcv->epochs[i];
				}
				break;
			case MODE_WARP_CV:
				wcv = (warp_level_cv*)p.ptr;
				make_tcv();
				
				tcv->cv_epochs.insert(otcv->cv_epochs.begin(), otcv->cv_epochs.end());
				tcv->block = std::max(wcv->block, otcv->block);
				for (int i = 0; i < WARP_SIZE; ++i)
				{
					if (tcv->epochs[i] < otcv->epochs[i])
						tcv->epochs[i] = otcv->epochs[i];
				}
				CHECK_AND_DELETE(wcv);
				break;
			case MODE_TOTAL_CV:
				tcv = (total_level_cv*)p.ptr;
				tcv->block = std::max(tcv->block, otcv->block);
				for (int i = 0; i < WARP_SIZE; ++i)
				{
					if (tcv->epochs[i] < otcv->epochs[i])
						tcv->epochs[i] = otcv->epochs[i];
				}
				for (auto it = otcv->cv_epochs.begin();
					it != otcv->cv_epochs.end();
					++it)
				{
					auto mine = tcv->cv_epochs.find(it->first);
					if (mine == tcv->cv_epochs.end())
						tcv->cv_epochs.insert(*it);
					else
						mine->second = std::max(mine->second, it->second);
				}
				break;
			}
			break;
		}
	}

	void block_max_of(tid_t my_tid, uint32_t bcv)
	{
		total_level_cv* tcv;
		int my_block;
		switch (b.flags)
		{
		case MODE_BASIC:
			SLIM_ASSERT(b.warp <= bcv);
			SLIM_ASSERT(b.block <= bcv);
			b.block = bcv;
			b.warp = bcv;
			break;

		case MODE_WARP_CV:
			release();
			b.flags = MODE_BASIC;
			b.warp = bcv;
			b.block = bcv;
			break;

		case MODE_TOTAL_CV:
			tcv = (total_level_cv*)p.ptr;
			my_block = (int)(my_tid >> BLOCK_SHIFT);

			bool deflate = true;
			for(auto it = tcv->cv_epochs.begin(); it != tcv->cv_epochs.end(); )
			{
				auto cur = it ++; 
				int curb = (int)(cur->first >> BLOCK_SHIFT);
				if (curb != my_block)
					deflate = false;
				else
					tcv->cv_epochs.erase(cur);
			}
			if (deflate)
			{
				release();
				b.flags = MODE_BASIC;
				b.warp = bcv;
				b.block = bcv;
			}
			else
			{
				tcv->block = bcv;
				for (int i = 0; i < WARP_SIZE; ++i)
				{
					SLIM_ASSERT(tcv->epochs[i] <= bcv);
					tcv->epochs[i] = bcv;
				}
			}

		}
	}

	enum {
		ZERO_EPOCH = 0,
	};
	uint32_t get_epoch(tid_t mytid, tid_t tid)
	{
		return get_epoch(mytid, tid, (((mytid >> WARP_SHIFT) == (tid >> WARP_SHIFT))));
	}

	uint32_t get_epoch(tid_t mytid, tid_t tid, bool in_warp)
	{
		warp_level_cv *wcv;
		total_level_cv* tcv;
		switch (b.flags)
		{
		case MODE_BASIC:
			return in_warp ? b.warp : b.block;

		case MODE_WARP_CV:
			wcv = (warp_level_cv*)p.ptr;
			return in_warp ? wcv->epochs[tid & ~((1 << WARP_SHIFT) - 1)] : wcv->block;

		case MODE_TOTAL_CV:
			tcv = (total_level_cv*)p.ptr;
			uint32_t base = 0;
			if (mytid >> BLOCK_SHIFT == tid >> BLOCK_SHIFT)
				base = tcv->block;
			auto it = tcv->cv_epochs.find(tid);
			if (it != tcv->cv_epochs.end())
			{
				uint32_t value = it->second;
				if (value > base)
					return value;
			}

			return base;
		}
		SLIM_ASSERT(false);
		return 0;
	}

	FINLINE void release()
	{
		warp_level_cv* wcv;
		total_level_cv* tcv;

		switch (b.flags)
		{
		case MODE_BASIC:
			break;

		case MODE_WARP_CV:
			wcv = (warp_level_cv*)p.ptr;
			CHECK_AND_DELETE(wcv);
			break;

		case MODE_TOTAL_CV:
			tcv = (total_level_cv*)p.ptr;
			CHECK_AND_DELETE(tcv);
			break;
		}
	}

private:
	FINLINE void* get_ptr()
	{
		return (void*)(uintptr_t)p.ptr;
	}


private:
	enum { MODE_BASIC = 0 };
	enum { MODE_WARP_CV = 1 };
	enum { MODE_TOTAL_CV = 2 };

	enum { WARP_SIZE = 32 };

	struct basic
	{
		uint32_t warp;
		uint32_t block : 30;
		uint32_t flags : 2;
	};
	struct pointer
	{
		uint64_t ptr : 62;
		uint64_t flags : 2;
	};
	struct warp_level_cv
	{
		uint32_t  block;
		uint32_t  epochs[WARP_SIZE];
	};
	struct total_level_cv : public warp_level_cv
	{
		std::unordered_map<uint64_t, uint32_t> cv_epochs;
	};
private:
    
    uint32_t get_block()
    {
        if(b.flags == MODE_BASIC)
            return b.block;
        
        warp_level_cv* cv = (warp_level_cv*)p.ptr;
        return cv->block;
    }

	FINLINE warp_level_cv* get_warp_ptr()
	{
		return (warp_level_cv*)(uintptr_t)p.ptr;
	}

	void make_basic(uint32_t warp)
	{
        uint32_t block = get_block();
		release();
		b.flags = MODE_BASIC;
		b.warp = warp;
        b.block = block;
	}

	warp_level_cv *make_wcv()
	{
		warp_level_cv *wcv = new warp_level_cv();
		p.ptr = (uintptr_t)wcv;
		p.flags = MODE_WARP_CV;
		return wcv;
	}

	total_level_cv *make_tcv()
	{
		total_level_cv *tcv = new total_level_cv();
		p.ptr = (uintptr_t)tcv;
		p.flags = MODE_TOTAL_CV;
		return tcv;
	}

private:
	union
	{
		basic b;
		pointer p;
	};
};
