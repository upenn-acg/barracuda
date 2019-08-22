#pragma once

#include "epoch.hpp"
#include "thread_cv.hpp"
#include "compiler.hpp"
#include <climits>

struct active_epoch
{
	uint32_t mask;
	Epoch epoch;
	thread_cv cv;

	enum TestResult { EARLIER, GENERAL_RACE, BLOCK_RACE, INTRA_WARP_RACE, DIVERGENCE_RACE };
	TestResult get_fail_type(uint64_t my_tid, uint64_t other_tid)
	{
		if (((my_tid >> WARP_SHIFT) == (other_tid >> WARP_SHIFT)))
			return DIVERGENCE_RACE;

		if (((my_tid >> BLOCK_SHIFT) == (other_tid >> BLOCK_SHIFT)))
			return BLOCK_RACE;

		return GENERAL_RACE;
	}

	TestResult test_epoch(uint64_t my_tid, uint64_t other_tid, uint32_t mem_epoch)
	{
		static const int WARP_MASK = 31; // XXXXXX XXX
		
//		SLIM_ASSERT( my_tid != other_tid );
		TestResult fail_type = GENERAL_RACE;
		if (((my_tid >> WARP_SHIFT) == (other_tid >> WARP_SHIFT)))
		{
			if (mask & (1 << (other_tid & WARP_MASK)))
				return (mem_epoch > epoch) ? INTRA_WARP_RACE : EARLIER;

			fail_type = DIVERGENCE_RACE;
		}

		if (((my_tid >> BLOCK_SHIFT) == (other_tid >> BLOCK_SHIFT)))
			fail_type = BLOCK_RACE;

		uint32_t cv_epoch = cv.get_epoch(my_tid, other_tid, false);
		if (mem_epoch > cv_epoch)
			return fail_type;
		return EARLIER;
	}
};

class warp_cv // XXX: add pool
{
public:
	warp_cv() 
	{
		memset(_data, 0, sizeof(_data));
		SLIM_ASSERT((uintptr_t)&((active_epoch*)nullptr)->epoch == sizeof(uint32_t));
		SLIM_ASSERT((uintptr_t)&((active_epoch*)nullptr)->cv == (uintptr_t)&((extra_data*)nullptr)->fill);
		//SLIM_ASSERT(sizeof(active_epoch) == 16);
	}

	~warp_cv()
	{
		extra_data* extra = get_extra();
        if(extra != nullptr) {
            free((active_epoch *)(extra->data & ~EXTRA_PTR_MASK));
            extra->data = 0;
        }
	}

	FINLINE void init(uint32_t active_mask)
	{
		SLIM_ASSERT(active_mask != 0);
		_data[0].mask = active_mask;
		_data[0].epoch.inc();
		_last = &_data[0];
	}

	FINLINE active_epoch* step(uint32_t active_mask)
	{
		SLIM_ASSERT(_data[0].mask != 0);
		SLIM_ASSERT(active_mask != 0);
		SLIM_ASSERT((active_mask & _data[0].mask) == active_mask);

		if (_last->mask != active_mask)
			_last = realize_stack(active_mask);
		_last->epoch.inc();
		return _last;
	}

private:
	enum { MAX_LOCAL = 3 };
	enum { MAX_EXTRA = 65 };

	const uint64_t EXTRA_PTR_MASK = 0x8000000000000000ULL;



	struct extra_data
	{
		uintptr_t data;
        // let's make 'fill' the count of elements 
		int fill;
	};

	FINLINE extra_data* get_extra() const
	{
		extra_data* ed = (extra_data*)&_data[MAX_LOCAL - 1];
		if ((ed->data & EXTRA_PTR_MASK) != 0)
			return ed;
		return nullptr;
	}

	active_epoch* realize_stack(uint32_t active_mask)
	{
		extra_data* extra = get_extra();
		const bool no_extra = (extra == nullptr);
		int fill = 0;
		active_epoch* data = _data;
        SLIM_ASSERT(data[0].mask != 0);
		if (no_extra)
		{
			if (_data[1].mask != 0)
				fill = (_data[MAX_LOCAL - 1].mask != 0) ? 3 : 2;
		}
		else
		{
			fill = extra->fill;
			data = (active_epoch*)(extra->data & ~EXTRA_PTR_MASK);
		}

		active_epoch* src = data;
		active_epoch* dst = data;
		active_epoch* merge_to = nullptr;
		active_epoch* last_parent = data;
        // is fill a count? 
		for (int i = 0; i < fill; ++i, ++ src, ++ dst)
		{
			const uint32_t dim = src->mask;
			const uint32_t intersection = dim & active_mask;
			if (src != dst)
			{
				dst->cv.release();
				memcpy(dst, src, sizeof(*dst));
			}

			if (intersection == dim && merge_to == nullptr)
			{
				SLIM_ASSERT(merge_to == nullptr);
				dst->mask |= active_mask;
				dst->epoch.max_of(last_parent->epoch);
				dst->cv.max_of(last_parent->cv);
				last_parent = merge_to = dst;
				continue;
			}
			if (intersection == active_mask)
			{
				last_parent = dst;
				continue;
			}
			if (intersection != 0)
			{
				if (merge_to == nullptr)
				{
					merge_to = dst;
					merge_to->mask |= active_mask;
					merge_to->epoch.max_of(last_parent->epoch);
				}
				else
				{
					merge_to->mask |= active_mask | dim;
					merge_to->epoch.max_of(src->epoch);
					dst->mask = 0;
					dst -= 1;
				}
				merge_to->cv.max_of(src->cv);
				continue;
			}
			for (; src < dst; ++src)
				src->cv.release();
		}
		const int count = (int)(dst - &data[0]);
		if (merge_to != nullptr)
		{
			if (data != _data && count <= 2)
			{
				memcpy(_data, data, count * sizeof(data[0]));
				merge_to = &_data[merge_to - data];
				free(data); // XXX: pooling!
			}
			return merge_to;
		}

		if (no_extra)
		{
			if (count >= MAX_LOCAL)
			{
                extra = create_extra();
                extra->fill = count + 1;
			    data = (active_epoch*)(extra->data & ~EXTRA_PTR_MASK);
			}
		}
        else
    		extra->fill = count + 1;

		data[count].mask = active_mask;
		data[count].epoch = last_parent->epoch;
		data[count].cv.diverge_from(last_parent->cv, active_mask, last_parent->epoch);
        SLIM_ASSERT(data[0].mask != 0);
		return &data[count];
	}

    extra_data* create_extra()
    {
		extra_data* extra = (extra_data*)&_data[MAX_LOCAL - 1];
		// XXX: pooling!
		active_epoch* data = (active_epoch*)malloc(sizeof(data[0]) * MAX_EXTRA);
        memset(data, 0, sizeof(data[0]) * MAX_EXTRA);
        // we should avoid copying the last element, as it's extra_data!
		memcpy(data, _data, sizeof(data[0]) * (MAX_LOCAL - 1));
		extra->data = ((uintptr_t)data) | EXTRA_PTR_MASK;
		return extra;
    }

	active_epoch _data[MAX_LOCAL];
	active_epoch* _last;
};
