#pragma once

#include <tbb/concurrent_unordered_map.h>
#include "thread_cv.hpp"

typedef int bid_t;
typedef tbb::concurrent_unordered_map< bid_t, thread_cv> perblock_vcs;

// returns true iff. this method initialized an entry 
template< typename MapType, typename KeyType, typename ValueType>
bool try_initialize(MapType & m, KeyType k,  ValueType & v)
{
	bool inserted = false;
	if(m.count(k) == 0)
	{
		m[k] = v;
		inserted = true;
	}	
	SLIM_ASSERT( m.count( k ) > 0 );
	return inserted;
}

class sync_vcs {
public:
	sync_vcs(): gl_vcs(), pb_vcs()
	{
		
	}
	~sync_vcs()
	{
	}
	void max_gl_vc( void * x,  thread_cv & other_vc)
	{
		if(! try_initialize( gl_vcs, x, other_vc) )
		{
			gl_vcs[x].max_of( other_vc );
		}
	}
	// NB: return the max of all vcs ( including both gl_vcs and pb_vcs)
	thread_cv get_global_max_vc( void * x)
	{
		thread_cv t;
		perblock_vcs p;
		try_initialize( gl_vcs, x, t);
		try_initialize( pb_vcs, x, p);
		thread_cv max_cv = gl_vcs[x];
		for(auto it = pb_vcs[x].begin(), end = pb_vcs[x].end(); it != end; it++)
		{
			max_cv.max_of( it-> second );
		}
		return max_cv;
		 
	}
	void max_pb_vc( void * x, bid_t blk,  thread_cv & other_vc)
	{
		perblock_vcs p;
		try_initialize( pb_vcs, x, p);
		if(! try_initialize( pb_vcs[x], blk, other_vc) )
		{
			pb_vcs[x][blk].max_of( other_vc );
		}
	} 
	//NB: returns the union of gl_vcs[x] and pb_vcs[x][blk]
	thread_cv get_block_vc( void * x, bid_t blk)
	{
		thread_cv t;
		try_initialize( gl_vcs, x, t);
		thread_cv block_cv = gl_vcs[x];
		perblock_vcs p;
		try_initialize( pb_vcs, x, p);
		auto it = pb_vcs[x].find(blk);
		if(it != pb_vcs[x].end())
		{
			block_cv.max_of( it-> second );
		} 
		return block_cv;
	}
private:
	tbb::concurrent_unordered_map< void *, thread_cv> gl_vcs;
	tbb::concurrent_unordered_map< void *, perblock_vcs> pb_vcs;
};
