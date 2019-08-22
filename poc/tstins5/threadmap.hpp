#pragma once

#include "CV.hpp"
#include "threadblocks.hpp"
#include "streammap.hpp"

#include <map>

class ThreadMap
{
public:
    ThreadMap()
    {
    }

    ~ThreadMap()
    {
    }

    CV* get_thread_cv(tid_t t)
    {
    }

    void create_stream(unsigned short stream_id)
    {
    }

    void destroy_stream(unsigned short stream_id)
    {
    }

    CV* create_thread_blocks(unsigned short stream_id, dim3 grid_dim)
    {
    }

    void destroy_thread_blocks(unsigned short stream_id)
    {
    }

private:
    typedef std::map<blk_t, StreamMap*> MapType;
    MapType _map;
}
