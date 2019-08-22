#pragma once

#include <map>
#include <stdint.h>

typedef uint32_t blk_t;
typedef uint64_t tid_t;
typedef uint32_t lc_t;

class CV
{
public:
    static CV* make_cv()
    {
        return new CV();
    }

    ~CV()
    {
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

private:
    CV()
    {
    }

    std::map<tid_t, lc_t> _lcs; // XXX: make two level?
};


