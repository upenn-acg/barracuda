#pragma once

static const int MAX_STREAMS = 4096;

class stream_threads_map
{
public: 
    stream_threads_map() : _cvs(nullptr), _bids(nullptr)
    {
    }

    ~stream_threads_map()
    { 
        destroy();
    }

    void reset(unsigned int block_count, unsigned int block_size)
    {
        destroy();
        _blk_size = block_size;
        _blk_count = block_count;
        int amount = (_blk_size * _blk_count); // + (1 << WARP_SHIFT) - 1) >> WARP_SHIFT;
        SLIM_DEBUGONLY(_size = amount;)
        _cvs = new warp_cv[amount];
        // bug: 1) wrong number of bytes; 2) would set the const field EXTRA_PTR_MASK to 0
        //memset(_cvs, 0, sizeof(_cvs) * amount);
        _bids = new uint32_t[_blk_count];
        memset(_bids, 0, sizeof(_bids[0]) * _blk_count);
        SLIM_DPRINTF(VL_VERBOSE, "Creating %i wap CVs. New _cvs=%p _bids=%p [blk_count=%i]", amount, _cvs, _bids, _blk_count);
    }
        
    void destroy()
    {
        if(_cvs != nullptr)
        {
            SLIM_DPRINTF(VL_VERBOSE, "Destroying warp CVs. _cvs=%p bids=%p", _cvs, _bids);
            delete[] _cvs;
            delete[] _bids;
            _cvs = nullptr;
            _bids = nullptr;
        }
    }

    inline warp_cv& cv(uint64_t wid)
    {
        int blk = (int)(wid >> BLOCK_SHIFT);
        if(blk >= _blk_count)
        {
            printf("\n\n*** ERROR! Invalid WID: %lx - blk=%x blk_count=%x\n\n", wid, blk, _blk_count);
            exit(-1);
        }
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
    /* stores max epoch of each block*/
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

    void reset(int stream, unsigned int block_count, unsigned int block_size)
    {
        stream = 0; // XXX: single stream
        SLIM_ASSERT(stream < MAX_STREAMS);
        _st[stream].reset(block_count, block_size);
        SLIM_DPRINTF(VL_VERBOSE, "Resetting new _stm for stream %i.", stream);
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

