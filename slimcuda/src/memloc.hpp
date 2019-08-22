#pragma once


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

    static memloc* alloc_memlocs(int count, bool shared)
    {
        memloc* result = new memloc[count];
        memset(result, 0, sizeof(memloc) * count);

        if(shared)
            for (int i = 0; i < count; ++i)
                result[i].is_shared = true;
        return result;
    }
};
    
class guard
{
public:
    enum {
        UNLOCKED = 0, LOCKED = 1
    };

    guard(memloc* loc) : _loc(loc)
    {
        //printf("Thread %lx: starting to locki: %p (state: %i)\n",  pthread_self(), &loc->lock, loc->lock);
        while (__sync_lock_test_and_set((uint32_t*)&loc->lock, LOCKED) != UNLOCKED)
        {
            volatile auto* lock = &loc->lock;
            while (*lock);
        }
        //printf("Thread %lx: locked: %p\n",  pthread_self(), &loc->lock);
    }

    ~guard()
    {
        __sync_lock_release((uint32_t*)&_loc->lock);
        //printf("Thread %lx: unlocked: %p\n",  pthread_self(), &_loc->lock);
    }
private:
    memloc* _loc;
};

