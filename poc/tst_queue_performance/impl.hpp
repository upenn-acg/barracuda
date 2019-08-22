#pragma once

#include <pthread.h>
#include "debug.h"

#define PCHEADER(base, idx, pcqbuffer_size) (PCHeader*)(((char*)base) + idx * (HEADER_SIZE + pcqbuffer_size * sizeof(PCRecord)));
#define PCSTART(hdr) ((PCRecord*)(((char*)hdr) + HEADER_SIZE))

#define ALIGN(type, ptr, alignment) (type*)(((uintptr_t)(ptr) + (alignment) - 1) & ~((alignment) - 1))

__device__ void* __gHooklib_shadow_base = NULL;
__device__ unsigned char *pcqheader;
__device__ size_t    pcqbuffer_size = 0;
__device__ unsigned int pcqnum;

void* dev_pcqheader = NULL;
size_t    dev_pcqbuffer_size = 0;
int       dev_numq = 0;
const int HEADER_SIZE = 64; // max(sizeof(PCHeader), alignment(PCRecord));

__global__ void __autogen_cb_devinit(int numq, void* shadow_base, size_t shadow_size)
{
    __gHooklib_shadow_base = shadow_base;

    pcqbuffer_size = (shadow_size - 1- numq * HEADER_SIZE) / sizeof(PCRecord) / numq;
    pcqnum = numq;
    //pcqbuffer_size = (shadow_size - 1- numq * 64) / sizeof(PCRecord) / numq;
    pcqheader = (unsigned char*)shadow_base;

    for(int i = 0; i < numq; ++ i)
    {
        PCHeader* pcheader = PCHEADER(pcqheader, i, pcqbuffer_size);
        pcheader->read_head = 0;
        pcheader->write_head = 0;
        pcheader->tail = 0;
        DEBUGONLY(printf("PC Buffer %i initialized, at %p, size is: %i\n", i, pcheader, pcqbuffer_size);)
    }

    __threadfence_system();
}

static void __autogen_cb(void* shadow_base, size_t shadow_size)
{
    // XXX: remove duplication
    dev_pcqheader = shadow_base;
    dev_pcqbuffer_size = (shadow_size - 1- dev_numq * HEADER_SIZE) / sizeof(PCRecord) / dev_numq;
    __autogen_cb_devinit<<<1,1>>>(dev_numq, shadow_base, shadow_size);   
}

class Impl
{
public:
    Impl(int numq, long expected_logs) : _numq(numq), _quit(false), _expected_logs(expected_logs)
    {
        dev_numq = numq;
	    __libhook_register_init_cb(__autogen_cb);
        cudaDeviceSynchronize(); // XXX force library init
   
        _tdata = new ThreadData[numq];
        for(int i = 0; i < numq; ++ i)
        {
            ThreadData* td = &_tdata[i];
            td->impl = this;
            td->id = i;
            if(0 != pthread_create(&td->pth, NULL, consumer_thread, (void*)td))
            {
                fprintf(stderr, "Failed creating consumer thread!\n");
                exit(1);
            }
        }
    }

    ~Impl()
    {
        _quit = true;
        __sync_synchronize();

        for(int i = 0; i < _numq; ++ i)
        {
            DEBUGONLY(printf("Joining %i.\n", i);)
            pthread_join(_tdata[i].pth, NULL);
            _logged += _tdata[i].logged;
        }
        delete[] _tdata;
            
   
        DEBUGONLY(printf("Joined.\n");)
        cudaDeviceReset();
        printf("@Missed:\t%s\n", (_logged != _expected_logs) ? "true" : "false");
        printf("@Logged:\t%li\n", _logged);
        printf("@Expected:\t%li\n", _expected_logs);
    }

private:
    static void* consumer_thread(void* arg)
    {
        ThreadData* tdata = (ThreadData*)arg;
        DEBUGONLY(printf("** Consumer thread %i starting.\n", tdata->id);) 
        PCHeader* pcheader = PCHEADER(dev_pcqheader, tdata->id, dev_pcqbuffer_size);
        PCRecord* pcstart = PCSTART(pcheader);
        Consumer consumer(pcheader, pcstart, dev_pcqbuffer_size);
        consumer.run(&tdata->impl->_quit);
        tdata->logged = consumer.get_logged();
        DEBUGONLY(printf("** Consumer thread %i done.\n", tdata->id);) 
        return NULL;
    }

private:
    struct ThreadData
    {
        pthread_t pth;
        Impl* impl;
        int id;
        int logged;
    };
    
    ThreadData* _tdata;
    int _numq;
    volatile bool _quit;
    long _logged;
    long _expected_logs;
};


