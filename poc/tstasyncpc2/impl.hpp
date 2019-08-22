#pragma once

#include <pthread.h>
#include "debug.h"

#define ALIGN(type, ptr, alignment) (type*)(((uintptr_t)(ptr) + (alignment) - 1) & ~((alignment) - 1))

__device__ void* __gHooklib_shadow_base = NULL;
__device__ PCHeader* pcheader = NULL;
__device__ PCRecord* pcstart = NULL;
__device__ size_t    pcbuffer_size = 0;

PCHeader* dev_pcheader = NULL;
PCRecord* dev_pcstart = NULL;
size_t    dev_pcbuffer_size = 0;

__global__ void __autogen_cb_devinit(void* shadow_base, size_t shadow_size)
{
    __gHooklib_shadow_base = shadow_base;
    pcheader = (PCHeader*)shadow_base;
    pcbuffer_size = (shadow_size - 1- sizeof(PCHeader)) / sizeof(PCRecord);
//    pcbuffer_size = 2;
    pcstart = ALIGN(PCRecord, pcheader + 1, sizeof(void*));
    pcheader->read_head = 0;
    pcheader->write_head = 0;
    pcheader->tail = 0;
    DEBUGONLY(printf("PC Buffer initialized, size is: %i\n", pcbuffer_size);)
    DEBUGONLY(printf("PCStart: %p\n", pcstart);)
    __threadfence_system();
}

static void __autogen_cb(void* shadow_base, size_t shadow_size)
{
    // XXX: remove duplication
    dev_pcheader = (PCHeader*)shadow_base;
    dev_pcstart = ALIGN(PCRecord, dev_pcheader + 1, sizeof(void*));
    dev_pcbuffer_size = (shadow_size - 1 - sizeof(PCHeader)) / sizeof(PCRecord);
    //dev_pcbuffer_size = 2;
    __autogen_cb_devinit<<<1,1>>>(shadow_base, shadow_size);   
}

class Impl
{
public:
    Impl(long expected_logs) : _quit(false), _expected_logs(expected_logs)
    {
	    __libhook_register_init_cb(__autogen_cb);
        cudaDeviceSynchronize(); // XXX force library init
    
        if(0 != pthread_create(&_pth, NULL, consumer_thread, (void*)this))
        {
            fprintf(stderr, "Failed creating consumer thread!\n");
            exit(1);
        }
    }

    ~Impl()
    {
        _quit = true;
        __sync_synchronize();

        DEBUGONLY(printf("Joining.\n");)
        pthread_join(_pth, NULL);
   
        DEBUGONLY(printf("Joined.\n");)
        cudaDeviceReset();
        printf("@Missed:\t%s\n", (_logged != _expected_logs) ? "true" : "false");
        printf("@Logged:\t%li\n", _logged);
        printf("@Expected:\t%i\n", _expected_logs);
    }

private:
    static void* consumer_thread(void* arg)
    {
        Impl* impl = (Impl*)arg;
        impl->consumer_thread_imp();
        return NULL;
    }

    void consumer_thread_imp()
    {
        DEBUGONLY(printf("** Consumer thread starting.\n");) 
        Consumer consumer(dev_pcheader, dev_pcstart, dev_pcbuffer_size);
        consumer.run(&_quit);
        _logged = consumer.get_logged();
    }

private:
    volatile bool _quit;
    pthread_t _pth;
    long _logged;
    long _expected_logs;
};


