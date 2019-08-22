
#pragma once
#include "libhookinfra.hpp"
#include "hooks.hpp"
#include "debug.h"
#include "hostside.hpp"
#include "devarea.hpp"
#include <stdlib.h>
#include <algorithm>
#include "ptx_stub.h"

#include <unistd.h>

char* dev_area_ptr = NULL;
double regclock_now();

class Consumer
{
public:
    enum { BUFFER_LEN = 16 * 1024 * 1024 };
    enum { SLEEP_TIME_US = 50000 };

    Consumer()
    {
        VERIFY(ORIG(cudaStreamCreateWithFlags)(&_stream, cudaStreamNonBlocking));
        VERIFY(ORIG(cudaHostAlloc)((void**)&dev_area, sizeof(DeviceArea), cudaHostAllocPortable)); 
    }

    ~Consumer()
    {
    }

    void run()
    {
        for(;;)
        {
            sleep(1);
            print_status();
        }
    }

    void print_status()
    {
        printf("status===========================\n");
        long total_count = 0, total_ins = 0;
        int val = cudaMemcpyFromSymbolAsync(dev_area, &dev_area_ptr, sizeof(*dev_area), 0, cudaMemcpyDeviceToHost, _stream);
        printf("Ret: %i\n", val);
        for(int i = 0; i < 64; ++ i)
        {
            long count = dev_area->get_count(i);
            long ins = dev_area->get_ins(i);
            printf("(%i) %li %li\n", i, count, ins);
            total_ins += ins;
            total_count += count;
        }
        printf("(total) %li %li = %lg\n", total_count, total_ins, ((double)total_ins/(double)total_count));
        double now = regclock_now();
        printf("(rate) %lg %lg\n", total_count / now, total_ins / now);
    }

private:
    cudaStream_t _stream;
    DeviceArea *dev_area;
};

