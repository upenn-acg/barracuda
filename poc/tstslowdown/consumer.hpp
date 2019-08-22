#pragma once
#include "libhookinfra.hpp"
#include "hooks.hpp"
#include "debug.h"
#include "slimfast.hpp"
#include "devarea.hpp"
#include <stdlib.h>
#include <algorithm>
#include <unistd.h>

class Consumer
{
public:
    enum { BUFFER_LEN = 16 * 1024 * 1024 };
    enum { SLEEP_TIME_US = 50000 };

    Consumer(const DeviceArea& area, int id) :
         _header(area.header(id))
        ,_start(DeviceArea::start(_header))
        ,_area_size(area.qbuf_size())
        ,_logged(0)
    {
        _buffer_len = std::min(_area_size, (size_t)BUFFER_LEN);
        VERIFY(ORIG(cudaStreamCreateWithFlags)(&_stream, cudaStreamNonBlocking));
        VERIFY(ORIG(cudaHostAlloc)((void**)&_buffer, sizeof(PCRecord) * _buffer_len, cudaHostAllocPortable)); 
    }

    ~Consumer()
    {
        VERIFY(ORIG(cudaStreamDestroy)(_stream));
        ORIG(cudaFreeHost)(_buffer);
    }

    void run(volatile bool* quit)
    {
        while(!*quit)
        {
            usleep(SLEEP_TIME_US);
            consume_available();
        }
        consume_available();
    }

    long get_logged()
    {
        return _logged;
    }

private:
    void consume_available()
    {
        //printf("Logged = %li.\n", _logged);
        for(;;)
        { 
            PCHeader pch;
            VERIFY(ORIG(cudaStreamSynchronize)(_stream));
            VERIFY(ORIG(cudaMemcpyAsync)(&pch, _header, 
                            sizeof(PCHeader), cudaMemcpyDeviceToHost, _stream));
            VERIFY(ORIG(cudaStreamSynchronize)(_stream));
    
            DEBUGONLY(printf("** Read Head at: %i\n", pch.read_head);)
            DEBUGONLY(printf("** Write Head at: %i\n", pch.write_head);)
            DEBUGONLY(printf("** Tail at: %i\n", pch.tail);)
            //printf("** Read Head at: %i\n", pch.read_head);
            //printf("** Write Head at: %i\n", pch.write_head);
            //printf("** Tail at: %i\n", pch.tail);


            int amount = std::min((ptrdiff_t)(pch.read_head - pch.tail), (ptrdiff_t)_buffer_len);
            if(amount == 0)
                break;
            DEBUGONLY(printf("** Reading %i records.\n", amount);)
            //printf("** Reading %i records.\n", amount);
 
            PCRecord* start = _start + (pch.tail % _area_size);
            VERIFY(ORIG(cudaMemcpyAsync)(_buffer, start,
                        sizeof(PCRecord) * amount, cudaMemcpyDeviceToHost, _stream));

            VERIFY(ORIG(cudaStreamSynchronize)(_stream));
           
            //printf("Logged before=%li, amount=%i\n", _logged, amount); 
            SlimFast::process_records(_buffer, amount);
            pch.tail += amount;
            _logged += amount;
        
            VERIFY(ORIG(cudaMemcpyAsync)(&_header->tail, &pch.tail, 
                        sizeof(pch.tail), cudaMemcpyHostToDevice, _stream));
            VERIFY(ORIG(cudaStreamSynchronize)(_stream));
        }    
        
    }

private:
    cudaStream_t _stream;
    PCHeader* _header;
    PCRecord* _start;
    PCRecord* _buffer;
    size_t    _area_size;
    size_t    _buffer_len;
    long _logged;
};

