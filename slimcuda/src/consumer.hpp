#pragma once
#include "libhookinfra.hpp"
#include "hooks.hpp"
#include "debug.h"
#include "hostdetector.hpp"
#include "devarea.hpp"
#include <stdlib.h>
#include <algorithm>
#include <unistd.h>

class Consumer
{
public:
    enum { BUFFER_LEN = 16 * 1024 * 1024 };
    enum { SLEEP_TIME_US = 5000 };

    Consumer(const DeviceArea& area, int id) :
         _id(id)
        , _header(area.header(id))
        ,_start(DeviceArea::start(_header))
        ,_area_size(area.qbuf_size())
        ,_logged(0)
    {
        _buffer_len = std::min(_area_size, (size_t)BUFFER_LEN);
        _end = _start + _buffer_len;
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
            HostDetector::check_for_pause();

            PCHeader pch;
            VERIFY(ORIG(cudaStreamSynchronize)(_stream));
            VERIFY(ORIG(cudaMemcpyAsync)(&pch, _header, 
                            sizeof(PCHeader), cudaMemcpyDeviceToHost, _stream));
            VERIFY(ORIG(cudaStreamSynchronize)(_stream));
    
            SLIM_DPRINTF(VL_VERBOSE, "Queue %i: RWT=(%i,%i,%i)", _id, pch.read_head, pch.write_head, pch.tail);


            PCRecord* start = _start + (pch.tail % _area_size);
            int amount = std::min((ptrdiff_t)(pch.read_head - pch.tail), _end - start);
            if(amount == 0)
                break;
            SLIM_DPRINTF(VL_VERBOSE, "Queue %i: Reading %i records.", _id, amount);
 
            VERIFY(ORIG(cudaMemcpyAsync)(_buffer, start,
                        sizeof(PCRecord) * amount, cudaMemcpyDeviceToHost, _stream));

            VERIFY(ORIG(cudaStreamSynchronize)(_stream));
           
            HostDetector::process_records(_buffer, amount);
            pch.tail += amount;
            _logged += amount;
        
            VERIFY(ORIG(cudaMemcpyAsync)(&_header->tail, &pch.tail, 
                        sizeof(pch.tail), cudaMemcpyHostToDevice, _stream));
            VERIFY(ORIG(cudaStreamSynchronize)(_stream));
        }    
        
    }

private:
    int _id;
    cudaStream_t _stream;
    PCHeader* _header;
    PCRecord* _start;
    PCRecord* _end;
    PCRecord* _buffer;
    size_t    _area_size;
    size_t    _buffer_len;
    long _logged;
};

