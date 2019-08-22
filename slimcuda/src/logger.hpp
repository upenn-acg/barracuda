#pragma once

#include <pthread.h>
#include "consumer.hpp"
#include "devarea.hpp"
#include "protocol.hpp"

class Logger
{
public:
    Logger(DeviceArea area) : _dev_area(area), _quit(false), _logged(0)
    {
        _tdata = new ThreadData[_dev_area.numq()];
        for(int i = 0; i < _dev_area.numq(); ++ i)
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

    ~Logger()
    {
        _quit = true;
        __sync_synchronize();

        for(int i = 0; i < _dev_area.numq(); ++ i)
        {
            SLIM_DPRINTF(VL_VERBOSE, "Joining %i.", i);
            pthread_join(_tdata[i].pth, NULL);
            _logged += _tdata[i].logged;
        }
        delete[] _tdata;
            
   
        SLIM_DPRINTF(VL_VERBOSE, "Joined.\n");
        ORIG(cudaDeviceReset)();
//        printf("@Logged:\t%li\n", _logged);
    }

private:
    static void* consumer_thread(void* arg)
    {
        ThreadData* tdata = (ThreadData*)arg;
        SLIM_DPRINTF(VL_VERBOSE, "Consumer thread %i starting.", tdata->id);
        Consumer consumer(tdata->impl->_dev_area, tdata->id);
        consumer.run(&tdata->impl->_quit);
        tdata->logged = consumer.get_logged();
        SLIM_DPRINTF(VL_VERBOSE, "Consumer thread %i done.", tdata->id);
        return NULL;
    }

private:
    struct ThreadData
    {
        pthread_t pth;
        Logger* impl;
        int id;
        int logged;
    };
    
    ThreadData* _tdata;
    DeviceArea _dev_area;
    volatile bool _quit;
    long _logged;
};


