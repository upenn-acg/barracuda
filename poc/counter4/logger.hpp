#pragma once

#include <pthread.h>
#include "consumer.hpp"
#include "devarea.hpp"
#include "protocol.hpp"

class Logger
{
public:
    Logger() 
    {
        pthread_t pth;
        if(0 != pthread_create(&pth, NULL, consumer_thread, &consumer))
        {
            fprintf(stderr, "Failed creating consumer thread!\n");
            exit(1);
        }
    }

    ~Logger()
    {
        __sync_synchronize();
        print();

        DEBUGONLY(printf("Joined.\n");)
        ORIG(cudaDeviceReset)();
//        printf("@Logged:\t%li\n", _logged);
    }

    void print()
    {
        consumer.print_status();
    }

private:
    static void* consumer_thread(void* _consumer)
    {
        Consumer* consumer = (Consumer*)_consumer;
        consumer->run();
        return NULL;
    }

private:
        Consumer consumer;
};


