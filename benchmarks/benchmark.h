#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <time.h>
#include <unistd.h>

//////////////////////////////////////////////////
class Benchmark
{
public:
    void start_total()
    {
        clock_gettime(CLOCK_MONOTONIC, &_total_start);
    }

    void end_total()
    {
        struct timespec total_end;
        clock_gettime(CLOCK_MONOTONIC, &total_end);
        long time;
        if(total_end.tv_nsec >= _total_start.tv_nsec)
        {
            time = (total_end.tv_sec - _total_start.tv_sec) * 1000000000L;
            time += (total_end.tv_nsec - _total_start.tv_nsec);
        }
        else
        {
            time = (total_end.tv_sec - _total_start.tv_sec - 1) * 1000000000L;
            time += (1000000000 + total_end.tv_nsec - _total_start.tv_nsec);
        }
        _total_time = ((double)time / 1000000.0);
    }
    
    void start_kernel()
    {
        cudaEventCreate(&_start);
        cudaEventCreate(&_stop);
        cudaEventRecord(_start);
    }

    void end_kernel()
    {
        cudaEventRecord(_stop);
        cudaEventSynchronize(_stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, _start, _stop);
        _kernel_time += milliseconds;
        cudaEventDestroy(_start);
        cudaEventDestroy(_stop);
    }

    Benchmark() : _success(true), _total_time(0), _kernel_time(0)
    {
    }

    ~Benchmark()
    {
        char linkname[1024] = "<unknown>";
        int len = readlink("/proc/self/exe", linkname, sizeof(linkname));
        const char* name = strrchr(linkname, '/');
        if(name == NULL) name = linkname; else name += 1;
        const char* enabled = (getenv("SC_NOFUNCS") != NULL) ? "DISABLED" : "ENABLED";
        const char* status = _success ? "SUCCESS" : "FAILURE";
        fflush(stdout);
        fflush(stderr);
      	fprintf(stderr, "TIME TOTAL %s %lf %s %s\nTIME GPU %s %lf %s %s\n", 
            name, _total_time, enabled, status, 
            name, _kernel_time == 0 ? 1 : _kernel_time, enabled, status);
        fflush(stdout);
        fflush(stderr);
    }

    void fail()
    {
        _success = false;
    }

private:
    struct timespec _total_start;
    cudaEvent_t _start, _stop;
    double _total_time, _kernel_time;
    bool _success;
};

Benchmark BENCHMARK;


