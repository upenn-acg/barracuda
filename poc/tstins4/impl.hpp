#pragma once
#include <memory>
#include "compiler.hpp"
#include "config.h"
#include "protocol.hpp"
#include "fatbinary.hpp"
#include "logger.hpp"

class Impl
{
public:
    Impl() :
        _unloading(false), _is_init(false)
    {
    }

    ~Impl()
    {
        _unloading = true;
        stop_logger();
    }

    void** register_binary(void* fatCubin)
    {
        _Hook_BASE::load_hooks(RTLD_NEXT);
        
        _fatbin = std::make_unique<FatBinary>(fatCubin);
        void* newfatCubin = _fatbin->instrument(); 

        void** res = ORIG(__cudaRegisterFatBinary)(newfatCubin);
    	fprintf(stderr, "cudaRegisterFatBinary(%p)=%p [%p] old=%p.\n", newfatCubin, res, (res == NULL)?0 : *res, fatCubin);

        ORIG(__cudaRegisterFunction)(res, &_device_area_init_handle,
                                (char*)"_autogen_cb_devinit", "_autogen_cb_devinit", 0xffffffff, 0, 0, 0, 0, 0); // XXX: const!
        return res;
    }

    void ensure_configured()
    {
        // XXX: lock!
        if(_unloading || _is_init) return;

    	size_t free = 0, total = 0;
    	VERIFY(ORIG(cudaMemGetInfo)(&free, &total));
    
    	size_t size = total / (1 + SHADOW_MEMORY_RATIO);
    	printf("Free=%lx total=%lx size=%lx\n", free, total, size);
    
        void* base;
    	VERIFY(ORIG(cudaMalloc)(&base, size));
    	printf("Shadow alloced at %p\n", base);
   
        _dev_area = DeviceArea(base, size, 1); // XXX: numq not 1

        VERIFY(ORIG(cudaDeviceSynchronize)());
        printf("library init.\n");
        _is_init = true;
    
        run_logger();
    }

    void before_reset()
    {
        // XXX: lock
        stop_logger();
    }

    void after_reset()
    {
        ensure_configured();
    }

private:

    void run_logger()
    {
        if(_unloading) 
            return;
        fprintf(stderr, "init_logger().\n");

        printf("Creating logger.\n");
        _logger = std::make_unique<Logger>(_dev_area);
    }

    void stop_logger()
    {
        _logger.reset();
    }

    void init_device_area()
    {
        dim3 d;
        d.x = d.y = d.z = 1;
        VERIFY(ORIG(cudaConfigureCall)(d, d, 0, 0));
        // XXX: calculate instead of constant
        int value0 = _dev_area.numq();
        VERIFY(ORIG(cudaSetupArgument)(&value0, sizeof(int), 0));
        void* value8 = _dev_area.base();
        VERIFY(ORIG(cudaSetupArgument)(&value8, sizeof(void*), 8));
        size_t value16 = _dev_area.size();
        VERIFY(ORIG(cudaSetupArgument)(&value16, sizeof(void*), 16));

        ORIG(cudaLaunch)(&_device_area_init_handle);

    }

private:
    char _device_area_init_handle;
    std::unique_ptr<Logger> _logger;
    std::unique_ptr<FatBinary> _fatbin;
    bool _unloading;
    bool _is_init;
    DeviceArea _dev_area;
};

