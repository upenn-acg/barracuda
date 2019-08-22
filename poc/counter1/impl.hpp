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
        _disabled = getenv("SC_DISABLED") != NULL;
    }

    ~Impl()
    {
        _unloading = true;
        stop_logger();
        SlimFast::instance().print();
    }

    void** register_binary(void* fatCubin)
    {
        _Hook_BASE::load_hooks(RTLD_NEXT);
        
        _fatbin = std::make_unique<FatBinary>(fatCubin);
        void* newfatCubin = _fatbin->instrument(_disabled); 

        void** res = ORIG(__cudaRegisterFatBinary)(newfatCubin);
//    	fprintf(stderr, "cudaRegisterFatBinary(%p)=%p [%p] old=%p.\n", newfatCubin, res, (res == NULL)?0 : *res, fatCubin);

        ORIG(__cudaRegisterFunction)(res, &_device_area_init_handle,
                                (char*)NAMEOF_INIT_FUNCTION_NAME, NAMEOF_INIT_FUNCTION_NAME, 0xffffffff, 0, 0, 0, 0, 0); 
//        fprintf(stderr, "cudaRegisterFunction(%p, %p, %s, %s, %i)\n", res, &_device_area_init_handle, 
//                                (char*)NAMEOF_INIT_FUNCTION_NAME, NAMEOF_INIT_FUNCTION_NAME, 0xffffffff);
        return res;
    }

    void ensure_configured()
    {
        // XXX: lock!
        if(_unloading || _is_init || _disabled) return;

    	size_t free = 0, total = 0;
    	VERIFY(ORIG(cudaMemGetInfo)(&free, &total));
    
    	size_t size = total / (1 + SHADOW_MEMORY_RATIO);
//    	printf("Free=%lx total=%lx size=%lx\n", free, total, size);
    
        void* base;
    	VERIFY(ORIG(cudaMalloc)(&base, size));
//    	printf("Shadow alloced at %p\n", base);
   
        _dev_area = DeviceArea(base, size, 16); // XXX: numq not 1
//        _dev_area = DeviceArea(base, size, 1); // XXX: numq not 1
        VERIFY(ORIG(cudaDeviceSynchronize)());
//        printf("library init.\n");
        _is_init = true;
  
//        printf("Running %s\n", NAMEOF_INIT_FUNCTION_NAME); 
        init_device_area();
        run_logger();
    }

    const void* get_hnd()
    {
        return &_device_area_init_handle;
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
//        fprintf(stderr, "init_logger().\n");

//        printf("Creating logger.\n");
        _logger = std::make_unique<Logger>(_dev_area);
    }

    void stop_logger()
    {
        _logger.reset();
    }

    void init_device_area()
    {
        dim3 g, d;
        d.x = d.y = d.z = 1;
        g.x = g.y = g.z = 1;
        VERIFY(ORIG(cudaConfigureCall)(g, d, 0, 0));
        // XXX: calculate instead of constant
        VERIFY(ORIG(cudaSetupArgument)(&_dev_area, sizeof(_dev_area), 0));
        VERIFY(ORIG(cudaLaunch)(&_device_area_init_handle));
        VERIFY(ORIG(cudaDeviceSynchronize)());
        printf("Device area init.\n");
    }

private:
    char _device_area_init_handle;
    std::unique_ptr<Logger> _logger;
    std::unique_ptr<FatBinary> _fatbin;
    bool _unloading;
    bool _is_init;
    bool _disabled;
    DeviceArea _dev_area;
};

