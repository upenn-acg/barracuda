#pragma once
#include <memory>
#include "compiler.hpp"
#include "env.hpp"
#include "config.h"
#include "protocol.hpp"
#include "fatbinary.hpp"
#include "logger.hpp"

size_t LH_MY_MEMORY = 0;
off_t BASE_MEM = 0;

class Impl
{
public:
    Impl() :
        _unloading(false), _is_init(false)
    {
        _device_area_init_handle_ptr = nullptr;
        _disabled = Env::is_disabled();
    }

    ~Impl()
    {
        _unloading = true;
        stop_logger();
		HostDetector::INSTANCE.print_hazards();
        HostDetector::print_memory();
    }

    void** register_binary(void* fatCubin)
    {
        _Hook_BASE::load_hooks(RTLD_NEXT);

        if(_disabled)
        {
            return ORIG(__cudaRegisterFatBinary)(fatCubin);
        }
        
        _fatbin = std::make_unique<FatBinary>(fatCubin);
        void* newfatCubin = _fatbin->instrument(Env::compile_time_ptx_instrumentation()); 
        void** res = ORIG(__cudaRegisterFatBinary)(newfatCubin);
    	SLIM_DPRINTF(VL_DEBUG, "cudaRegisterFatBinary(%p)=%p [%p] old=%p.", newfatCubin, res, (res == NULL)?0 : *res, fatCubin);
        _device_area_init_handle_ptr = &_device_area_init_handle_val;
        printf("registered: %p\n", _device_area_init_handle_ptr);
        ORIG(__cudaRegisterFunction)(res, _device_area_init_handle_ptr,
                               (char*)NAMEOF_INIT_FUNCTION_NAME, NAMEOF_INIT_FUNCTION_NAME, 0xffffffff, 0, 0, 0, 0, 0); 
        return res;
    }

    void load_preinstrumented_init_function(const char* hostFun)
    {
        _device_area_init_handle_ptr = (char*)hostFun;
        SLIM_DPRINTF(VL_VERBOSE, "load_preinstrumented_init_function(): Original function at %p", hostFun);
    }


    void ensure_configured()
    {
        if(_disabled)
            HostDetector::disable();

        // XXX: lock!
        if(_unloading || _is_init || _disabled) return;

    	size_t free = 0, total = 0;
    	VERIFY(ORIG(cudaMemGetInfo)(&free, &total));

        //uintptr_t ptr;
        //VERIFY(ORIG(cudaMalloc)((void**)&ptr, free));
        //VERIFY(ORIG(cudaFree)((void*)ptr));    

    	size_t size = total / (1 + SHADOW_MEMORY_RATIO);
        LH_MY_MEMORY = size;
//    	printf("Free=%lx total=%lx size=%lx\n", free, total, size);
    
        void* base;
    	VERIFY(ORIG(cudaMalloc)(&base, size));
//    	printf("Shadow alloced at %p\n", base);
   
        int numq = Env::get_num_queues();
        SLIM_DPRINTF(VL_VERBOSE, "Initializing %i queues.", numq);
        _dev_area = DeviceArea(base, size, numq); 
		VERIFY(ORIG(cudaDeviceSynchronize)());
//        printf("library init.\n");
        _is_init = true;
		HostDetector::INSTANCE.on_init(_dev_area.numq());
  
//        printf("Running %s\n", NAMEOF_INIT_FUNCTION_NAME); 
        init_device_area();
        run_logger();
    	
        free = 0; total = 0;
    	VERIFY(ORIG(cudaMemGetInfo)(&free, &total));
        BASE_MEM = total - free;
    }

    const void* get_hnd()
    {
        return _device_area_init_handle_ptr;
    }

    void before_reset()
    {
        // XXX: lock
        stop_logger();
		HostDetector::INSTANCE.print_hazards();
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
        printf("init_func_ptr: %p\n", _device_area_init_handle_ptr);
        SLIM_DPRINTF(VL_VERBOSE, "Init device area begins.");
        dim3 g, d;
        d.x = d.y = d.z = 1;
        g.x = g.y = g.z = 1;
        VERIFY(ORIG(cudaConfigureCall)(g, d, 0, 0));
        // XXX: calculate instead of constant
        VERIFY(ORIG(cudaSetupArgument)(&_dev_area, sizeof(_dev_area), 0));
        VERIFY(ORIG(cudaLaunch)(_device_area_init_handle_ptr));
        VERIFY(ORIG(cudaDeviceSynchronize)());
        //printf("Device area init.\n");
        SLIM_DPRINTF(VL_VERBOSE, "Init device area ends.");
    }

private:
    static char _device_area_init_handle_val;
    static char* _device_area_init_handle_ptr;
    std::unique_ptr<Logger> _logger;
    std::unique_ptr<FatBinary> _fatbin;
    bool _unloading;
    bool _is_init;
    bool _disabled;
    DeviceArea _dev_area;
};

char Impl::_device_area_init_handle_val = 0;
char* Impl::_device_area_init_handle_ptr = nullptr;

