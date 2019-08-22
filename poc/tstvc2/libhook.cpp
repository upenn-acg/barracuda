#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include "hooks.hpp"
#include "impl.hpp"
#include "env.hpp"

//////////////////////////////////////////////////

static Impl* __impl = NULL;
static timespec ts_start = {0,0}, ts_end = {0,0};
static bool start_set = false;

static inline void regclock_before()
{
    if(!start_set)
    {
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
        start_set = true;
    }
}

static inline void regclock_after()
{
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
}

const char* LH_NAME = "<unknown>"; // XXX
const char* LH_ENABLED = "DISABLED";
const char* LH_KERNEL_INVOCATIONS = 0;

__attribute__((constructor)) void __lib_constructor(void)
{
    static char linkname[1024] = "<unknown>";
    readlink("/proc/self/exe", linkname, sizeof(linkname));
    const char* name = strrchr(linkname, '/');
    if(name == NULL) name = linkname; else name += 1;
    LH_ENABLED = Env::dont_instrument_funcs() ? "DISABLED" : "ENABLED";
    LH_NAME = name;
}

__attribute__((destructor)) void __lib_destructor(void)
{
    static bool done = false;
    if(done)
        return;
    done = true;

    fflush(stdout);
    fflush(stderr);
    if(ts_end.tv_sec == 0 && ts_end.tv_nsec == 0)
        regclock_after();

    if(start_set)
    {
        long time;
        if(ts_end.tv_nsec >= ts_start.tv_nsec)
        {
            time = (ts_end.tv_sec - ts_start.tv_sec) * 1000000000L;
            time += (ts_end.tv_nsec - ts_start.tv_nsec);
        }
        else
        {
            time = (ts_end.tv_sec - ts_start.tv_sec - 1) * 1000000000L;
            time += (1000000000 + ts_end.tv_nsec - ts_start.tv_nsec);
        }
    	fprintf(stderr, "TIME MS %s %li %s\n", LH_NAME, time / 1000000, LH_ENABLED);
    }
    else
    	fprintf(stderr, "TIME MS %s ERROR %s\n", LH_NAME, LH_ENABLED);
    if(!Env::gpu_only())
       	fprintf(stderr, "COUNTS %s %s %i %i %i\n", LH_NAME, LH_ENABLED, NUM_BEFORE_PTX_INSTRUCTIONS, NUM_AFTER_PTX_INSTRUCTIONS, NUM_INSTRUMENTED_INSTRUCTIONS);
    fflush(stderr);
}

static void close_impl()
{
    Impl* impl = __sync_lock_test_and_set(&__impl, NULL);

    if(impl != NULL)
        delete impl;

}

static inline void register_at_exit() // hack as our at-exit has to be register AFTER cuda's
{
    static bool _registered = false;

    if(!_registered)
    {
        _registered = true;
        //printf("Registering at exit.\n");
        atexit(close_impl);
    }
}

//////////////////////////////////////////////////

extern "C" __host__ __cudart_builtin__ void** __cudaRegisterFatBinary(void *fatCubin)
{
    if(__impl != NULL)
    {
        fprintf(stderr, "Only single fat binary supported!\n");
        exit(1);
    }
    __impl = new Impl();
    return __impl->register_binary(fatCubin);
}

extern "C" __host__ __cudart_builtin__ void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
    ORIG(__cudaUnregisterFatBinary)(fatCubinHandle);
}

extern "C" __host__ __cudart_builtin__ void __cudaRegisterFunction(
        void   **fatCubinHandle,
        const char    *hostFun,
        char    *deviceFun,
        const char    *deviceName,
        int      thread_limit,
        uint3   *tid,
        uint3   *bid,
        dim3    *bDim,
        dim3    *gDim,
        int     *wSize
)
{
//    fprintf(stderr, "cudaRegisterFunction(%p, %p, %s, %s, %i)\n", fatCubinHandle, hostFun, deviceName, deviceName, thread_limit);
    ORIG(__cudaRegisterFunction)(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);

    if(!Env::gpu_only())
  	    fprintf(stderr, "KERNEL NAME %s %s %s\n", LH_NAME, LH_ENABLED, deviceName);
}

/*extern "C" __host__ __cudart_builtin__ char __cudaInitModule(void **fatCubinHandle)
{
    char res = ORIG(__cudaInitModule)(fatCubinHandle);
	fprintf(stderr, "cudaInitModule(%p)=%i.\n", fatCubinHandle, res);
    return res;
}*/

extern "C" __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceSynchronize(void)
{
	cudaError_t status;
//	fprintf(stderr, "cudaDeviceSynchronize().\n");

    regclock_before();
    if(__impl != nullptr)
        __impl->ensure_configured();
	status = ORIG(cudaDeviceSynchronize)();
    register_at_exit(); // has to be at end of function    
    regclock_after();
	return status;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
{
	cudaError_t status;
//	fprintf(stderr, "cudaConfigureCall().\n");

    size_t free = 0, total = 0;
  	VERIFY(ORIG(cudaMemGetInfo)(&free, &total));
    fflush(stderr);
    if(!Env::gpu_only())
   	    fprintf(stderr, "MEM_BEFORE %s %s %li\n", LH_NAME, LH_ENABLED, total - free - LH_MY_MEMORY);
    int count_blocks = gridDim.x * gridDim.y * gridDim.z;
    int count_threads = blockDim.x * blockDim.y * blockDim.z;
    if(!Env::gpu_only())
   	    fprintf(stderr, "CONFIGURE %s %s %i %i %i\n", LH_NAME, LH_ENABLED, count_blocks, count_threads, count_threads * count_blocks);
    fflush(stderr);
    
    regclock_before();
    if(__impl != nullptr)
    {
        __impl->ensure_configured();
        HostDetector::on_kernel_launched(0, gridDim, blockDim); // XXX stream
    }
	status = ORIG(cudaConfigureCall)(gridDim, blockDim, sharedMem, stream);
    register_at_exit(); // has to be at end of function
    regclock_after();
	return status;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaLaunch(const void *func)
{
	cudaError_t status;
//	fprintf(stderr, "cudaLaunch(func=%p).\n", func);

    regclock_before();
	status = ORIG(cudaLaunch)(func);
//	fprintf(stderr, "cudaLaunch(func=%p) => %i.\n", func, status);
    register_at_exit(); // has to be at end of function
    regclock_after();

	return status;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaDeviceReset(void)
{
	cudaError_t status;
//	fprintf(stderr, "cudaDeviceReset().\n");

    __impl->before_reset();
	status = ORIG(cudaDeviceReset)();
    __impl->after_reset();

    register_at_exit(); // has to be at end of function
    regclock_after();

	return status;
}

extern "C" __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size)
{
	cudaError_t status;
//	fprintf(stderr, "cudaMalloc().\n");
    regclock_before();
    if(__impl != nullptr)
        __impl->ensure_configured();
	status = ORIG(cudaMalloc)(devPtr, size);
//	fprintf(stderr, "cudaMalloc(size=%lx): returned %p.\n", size, *devPtr);
//    atexit(close_impl);;

    register_at_exit(); // has to be at end of function
    regclock_after();
	return status;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
	cudaError_t status;
//	fprintf(stderr, "cudaMemcpy(dst=%p, src=%p, count=%lu kind=%i).\n", dst, src, count, kind);
	status = ORIG(cudaMemcpy)(dst, src, count, kind);
    regclock_after();
	return status;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	cudaError_t status;
//	fprintf(stderr, "cudaMemcpyAsync(dst=%p, src=%p, count=%lu kind=%i).\n", dst, src, count, kind);
	status = ORIG(cudaMemcpyAsync)(dst, src, count, kind, stream);
	return status;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *pStream)
{
    register_at_exit();
    
	cudaError_t status;
    regclock_before();
    if(__impl != nullptr)
        __impl->ensure_configured();
//	fprintf(stderr, "cudaStreamCreate().\n");
	status = ORIG(cudaStreamCreate)(pStream);
	return status;
}

extern "C" __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
{
	cudaError_t status;
    regclock_before();
    if(__impl != nullptr)
        __impl->ensure_configured();
//	fprintf(stderr, "cudaStreamCreateWithFlags().\n");
	status = ORIG(cudaStreamCreateWithFlags)(pStream, flags);
	return status;
}

extern "C" __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream)
{
	cudaError_t status;
//	fprintf(stderr, "cudaStreamDestroy().\n");
	status = ORIG(cudaStreamDestroy)(stream);
	return status;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free, size_t *total)
{
	cudaError_t status;
    regclock_before();
    if(__impl != nullptr)
        __impl->ensure_configured();
//	fprintf(stderr, "cudaMemGetInfo().\n");
	status = ORIG(cudaMemGetInfo)(free, total);
    register_at_exit(); // has to be at end of function
	return status;
}

////////////////////////////////////////////////////


