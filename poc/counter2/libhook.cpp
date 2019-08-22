#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <pthread.h>
#include "hooks.hpp"
#include "impl.hpp"

//////////////////////////////////////////////////

static Impl* __impl = NULL;

__attribute__((constructor)) void __lib_constructor(void)
{
}

__attribute__((destructor)) void __lib_destructor(void)
{
	fprintf(stderr, "hooklib.so unloading.\n");
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
        printf("Registering at exit.\n");
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

    __impl->ensure_configured();
	status = ORIG(cudaDeviceSynchronize)();
    register_at_exit(); // has to be at end of function    
	return status;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
{
	cudaError_t status;
//	fprintf(stderr, "cudaConfigureCall().\n");

    __impl->ensure_configured();
	status = ORIG(cudaConfigureCall)(gridDim, blockDim, sharedMem, stream);
    register_at_exit(); // has to be at end of function
	return status;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaLaunch(const void *func)
{
	cudaError_t status;
//	fprintf(stderr, "cudaLaunch(func=%p).\n", func);

	status = ORIG(cudaLaunch)(func);
//	fprintf(stderr, "cudaLaunch(func=%p) => %i.\n", func, status);
    register_at_exit(); // has to be at end of function

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

	return status;
}

extern "C" __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size)
{
	cudaError_t status;
//	fprintf(stderr, "cudaMalloc().\n");
    __impl->ensure_configured();
	status = ORIG(cudaMalloc)(devPtr, size);
//	fprintf(stderr, "cudaMalloc(size=%lx): returned %p.\n", size, *devPtr);
//    atexit(close_impl);;

    register_at_exit(); // has to be at end of function
	return status;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
	cudaError_t status;
//	fprintf(stderr, "cudaMemcpy(dst=%p, src=%p, count=%lu kind=%i).\n", dst, src, count, kind);
	status = ORIG(cudaMemcpy)(dst, src, count, kind);
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
    __impl->ensure_configured();
//	fprintf(stderr, "cudaStreamCreate().\n");
	status = ORIG(cudaStreamCreate)(pStream);
	return status;
}

extern "C" __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
{
	cudaError_t status;
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
    __impl->ensure_configured();
//	fprintf(stderr, "cudaMemGetInfo().\n");
	status = ORIG(cudaMemGetInfo)(free, total);
    register_at_exit(); // has to be at end of function
	return status;
}

////////////////////////////////////////////////////


