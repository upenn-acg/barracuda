#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "debug.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <pthread.h>
#include "hooklib.h"
#include "libhookinfra.hpp"

//////////////////////////////////////////////////

#define SHADOW_MEMORY_RATIO 1

#define LIBRARY_NOT_INIT 0
#define LIBRARY_MID_INIT 1
#define LIBRARY_END_INIT 2

static int gLibrary_init = LIBRARY_NOT_INIT;
static pthread_mutex_t gLibrary_mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
static void* gLibrary_shadowbase = NULL;
static autogen_cb_t gLibrary_autogen_cb = NULL;

static void init_library();
static void init_library_mem_space();


DECLARE_HOOK_FUNC(cudaDeviceSynchronize,(void))
DECLARE_HOOK_FUNC(cudaConfigureCall,(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream))
DECLARE_HOOK_FUNC(cudaLaunch, (const void *func))
DECLARE_HOOK_FUNC(cudaDeviceReset,(void))
DECLARE_HOOK_FUNC(cudaMalloc,(void **devPtr, size_t size))
DECLARE_HOOK_FUNC(cudaMemcpy,(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind))
DECLARE_HOOK_FUNC(cudaMemcpyAsync, (void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream))
DECLARE_HOOK_FUNC(cudaMemGetInfo,(size_t *free, size_t *total))
DECLARE_HOOK_FUNC(cudaStreamCreate, (cudaStream_t *pStream))
DECLARE_HOOK_FUNC(cudaStreamCreateWithFlags, (cudaStream_t *pStream, unsigned int flags))
DECLARE_HOOK_FUNC(cudaStreamDestroy, (cudaStream_t stream))


#define VERIFY(f) for(;;) { if(cudaSuccess != (f)) { fprintf(stderr,"Function failed: '%s'\n", #f); exit(1); } break;}

//////////////////////////////////////////////////

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceSynchronize(void)
{
	cudaError_t status;
	init_library();
	DEBUGONLY(fprintf(stderr, "cudaDeviceSynchronize().\n");)

	status = ORIG(cudaDeviceSynchronize)();
	return status;
}

extern __host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
{
	cudaError_t status;
	init_library();
	DEBUGONLY(fprintf(stderr, "cudaConfigureCall().\n");)

	status = ORIG(cudaConfigureCall)(gridDim, blockDim, sharedMem, stream);
	return status;
}

extern __host__ cudaError_t CUDARTAPI cudaLaunch(const void *func)
{
	cudaError_t status;
	init_library();
	DEBUGONLY(fprintf(stderr, "cudaLaunch(func=%p).\n", func);)

	status = ORIG(cudaLaunch)(func);
	return status;
}

extern __host__ cudaError_t CUDARTAPI cudaDeviceReset(void)
{
	cudaError_t status;
	DEBUGONLY(fprintf(stderr, "cudaDeviceReset().\n");)

	status = ORIG(cudaDeviceReset)();
	gLibrary_init = 0;

	return status;
}

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size)
{
	cudaError_t status;
	init_library();
	DEBUGONLY(fprintf(stderr, "cudaMalloc().\n");)
	status = ORIG(cudaMalloc)(devPtr, size);
	DEBUGONLY(fprintf(stderr, "cudaMalloc(size=%lx): returned %p.\n", size, *devPtr);)
	return status;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
	cudaError_t status;
	init_library();
	DEBUGONLY(fprintf(stderr, "cudaMemcpy(dst=%p, src=%p, count=%lu kind=%i).\n", dst, src, count, kind);)
	status = ORIG(cudaMemcpy)(dst, src, count, kind);
	return status;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	cudaError_t status;
	init_library();
	DEBUGONLY(fprintf(stderr, "cudaMemcpyAsync(dst=%p, src=%p, count=%lu kind=%i).\n", dst, src, count, kind);)
	status = ORIG(cudaMemcpyAsync)(dst, src, count, kind, stream);
	return status;
}

extern __host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *pStream)
{
	cudaError_t status;
	init_library();
	DEBUGONLY(fprintf(stderr, "cudaStreamCreate().\n");)
	status = ORIG(cudaStreamCreate)(pStream);
	return status;
}

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
{
	cudaError_t status;
	init_library();
	DEBUGONLY(fprintf(stderr, "cudaStreamCreateWithFlags().\n");)
	status = ORIG(cudaStreamCreateWithFlags)(pStream, flags);
	return status;
}

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream)
{
	cudaError_t status;
	init_library();
	DEBUGONLY(fprintf(stderr, "cudaStreamDestroy().\n");)
	status = ORIG(cudaStreamDestroy)(stream);
	return status;
}



extern __host__ cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free, size_t *total)
{
	cudaError_t status;
	init_library();
	DEBUGONLY(fprintf(stderr, "cudaMemGetInfo().\n");)
	status = ORIG(cudaMemGetInfo)(free, total);
	return status;
}

////////////////////////////////////////////////////

__attribute__((constructor)) void __lib_con(void)
{
#ifdef DEBUG
	fprintf(stderr, "hooklibdbg.so loading.\n");
#else
	fprintf(stderr, "hooklib.so loading.\n");
#endif

}

extern "C" void __libhook_register_init_cb(autogen_cb_t cb)
{
	gLibrary_autogen_cb = cb;
}

static void init_library()
{
	if(gLibrary_init == LIBRARY_END_INIT)
		return;

	pthread_mutex_lock(&gLibrary_mutex);
	if(gLibrary_init > LIBRARY_NOT_INIT) // exit here even in mid init
    {
	    pthread_mutex_unlock(&gLibrary_mutex);		
		return;
    }
	gLibrary_init = LIBRARY_MID_INIT;

	if(gLibrary_autogen_cb == NULL)
	{
		fprintf(stderr, "__libhook_register_init_cb() was not injected into binary main.\n");
		exit(1);
	}

    _Hook_BASE::load_hooks(RTLD_NEXT);
	init_library_mem_space();
	gLibrary_init = LIBRARY_END_INIT;
	
	pthread_mutex_unlock(&gLibrary_mutex);		
}

static void init_library_mem_space()
{
	size_t free = 0, total = 0;
	size_t toalloc = 0;

	ORIG(cudaDeviceReset)();
	VERIFY(ORIG(cudaMemGetInfo)(&free, &total));

	toalloc = total / (1 + SHADOW_MEMORY_RATIO);
	DEBUGONLY(printf("Free=%lx total=%lx toalloc=%lx\n", free, total, toalloc);)

	VERIFY(ORIG(cudaMalloc)(&gLibrary_shadowbase, toalloc));
	DEBUGONLY(printf("Shadow alloced at %p\n", gLibrary_shadowbase);)

	gLibrary_autogen_cb(gLibrary_shadowbase, toalloc);
    VERIFY(ORIG(cudaDeviceSynchronize)());
}

