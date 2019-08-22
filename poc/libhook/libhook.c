#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <pthread.h>
#include "hooklib.h"

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
static void init_library_orig_funcs();
static void init_library_mem_space();

typedef cudaError_t (*cudaDeviceSynchronize_t)(void);
typedef cudaError_t (*cudaConfigureCall_t)(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
typedef cudaError_t (*cudaLaunch_t)(const void *func);
typedef cudaError_t (*cudaDeviceReset_t)(void);
typedef cudaError_t (*cudaMalloc_t)(void **devPtr, size_t size);
typedef cudaError_t (*cudaMemcpy_t)(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
typedef cudaError_t (*cudaMemGetInfo_t)(size_t *free, size_t *total);

static cudaDeviceSynchronize_t orig_cudaDeviceSynchronize;
static cudaConfigureCall_t orig_cudaConfigureCall;
static cudaLaunch_t orig_cudaLaunch;
static cudaDeviceReset_t orig_cudaDeviceReset;
static cudaMalloc_t orig_cudaMalloc;
static cudaMemcpy_t orig_cudaMemcpy;
static cudaMemGetInfo_t orig_cudaMemGetInfo;

#define VERIFY(f) for(;;) { if(cudaSuccess != (f)) { fprintf(stderr,"Function failed: '%s'\n", #f); exit(1); } break;}

//////////////////////////////////////////////////

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceSynchronize(void)
{
	cudaError_t status;
	init_library();
	fprintf(stderr, "cudaDeviceSynchronize().\n");

	status = orig_cudaDeviceSynchronize();
	return status;
}

extern __host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
{
	cudaError_t status;
	init_library();
	fprintf(stderr, "cudaConfigureCall().\n");

	status = orig_cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
	return status;
}

extern __host__ cudaError_t CUDARTAPI cudaLaunch(const void *func)
{
	cudaError_t status;
	init_library();
	fprintf(stderr, "cudaLaunch(func=%p).\n", func);

	status = orig_cudaLaunch(func);
	return status;
}

extern __host__ cudaError_t CUDARTAPI cudaDeviceReset(void)
{
	cudaError_t status;
	fprintf(stderr, "cudaDeviceReset().\n");

	status = orig_cudaDeviceReset();
	gLibrary_init = 0;

	return status;
}

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size)
{
	cudaError_t status;
	init_library();
	fprintf(stderr, "cudaMalloc().\n");
	status = orig_cudaMalloc(devPtr, size);
	fprintf(stderr, "cudaMalloc(size=%lx): returned %p.\n", size, *devPtr);
	return status;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
	cudaError_t status;
	init_library();
	fprintf(stderr, "cudaMemcpy(dst=%p, src=%p, count=%lu kind=%i).\n", dst, src, count, kind);
	status = orig_cudaMemcpy(dst, src, count, kind);
	return status;
}

extern __host__ cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free, size_t *total)
{
	cudaError_t status;
	init_library();
	fprintf(stderr, "cudaMemGetInfo().\n");
	status = orig_cudaMemGetInfo(free, total);
	return status;
}

////////////////////////////////////////////////////

__attribute__((constructor)) void __lib_con(void)
{
	fprintf(stderr, "hooklib.so loading.\n");
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
		return;
	gLibrary_init = LIBRARY_MID_INIT;

	if(gLibrary_autogen_cb == NULL)
	{
		fprintf(stderr, "__libhook_register_init_cb() was not injected into binary main.\n");
		exit(1);
	}

	init_library_orig_funcs();
	init_library_mem_space();
	gLibrary_init = LIBRARY_END_INIT;
	
	pthread_mutex_unlock(&gLibrary_mutex);		
}

static void* get_orig_func(void* handle, const char* symbol)
{
	void* sym = dlsym(handle, symbol);
	if(sym == NULL)
	{
		fprintf(stderr, "Failed loading function: '%s'\n", symbol);
		exit(1);
	}
	return sym;
}

#define LOAD_FUNC(name)	for(;;) { orig_##name = (name##_t)get_orig_func(RTLD_NEXT, #name); break; }

static void init_library_orig_funcs()
{
	LOAD_FUNC(cudaDeviceSynchronize);
	LOAD_FUNC(cudaConfigureCall);
	LOAD_FUNC(cudaLaunch);
	LOAD_FUNC(cudaDeviceReset);
	LOAD_FUNC(cudaMalloc);
	LOAD_FUNC(cudaMemcpy);
	LOAD_FUNC(cudaMemGetInfo);
}

static void init_library_mem_space()
{
	size_t free = 0, total = 0;
	size_t toalloc = 0;

	orig_cudaDeviceReset();
	VERIFY(orig_cudaMemGetInfo(&free, &total));

	toalloc = total / (1 + SHADOW_MEMORY_RATIO);
	printf("Free=%lx total=%lx toalloc=%lx\n", free, total, toalloc);

	VERIFY(orig_cudaMalloc(&gLibrary_shadowbase, toalloc));
	printf("Shadow alloced at %p\n", gLibrary_shadowbase);

	gLibrary_autogen_cb(gLibrary_shadowbase);
}

