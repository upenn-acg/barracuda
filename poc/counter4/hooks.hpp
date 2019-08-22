#pragma once


#include "libhookinfra.hpp"


// XXX: __cudaRegisterVar/ __cudaRegisterManagedVar
DECLARE_HOOK_RETFUNC(__cudaRegisterFatBinary,void**, (void *fatCubin))
DECLARE_HOOK_RETFUNC(__cudaUnregisterFatBinary,void, (void *fatCubinHandle))
DECLARE_HOOK_RETFUNC(__cudaRegisterFunction,void, (
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
))
DECLARE_HOOK_RETFUNC(__cudaInitModule,char, (void **fatCubinHandle))
DECLARE_HOOK_FUNC(cudaDeviceSynchronize,(void))
DECLARE_HOOK_FUNC(cudaConfigureCall,(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream))
DECLARE_HOOK_FUNC(cudaLaunch, (const void *func))
DECLARE_HOOK_FUNC(cudaSetupArgument, (const void *arg, size_t size, size_t offset))
DECLARE_HOOK_FUNC(cudaDeviceReset,(void))
DECLARE_HOOK_FUNC(cudaMalloc,(void **devPtr, size_t size))
DECLARE_HOOK_FUNC(cudaFree,(void *devPtr))
DECLARE_HOOK_FUNC(cudaHostAlloc, (void **pHost, size_t size, unsigned int flags))
DECLARE_HOOK_FUNC(cudaFreeHost,(void *devPtr))
DECLARE_HOOK_FUNC(cudaMemset,(void *devPtr, int value, size_t count))
DECLARE_HOOK_FUNC(cudaMemcpy,(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind))
DECLARE_HOOK_FUNC(cudaMemcpyAsync, (void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream))
DECLARE_HOOK_FUNC(cudaMemGetInfo,(size_t *free, size_t *total))
DECLARE_HOOK_FUNC(cudaStreamCreate, (cudaStream_t *pStream))
DECLARE_HOOK_FUNC(cudaStreamCreateWithFlags, (cudaStream_t *pStream, unsigned int flags))
DECLARE_HOOK_FUNC(cudaStreamDestroy, (cudaStream_t stream))
DECLARE_HOOK_FUNC(cudaStreamSynchronize, (cudaStream_t stream))
