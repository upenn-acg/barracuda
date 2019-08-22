// System includes
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdio.h>

// CUDA driver & runtime
#include <cuda.h>
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

int main(int argc, char **argv)
{
    const unsigned int nThreads = 256;
    const unsigned int nBlocks  = 64;
    const size_t memSize = nThreads * nBlocks * sizeof(int);

    CUmodule     hModule  = 0;
    CUfunction   hKernel  = 0;
    CUlinkState  lnkstate;
    int         *d_data   = 0;
    int         *h_data   = 0;

    if(argc != 2)
    {
        fprintf(stderr, "%s a.ptx\n", argv[0]);
        return -1;
    }

    CUjit_option options[6];
    void *optionVals[6];
    float walltime;
    char error_log[16384],
         info_log[16384];
    unsigned int logSize = sizeof(error_log);
    void *cuOut;
    size_t outSize;
    int err = 0;

    // Setup linker options
    // Return walltime from JIT compilation
    options[0] = CU_JIT_WALL_TIME;
    optionVals[0] = (void *) &walltime;
    // Pass a buffer for info messages
    options[1] = CU_JIT_INFO_LOG_BUFFER;
    optionVals[1] = (void *) info_log;
    // Pass the size of the info buffer
    options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    optionVals[2] = (void *) (long)logSize;
    // Pass a buffer for error message
    options[3] = CU_JIT_ERROR_LOG_BUFFER;
    optionVals[3] = (void *) error_log;
    // Pass the size of the error buffer
    options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    optionVals[4] = (void *) (long) logSize;
    // Make the linker verbose
    options[5] = CU_JIT_LOG_VERBOSE;
    optionVals[5] = (void *) 1;

    // Create a pending linker invocation
    checkCudaErrors(cuLinkCreate(6,options, optionVals, lnkstate));

    err = cuLinkAddFile(*lnkstate, CU_JIT_INPUT_PTX, argv[1] ,0,0,0);

    if (err != CUDA_SUCCESS)
    {
        // Errors will be put in error_log, per CU_JIT_ERROR_LOG_BUFFER option above.
        fprintf(stderr,"PTX Linker Error:\n%s\n",error_log);
    }

    // Complete the linker step
    checkCudaErrors(cuLinkComplete(*lnkstate, &cuOut, &outSize));

    // Linker walltime and info_log were requested in options above.
    printf("CUDA Link Completed in %fms. Linker Output:\n%s\n",walltime,info_log);

    // Load resulting cuBin into module
    checkCudaErrors(cuModuleLoadData(ptxmod, cuOut));

    checkCudaErrors(cuModuleGetFunction(kernel, *ptxmod, "_Z8myKernelPi"));
    checkCudaErrors(cuLinkDestroy(*lnkstate));

    cudaDeviceReset();

    return dataGood ? EXIT_SUCCESS : EXIT_FAILURE;
}
