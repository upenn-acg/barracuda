#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h> 
#include "devarea.hpp"

#include "ptx_program.hpp"
#include "ptx_parser.hpp"
#include "ptx_instrumentation.hpp"
#include "ptx_stub.h"

std::string read_file(const std::string& filename)
{
	std::ifstream stream(filename);
	std::string str;

	if (!stream)
	{
		std::cerr << "Could not read: " << filename << std::endl;
		exit(1);
	}

	stream.seekg(0, std::ios::end);
	str.reserve((unsigned int)stream.tellg());
	stream.seekg(0, std::ios::beg);

	str.assign((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
	return str;
}

//cudaError_t CUDARTAPI cudaDeviceReset() { return cudaError_t(); }

void load_ptx(const char* file)
{
    CUmodule     module  = 0;
    CUfunction   kernel  = 0;
    CUlinkState  lnkstate;

    const int log_size = 16384;
    CUjit_option options[6];
    void *option_vals[sizeof(options)/sizeof(options[0])];
    float walltime;
    char error_log[log_size], info_log[log_size];
    void *lnkout;
    size_t out_size;
    int err = 0;
    int dev_count = 0;

    cudaDeviceReset();

    checkCudaErrors(cuDeviceGetCount(&dev_count));
    if (dev_count == 0)
    {
        fprintf(stderr, "No CUDA devices found!\n");
        return;
    }
    CUdevice device;
    checkCudaErrors(cuDeviceGet(&device, 0));

    CUcontext context;
    checkCudaErrors(cuCtxCreate(&context, 0, device));
    
    options[0] = CU_JIT_WALL_TIME;
    option_vals[0] = (void *) &walltime;
    options[1] = CU_JIT_INFO_LOG_BUFFER;
    option_vals[1] = (void *) info_log;
    options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    option_vals[2] = (void *) (uintptr_t)log_size;
    options[3] = CU_JIT_ERROR_LOG_BUFFER;
    option_vals[3] = (void *) error_log;
    options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    option_vals[4] = (void *) (uintptr_t) log_size;
    options[5] = CU_JIT_LOG_VERBOSE;
    option_vals[5] = (void *) 1;

    // Create a pending linker invocation
    checkCudaErrors(cuLinkCreate(6,options, option_vals, &lnkstate));

    err = cuLinkAddFile(lnkstate, CU_JIT_INPUT_PTX, file ,0,0,0);

    if (err != CUDA_SUCCESS)
    {
        fprintf(stderr,"PTX Linker Error:\n%s\n",error_log);
        return;
    }

    checkCudaErrors(cuLinkComplete(lnkstate, &lnkout, &out_size));
    printf("CUDA Link Completed in %fms. Linker Output:\n%s\n", walltime, info_log);

    checkCudaErrors(cuModuleLoadData(&module, lnkout));
    checkCudaErrors(cuModuleGetFunction(&kernel, module, NAMEOF_INIT_FUNCTION_NAME));
    printf("Got kernel: %p\n", kernel);

    uint64_t* x;
    if(0 != cudaMalloc(&x, sizeof(uint64_t)))
    {
        printf("Failed cudaMalloc().\n");
        return;
    }
    void* buf;
    int buf_size = 64 * 1000;
    if(0 != cudaMalloc(&buf, buf_size))
    {
        printf("Failed cudaMalloc().\n");
        return;
    }

    checkCudaErrors(cuFuncSetBlockShape(kernel, 1, 1, 1));
    int param_offset = 0;
    checkCudaErrors(cuParamSetSize(kernel, param_offset));
    checkCudaErrors(cuLaunchGrid(kernel, 1, 1));
    printf("Kernel launched!\n");
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cuLinkDestroy(lnkstate));

}

int main(int argc, char* argv[])
{
	if(argc != 2)
	{
		fprintf(stderr, "Syntax: %s file.ptx\n", argv[0]);
		return 1;
	}
	std::string ptx_text = read_file(argv[1]);

	PtxInstrumentation instrumentation;
	std::string instrumented;
	if (!instrumentation.instrument(ptx_text, &instrumented))
	{
		std::cerr << "Error instrumenting PTX" << std::endl;
	}

    const char* outname = "out.ptx";
    {
    	std::ofstream out(outname);
        out << instrumented;
    }

    load_ptx(outname);

	return 0;
}

