#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "hooklib.h"

/** THESE FUNCTIONS SHOULD BE AUTO GENERTED BY THE COMPILER AND INJECTED INTO THE FINAL BINARY **/
__device__ void* __gHooklib_shadow_base = NULL;
__global__ void __autogen_cb_devinit(void* shadow_base)
{
    __gHooklib_shadow_base = shadow_base;
}

static void __autogen_cb(void* shadow_base)
{
    __autogen_cb_devinit<<<1,1>>>(shadow_base);   
}

/** END AUTO_GENERATED FUNCTIONS **/

__global__ void axpy(float a, float* x, float* y) {
    y[threadIdx.x] = a * x[threadIdx.x];
    __syncthreads();

    /* CODE INJECTED INTO AXPY */
    printf("Shadow at: %p\n", __gHooklib_shadow_base);
    /* END CODE INJECTION */
}

int main(int argc, char* argv[]) {
    /* CODE INJECTED INTO MAIN */
	__libhook_register_init_cb(__autogen_cb);
    /* END CODE INJECTED INTO MAIN */

    const int kDataLen = 4;

    float a = 2.0f;
    float host_x[kDataLen] = {1.0f, 2.0f, 3.0f, 4.0f};
    float host_y[kDataLen];

    // Copy input data to device.
    float* device_x;
    float* device_y;
     checkCudaErrors(cudaMalloc(&device_x, kDataLen * sizeof(float)));
     checkCudaErrors(cudaMalloc(&device_y, kDataLen * sizeof(float)));
     checkCudaErrors(cudaMemcpy(device_x, host_x, kDataLen * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Launch the kernel.
    axpy<<<1, kDataLen>>>(a, device_x, device_y);

    // Copy output data to host.
     checkCudaErrors(cudaDeviceSynchronize());
     checkCudaErrors(cudaMemcpy(host_y, device_y, kDataLen * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // Print the results.
    for (int i = 0; i < kDataLen; ++i) {
        std::cout << "y[" << i << "] = " << host_y[i] << "\n";
    }

    cudaDeviceReset();
    return 0;
}

