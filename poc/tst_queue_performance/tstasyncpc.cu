#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <memory>
#include "debug.h"
#include "hooklib.h"
#include "protocol.hpp"
#include "slimfast.hpp"
#include "consumer.hpp"
#include "impl.hpp"
#include "devlogger.hpp"

__global__ void benchmark_memwrite(const int num_repeats, bool enabled, volatile unsigned int* data) 
{
    int id = ((blockDim.x * blockIdx.x) + threadIdx.x);
    if(enabled)
    {
        for(int i = 0; i < num_repeats; ++i )
        {
            data[id] += 1;
            __store_op(7,(void*)&data[id], OP_READWRITE);
        }
        __threadfence_system(); // FLUSH INSTRUMENTATION
    }
    else
    {
        for(int i = 0; i < num_repeats; ++i )
        {
            data[id] += 1;
        }
    }
}
__global__ void benchmark_saxpy(const int num_repeats, bool enabled, int N, float a, float *x, float *y) 
{
    if(enabled)
    {
        for(int j = 0; j < num_repeats; ++j )
        {
            int i = blockIdx.x*blockDim.x + threadIdx.x;
            if (i < N)
            {
                float vx = x[i];
                __store_op(7,&x[i], OP_READ);
                float vy = y[i];
                __store_op(7,&y[i], OP_READ);
                y[i] = a*vx + vy;
                __store_op(7,&y[i], OP_WRITE);
            }
        }
        __threadfence_system(); // FLUSH INSTRUMENTATION
    }
    else
    {
        for(int j = 0; j < num_repeats; ++j )
        {
            int i = blockIdx.x*blockDim.x + threadIdx.x;
            if (i < N)
            {
                float vx = x[i];
                float vy = y[i];
                y[i] = a*vx + vy;
//                if(i == 0) printf("%f = %f*%f+%f\n", y[i], a, vx, vy);
            }
        }
    }
}
   
__global__ static void benchmark_timedReduction(const int num_repeats, bool enabled, const float *input, float *output, clock_t *timer)
{
    // __shared__ float shared[2 * blockDim.x];
    extern __shared__ float shared[];

    if(enabled)
    {
        for(int j = 0; j < num_repeats; ++ j)
        {
            const int tid = threadIdx.x;
            const int bid = blockIdx.x;
    
            if (tid == 0) 
            {
                timer[bid] = clock();
                __store_op(7,&timer[bid], OP_READ);
            }
    
            // Copy input.
            shared[tid] = input[tid];
            __store_op(7,&input[tid], OP_READ);
            __store_op(7,&shared[tid], OP_WRITE);
            shared[tid + blockDim.x] = input[tid + blockDim.x];
            __store_op(7,&input[tid + blockDim.x], OP_READ);
            __store_op(7,&shared[tid + blockDim.x], OP_WRITE);
    
            // Perform reduction to find minimum.
            for (int d = blockDim.x; d > 0; d /= 2)
            {
                __syncthreads();
        
                if (tid < d)
                {
                    float f0 = shared[tid];
                    __store_op(7,&shared[tid], OP_READ);
                    float f1 = shared[tid + d];
                    __store_op(7,&shared[tid + d], OP_READ);
        
                    if (f1 < f0)
                    {
                      shared[tid] = f1;
                      __store_op(7,&shared[tid], OP_WRITE);
                    }
                }
            }
    
            // Write result.
            if (tid == 0)
            {
                 output[bid] = shared[0];
                  __store_op(7,&shared[0], OP_READ);
                  __store_op(7,&output[bid], OP_WRITE);
            }
    
            __syncthreads();
    
            if (tid == 0)
            {
                 timer[bid+gridDim.x] = clock();
                 __store_op(7,&timer[bid+gridDim.x], OP_WRITE);
            }
        }
    }
    else
    {
        for(int j = 0; j < num_repeats; ++ j)
        {
            const int tid = threadIdx.x;
            const int bid = blockIdx.x;
    
            if (tid == 0) timer[bid] = clock();
    
            // Copy input.
            shared[tid] = input[tid];
            shared[tid + blockDim.x] = input[tid + blockDim.x];
    
            // Perform reduction to find minimum.
            for (int d = blockDim.x; d > 0; d /= 2)
            {
                __syncthreads();
        
                if (tid < d)
                {
                    float f0 = shared[tid];
                    float f1 = shared[tid + d];
        
                    if (f1 < f0)
                    {
                      shared[tid] = f1;
                    }
                }
            }
    
            // Write result.
            if (tid == 0) output[bid] = shared[0];
    
            __syncthreads();
    
            if (tid == 0) timer[bid+gridDim.x] = clock();
        }
    }
}

 
/*static long now()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    long out;
    out = ts.tv_nsec + ts.tv_sec * 1000000000;
    return out;
}*/


int main(int argc, char* argv[]) 
{
    if(argc != 7)
    {
        fprintf(stderr, "%s NUMQ BENCHMARK INSTRUMENTED THREADS BLOCKS REPEATS\n", argv[0]);
        fprintf(stderr, "  NUMQ - Number of queues\n");
        fprintf(stderr, "  BENCHMARK:\n");
        fprintf(stderr, "     1 - Mem write\n");
        fprintf(stderr, "     2 - Saxpy\n");
        fprintf(stderr, "     3 - Clock (cuda sample)\n");
        fprintf(stderr, "  INSTRUMENTED:\n");
        fprintf(stderr, "     0 - false\n");
        fprintf(stderr, "     1 - true\n");
        fprintf(stderr, "  THREADS: threads per threadblock\n");
        fprintf(stderr, "  BLOCKS:  threadblocks\n");
        fprintf(stderr, "  REPEATS: number of repeats\n");
        return 1;
    }

    const int NUMQ = atoi(argv[1]);
    const int TEST_ID = atoi(argv[2]);
    const int ENABLED = atoi(argv[3]);
    const int NUM_THREADS = atoi(argv[4]);
    const int NUM_BLOCKS = atoi(argv[5]);
    const int NUM_TOTAL = NUM_THREADS * NUM_BLOCKS;
    const int NUM_REPEATS = atoi(argv[6]);

    printf("@Numqueues:\t%i\n", NUMQ);
    printf("@TestId:\t%i\n", TEST_ID);
    printf("@Enabled:\t%i\n", ENABLED);
    printf("@Blocks:\t%i\n", NUM_BLOCKS);
    printf("@Threads:\t%i\n", NUM_THREADS);
    printf("@Total:\t%i\n", NUM_TOTAL);
    printf("@Repeats:\t%i\n", NUM_REPEATS);

    int enabled = ENABLED > 0;
    cudaEvent_t start, stop;
    std::auto_ptr<Impl> impl;

    switch(TEST_ID)
    {
    case 1:
        {
            const long expected = NUM_REPEATS * NUM_TOTAL / WARP_SIZE;
            impl.reset(new Impl(NUMQ, (enabled) ? expected : 0));

            // Launch the kernel.
            unsigned int* dev_data;
            checkCudaErrors(cudaMalloc(&dev_data, sizeof(unsigned int) * NUM_TOTAL));
            checkCudaErrors(cudaMemset(dev_data, 0, sizeof(unsigned int) * NUM_TOTAL));
            unsigned int* host_data = (unsigned int*)malloc(sizeof(unsigned int) * NUM_TOTAL);
    
            checkCudaErrors(cudaEventCreate(&start));
            checkCudaErrors(cudaEventCreate(&stop));

            cudaEventRecord(start);
            benchmark_memwrite<<<NUM_BLOCKS, NUM_THREADS>>>(NUM_REPEATS, enabled, dev_data);
            cudaEventRecord(stop);
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaMemcpy(host_data, dev_data, sizeof(unsigned int) * NUM_TOTAL, cudaMemcpyDeviceToHost));
            for(int i = 0; i < NUM_TOTAL; ++ i)
            {
                if(host_data[i] != NUM_REPEATS)
                {
                    fprintf(stderr, "Error at index: %i\n", i);
                    exit(-1);
                }
            }
        }
        break;

    case 2:
        {
            const long expected = 3 * NUM_REPEATS * NUM_TOTAL / WARP_SIZE;
            impl.reset(new Impl(NUMQ, (enabled) ? expected : 0));

            int N = NUM_TOTAL;
            float *x, *y, *d_x, *d_y;
            x = (float*)malloc(N*sizeof(float));
            y = (float*)malloc(N*sizeof(float));
            float bx = 1.0f, by = 2.0f;
            for (int i = 0; i < N; i++) {
                x[i] = bx;
                y[i] = by;
            }
    
            cudaMalloc(&d_x, N*sizeof(float)); 
            cudaMalloc(&d_y, N*sizeof(float));
            cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
            // Perform SAXPY on 1M elements
            float a = 2.0f;

            checkCudaErrors(cudaEventCreate(&start));
            checkCudaErrors(cudaEventCreate(&stop));
            checkCudaErrors(cudaEventRecord(start));
            benchmark_saxpy<<<NUM_BLOCKS, NUM_THREADS>>>(NUM_REPEATS, enabled, N, a, d_x, d_y);
            checkCudaErrors(cudaEventRecord(stop));
            checkCudaErrors(cudaDeviceSynchronize());
 
            cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
            checkCudaErrors(cudaDeviceSynchronize());
    
            float value = by;
            for(int i = 0; i < NUM_REPEATS; ++ i)
                value = a * bx + value;

            float maxError = 0.0f;
            for (int i = 0; i < N; i++)
                maxError = max(maxError, abs(y[i]-value));

            if(maxError > 0.1)
             printf("@Error: true\n");
        }
        break;

    case 3: // cuda sample '0_Simple/clock'
        {
            const long expected = NUM_REPEATS * NUM_TOTAL / WARP_SIZE;
            impl.reset(new Impl(NUMQ, (enabled) ? expected : 0));

            // Launch the kernel.
            float *dinput = NULL;
            float *doutput = NULL;
            clock_t *dtimer = NULL;
    
            clock_t timer[NUM_BLOCKS * 2];
            float input[NUM_THREADS * 2];
    
            for (int i = 0; i < NUM_THREADS * 2; i++)
            {   
                input[i] = (float)i;
            }
            
            checkCudaErrors(cudaEventCreate(&start));
            checkCudaErrors(cudaEventCreate(&stop));
    
            checkCudaErrors(cudaMalloc((void **)&dinput, sizeof(float) * NUM_THREADS * 2));
            checkCudaErrors(cudaMalloc((void **)&doutput, sizeof(float) * NUM_BLOCKS)); 
            checkCudaErrors(cudaMalloc((void **)&dtimer, sizeof(clock_t) * NUM_BLOCKS * 2));
    
            checkCudaErrors(cudaMemcpy(dinput, input, sizeof(float) * NUM_THREADS * 2, cudaMemcpyHostToDevice));
    
            cudaEventRecord(start);
            benchmark_timedReduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * 2 *NUM_THREADS>>>(NUM_REPEATS, enabled, dinput, doutput, dtimer);
            cudaEventRecord(stop);
    
            checkCudaErrors(cudaMemcpy(timer, dtimer, sizeof(clock_t) * NUM_BLOCKS * 2, cudaMemcpyDeviceToHost));
    
            checkCudaErrors(cudaFree(dinput));
            checkCudaErrors(cudaFree(doutput));
            checkCudaErrors(cudaFree(dtimer));

    
            // Compute the difference between the last block end and the first block start.
            clock_t minStart = timer[0];
            clock_t maxEnd = timer[NUM_BLOCKS];
    
            for (int i = 1; i < NUM_BLOCKS; i++)
            {   
                minStart = timer[i] < minStart ? timer[i] : minStart;
                maxEnd = timer[NUM_BLOCKS+i] > maxEnd ? timer[NUM_BLOCKS+i] : maxEnd;
            }
    
            printf("Total clocks = %Lf\n", (long double)(maxEnd - minStart));
        }
        break;
        
    default:
        fprintf(stderr, "Unknown test case: %i\n", TEST_ID);
        break;
    }

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("@Elapsed:\t%f\n", ms);

    return 0;
}

