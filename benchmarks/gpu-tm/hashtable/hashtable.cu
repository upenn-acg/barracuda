/* Implements a threadsafe binary heap for use on a GPU
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// includes, project
#include "cutil.h"
#include "../../benchmark.h"

#define BASETABLE_SIZE_MAX		1000000
#define MAX_ENTRIES		65536

__device__ __noinline__ void __tbegin() { }
__device__ __noinline__ void __tcommit() { }

typedef int key_t;
typedef int value_t;

struct TableEntry {
	key_t mKey;
	value_t mValue;
	unsigned mNext;
};

typedef struct TableEntry tTableEntry;

struct BaseEntry {
       unsigned mIndex;
       int mLock;
};

struct HashTable {
	BaseEntry mValues[BASETABLE_SIZE_MAX];
};

typedef struct HashTable tHashTable;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, const char** argv);
__global__ void kernel_buildtable_tm( tHashTable* g_hashtable, tTableEntry* g_entrypool, unsigned* g_keys, int g_baseTableSize );
__global__ void kernel_buildtable_atomic( tHashTable* g_hashtable, tTableEntry* g_entrypool, unsigned* g_keys, int g_baseTableSize );

extern "C"
int computeGold( int* gpuData, const int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, const char** argv)
{
    runTest( argc, argv);

    CUT_EXIT(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest( int argc, const char **argv)
{
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev;
    CUT_DEVICE_INIT(argc, argv);
    CUDA_SAFE_CALL(cudaChooseDevice(&dev, &deviceProp));
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));

    if(deviceProp.major > 1 || deviceProp.minor > 0)
    {
//        printf("Using Device %d: \"%s\"\n", dev, deviceProp.name);
        CUDA_SAFE_CALL(cudaSetDevice(dev));
    }
    else
    {
        printf("There is no device supporting CUDA compute capability 1.1. Hopefully using emu\n");
        //CUT_EXIT(argc, argv);
    }

    bool useTM = false;
    useTM = cutCheckCmdLineFlag(argc, argv, "tm"); 

//    unsigned int timer = 0;
//    CUT_SAFE_CALL( cutCreateTimer( &timer));
//    CUT_SAFE_CALL( cutStartTimer( timer));

    srand(2011);  // set seed for rand()

    int numThreads = 192;
    int numBlocks = 120;
    int baseTableSize = 8192;
    cutGetCmdLineArgumenti(argc, argv, "numThreads", &numThreads); 
    cutGetCmdLineArgumenti(argc, argv, "numBlocks", &numBlocks); 
    cutGetCmdLineArgumenti(argc, argv, "hashEntries", &baseTableSize);
    assert(numThreads >= 0); 
    assert(numBlocks >= 0); 
    assert(MAX_ENTRIES > (numThreads * numBlocks + 1));
    assert(baseTableSize >= 0 && baseTableSize <= BASETABLE_SIZE_MAX);
    
//    printf("Number of hash entries = %u\n", baseTableSize);
//    printf("Number of threads = %u\n", numThreads*numBlocks);

    // allocate host copy:
    tHashTable* h_hashtable = (tHashTable*)(calloc(1, sizeof(tHashTable)));
    tTableEntry* h_entries = (tTableEntry*)(calloc(MAX_ENTRIES, sizeof(tTableEntry)));
    unsigned* h_keys = (unsigned*)(calloc(MAX_ENTRIES, sizeof(unsigned)));

    // and device copy
    tHashTable* d_hashtable;
    tTableEntry* d_entries;
    unsigned* d_keys;

    // Build keys
    for(unsigned i=0; i<MAX_ENTRIES; i++) {
        h_keys[i] = rand();
    }

    BENCHMARK.start_total();

    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_hashtable, sizeof(tHashTable)));
    CUDA_SAFE_CALL( cudaMemcpy( d_hashtable, h_hashtable, sizeof(tHashTable), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_entries, sizeof(tTableEntry)*MAX_ENTRIES ) );    
    CUDA_SAFE_CALL( cudaMemcpy( d_entries, h_entries, sizeof(tTableEntry)*MAX_ENTRIES, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_keys, sizeof(unsigned)*MAX_ENTRIES ) );
    CUDA_SAFE_CALL( cudaMemcpy( d_keys, h_keys, sizeof(unsigned)*MAX_ENTRIES, cudaMemcpyHostToDevice) );
    
    // execute the first kernel, this throws some data into the kernel for testing...
    if(useTM) {
       kernel_buildtable_tm<<<numBlocks, numThreads>>>(d_hashtable, d_entries, d_keys, baseTableSize);
    } else {
        BENCHMARK.start_kernel();
       kernel_buildtable_atomic<<<numBlocks, numThreads>>>(d_hashtable, d_entries, d_keys, baseTableSize);
        BENCHMARK.end_kernel();
    }

    CUT_CHECK_ERROR("Kernel execution failed");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    //Copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( h_hashtable, d_hashtable, sizeof(tHashTable),
    cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy( h_entries, d_entries,
    sizeof(tTableEntry)*MAX_ENTRIES, cudaMemcpyDeviceToHost) );

    BENCHMARK.end_total();
    #ifdef DEBUG
    for( int i = 0; i < MAX_ENTRIES; ++i ) {
        if( h_entries[i].mValue ) 
            printf(" %u : %u ->  %u\n", i, h_entries[i].mKey, h_entries[i].mValue );
    }
    #endif 

    #define DEBUG
    // error checking 
    int nInsertedEntries = 0; 
    for (int h = 0; h < baseTableSize; h++) {
        unsigned entry_id = h_hashtable->mValues[h].mIndex; 
        while (entry_id != 0) {
            tTableEntry& tentry = h_entries[entry_id]; 
            #ifdef DEBUG
            if ( (tentry.mKey != h_keys[tentry.mValue]) || (tentry.mKey % baseTableSize != h)) {
               printf(" table[%d] -> %u : %u -> %u\n", h, entry_id, tentry.mKey, tentry.mValue); 
            }
            #else
            assert(tentry.mKey == h_keys[tentry.mValue]); // key-value consistency
            assert((tentry.mKey % baseTableSize) == h); // key-hash consistency
            #endif
            if(tentry.mKey != h_keys[tentry.mValue])
                BENCHMARK.fail();
            if((tentry.mKey % baseTableSize) != h)
                BENCHMARK.fail();
            entry_id = tentry.mNext; 
            nInsertedEntries += 1;
        }
    }
//    printf("nInsertedEntries = %d\n", nInsertedEntries); 
    assert(nInsertedEntries == (numThreads * numBlocks)); 
    if(nInsertedEntries != (numThreads * numBlocks))
        BENCHMARK.fail();

//    CUT_SAFE_CALL( cutStopTimer( timer));
//    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
//    CUT_SAFE_CALL( cutDeleteTimer( timer));
   
    // cleanup memory
    free(h_hashtable);
    CUDA_SAFE_CALL(cudaFree(d_hashtable));

//    printf("TEST PASSED\n"); 
}

__device__ void add_to_hash_tm( tHashTable *g_hashtable, tTableEntry* g_entrypool, unsigned key, unsigned value, int g_baseTableSize )
{
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned hash = key % g_baseTableSize;
    unsigned pool_slot = tid+1; // reserve zero for null
 
	BaseEntry *base = &g_hashtable->mValues[hash];
   tTableEntry *ent = &g_entrypool[pool_slot]; 
   
   // need something fancier if each thread can have more than one entry
   // TODO: implement something like Hoard (ASPLOS 2000) for CUDA
   ent->mKey = key;
   ent->mValue = value;
   ent->mNext = base->mIndex;
   g_hashtable->mValues[hash].mIndex = pool_slot;
}

__device__ void add_to_hash_atomic( tHashTable *g_hashtable, tTableEntry* g_entrypool, unsigned key, unsigned value, int g_baseTableSize )
{
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned hash = key % g_baseTableSize;
    unsigned pool_slot = tid+1; // reserve zero for null
 
	BaseEntry *base = &g_hashtable->mValues[hash];
   tTableEntry *ent = &g_entrypool[pool_slot]; 
   
   // need something fancier if each thread can have more than one entry
   // TODO: implement something like Hoard (ASPLOS 2000) for CUDA

   unsigned done = 0;
   while ( !done ) {
      if(atomicCAS(&base->mLock, 0, 1) == 0) {

         ent->mKey = key;
         ent->mValue = value;
         ent->mNext = base->mIndex;
         g_hashtable->mValues[hash].mIndex = pool_slot;

         base->mLock = 0;
         done = 1;
      }
   }
}

__global__ void kernel_buildtable_tm( tHashTable* g_hashtable, tTableEntry* g_entrypool, unsigned* g_keys, int g_baseTableSize ) 
{
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned key, value;

   __tbegin();
   key = g_keys[tid+1];
   value = tid+1;
   add_to_hash_tm(g_hashtable,g_entrypool,key,value,g_baseTableSize); 
   __tcommit();
}

__global__ void kernel_buildtable_atomic( tHashTable* g_hashtable, tTableEntry* g_entrypool, unsigned* g_keys, int g_baseTableSize ) 
{
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned key, value;

   key = g_keys[tid+1];
   value = tid+1;
   add_to_hash_atomic(g_hashtable,g_entrypool,key,value,g_baseTableSize); 
}
