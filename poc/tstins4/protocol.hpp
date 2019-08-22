#pragma once

#include <stdint.h>

static const int WARP_SIZE = 32;

#define ALIGN(type, ptr, alignment) (type*)(((uintptr_t)(ptr) + (alignment) - 1) & ~((alignment) - 1))


enum PROTO_OP
{
    OP_UNKNOWN           = 0,
    OP_READ              = 1,
    OP_WRITE             = 2,
    OP_READWRITE         = 3,
    OP_ATOMIC_WRITE      = 4,
    OP_ATOMIC_READWRITE  = 5,
    OP_SYNCTHREADS       = 6,
    PROTO_OP_MAX
};

typedef uint64_t slimptr_t;

struct PCRecord
{
    unsigned long  tid;
    unsigned int   active; 
    unsigned int   op; 
    slimptr_t address[WARP_SIZE];
} __attribute__((aligned (32)));

#define BUILD_ADDRESS(stream, block, warp) (((uint64_t)stream << 48) | ((uint64_t)block << 16) | (warp >> 5))

struct PCHeader
{
    unsigned int read_head;
    unsigned int write_head;
    unsigned int tail;
};

struct _cuda_dim3
{
    unsigned int x, y, z;
};

#define NAMEOF(x) #x

#define INIT_FUNCTION_NAME __slimcuda_init
#define NAMEOF_INIT_FUNCTION_NAME NAMEOF(INIT_FUNCTION_NAME)

#define GETTID_FUNCTION_NAME __slimcuda_gettid
#define NAMEOF_GETTID_FUNCTION_NAME NAMEOF(GETTID_FUNCTION_NAME)


