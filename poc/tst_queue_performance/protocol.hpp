#pragma once

static const int WARP_SIZE = 32;

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
} __attribute__((aligned (64)));

#define BUILD_ADDRESS(stream, block, warp) (((uint64_t)stream << 48) | ((uint64_t)block << 16) | (warp >> 5))

struct PCHeader
{
    unsigned int read_head;
    unsigned int write_head;
    unsigned int tail;
};

