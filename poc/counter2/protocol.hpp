#pragma once
#include <stdint.h>

static const int WARP_SIZE = 32;

enum PROTO_OP
{
    OP_UNKNOWN           = 0,
    OP_LOAD              = 1,
    OP_STORE             = 2,
    OP_ATOMIC			 = 3,
    OP_SYNCTHREADS       = 4,
	OP_FENCE_CTA		 = 5,
	OP_FENCE_GL			 = 6,
	OP_FENCE_SYS         = 7,
    OP_CONVERGE          = 8,
    OP_OTHER             = 9,
    PROTO_OP_MAX
};

typedef uint64_t slimptr_t;

struct PCRecord
{
    uint64_t  tid;
    uint32_t  active; 
    uint16_t  op; 
    uint16_t  predicated;
    slimptr_t address[WARP_SIZE];
};

#define BUILD_ADDRESS(stream, block, warp) (((uint64_t)stream << 48) | ((uint64_t)block << 16) | (warp >> 5))

struct PCHeader
{
    uint32_t read_head;
    uint32_t write_head;
    uint32_t tail;
};

