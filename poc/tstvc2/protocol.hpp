#pragma once
#include <stdint.h>
#include "compiler.hpp"

static const int WARP_SHIFT = 5;
static const int WARP_SIZE = 1 << WARP_SHIFT;
static const int BLOCK_SHIFT = 16;
static const int STREAM_SHIFT = 48;

BFUNC static FINLINE uint64_t build_tid(uint32_t stream, uint32_t block, uint32_t btid)
{
    return (((uint64_t)stream) << STREAM_SHIFT) 
         | (((uint64_t)block) << BLOCK_SHIFT) 
         | (((uint64_t)btid) >> WARP_SHIFT);
}

BFUNC FINLINE static uint32_t tid_to_stream(uint64_t tid)
{
    return (uint32_t)(tid >> STREAM_SHIFT);
}

BFUNC FINLINE static uint64_t tid_to_wid(uint64_t tid)
{
    return tid & ~((((uint64_t)1) << STREAM_SHIFT) - 1);
}

#define F_UNKNOWN 0x00
#define F_CTA 0x01
#define F_GL  0x02
#define F_SYS 0x03
#define F_ATOMIC 0x04

enum PROTO_OP
{
    OP_OTHER             = 0x00,
    OP_LOAD              = 0x01,
    OP_STORE             = 0x02,
    OP_ATOMIC			 = 0x03,
    OP_SYNCTHREADS       = 0x04,
    OP_CONVERGE          = 0x05,
    OP_START_KERNEL      = 0x06,
    OP_ACQUIRE           = 0x10,
	OP_ACQ_CTA		     = OP_ACQUIRE | F_CTA,
	OP_ACQ_GL			 = OP_ACQUIRE | F_GL,
	OP_ACQ_SYS           = OP_ACQUIRE | F_SYS,
	OP_ATOM_ACQ_CTA	     = OP_ACQUIRE | F_CTA | F_ATOMIC,
	OP_ATOM_ACQ_GL		 = OP_ACQUIRE | F_GL  | F_ATOMIC,
	OP_ATOM_ACQ_SYS      = OP_ACQUIRE | F_SYS | F_ATOMIC,
    OP_RELEASE           = 0x20,
	OP_REL_CTA		     = OP_RELEASE | F_CTA,
	OP_REL_GL			 = OP_RELEASE | F_GL,
	OP_REL_SYS           = OP_RELEASE | F_SYS,
	OP_ATOM_REL_CTA	     = OP_RELEASE | F_CTA | F_ATOMIC,
	OP_ATOM_REL_GL		 = OP_RELEASE | F_GL  | F_ATOMIC,
	OP_ATOM_REL_SYS      = OP_RELEASE | F_SYS | F_ATOMIC,
    OP_ACQREL            = 0x30,
	OP_ATOM_ACQREL_CTA	 = OP_ACQREL | F_CTA | F_ATOMIC,
	OP_ATOM_ACQREL_GL	 = OP_ACQREL | F_GL  | F_ATOMIC,
	OP_ATOM_ACQREL_SYS   = OP_ACQREL | F_SYS | F_ATOMIC,
    PROTO_OP_MAX
};

static const int GLOBAL_FLAG = 0x0100;

typedef uint64_t slimptr_t;

struct PCRecord
{
    uint64_t  wid;
    uint32_t  active; 
    uint16_t  op; 
    uint16_t  loc_id;
    slimptr_t address[WARP_SIZE];
};


struct PCHeader
{
    uint32_t read_head;
    uint32_t write_head;
    uint32_t tail;
};

