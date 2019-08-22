#pragma once
#include <stdint.h>

static const int WARP_SIZE = 32;

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
    uint64_t  tid;
    uint32_t  active; 
    uint16_t  op_state;
    uint16_t  loc_id;
    slimptr_t address[WARP_SIZE];
};

#define BUILD_ADDRESS(stream, block, warp) (((uint64_t)stream << 48) | ((uint64_t)block << 16) | (warp >> 5))

struct PCHeader
{
    uint32_t read_head;
    uint32_t write_head;
    uint32_t tail;
};

