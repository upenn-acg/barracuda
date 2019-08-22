#pragma once

static const int WARP_SIZE = 32;

enum PROTO_OP
{
    OP_UNKNOWN           = 0,
    OP_READ              = 1,
    OP_WRITE             = 2,
    OP_ATOMIC_WRITE      = 3,
    OP_ATOMIC_READWRITE  = 4,
    OP_SYNCTHREADS       = 5,
    PROTO_OP_MAX
};

struct PCRecord
{
    unsigned short warp_id;
    unsigned short block_id;
    unsigned short stream_id;
    unsigned short op;
    unsigned int   active; 
    void *address[WARP_SIZE];
};

struct PCHeader
{
    unsigned int read_head;
    unsigned int write_head;
    unsigned int tail;
};

