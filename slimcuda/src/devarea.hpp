#pragma once

#include "protocol.hpp"
#include "compiler.hpp"

class DeviceArea
{
public:
    enum { HEADER_SIZE = 64 }; // max(sizeof(PCHeader), alignment(PCRecord));

    BFUNC FINLINE DeviceArea()
    {
    }

    BFUNC FINLINE ~DeviceArea()
    {
    }

    BFUNC FINLINE DeviceArea(void* base, size_t size, unsigned int numq)
    {
        _base = (char*)base;
        _size = size;
        _numq = numq;
        _qbuf_size = (size - 1- numq * HEADER_SIZE) / sizeof(PCRecord) / numq;
    }

    BFUNC FINLINE void* base() const
    {
        return _base;
    }

    BFUNC FINLINE size_t size() const
    {
        return _size;
    }

    BFUNC FINLINE size_t qbuf_size() const
    {
        return _qbuf_size;
    }

    BFUNC FINLINE unsigned int numq() const
    {
        return _numq;
    }

    BFUNC FINLINE PCHeader* header(unsigned int qid) const
    {
        return (PCHeader*)(_base + qid * (HEADER_SIZE + _qbuf_size * sizeof(PCRecord)));
    }

    static BFUNC FINLINE PCRecord* start(PCHeader* hdr)
    {
        return (PCRecord*)(((char*)hdr) + HEADER_SIZE);
    }

private:
    char* _base;
    size_t _size;
    size_t _qbuf_size;
    unsigned int _numq;
};

