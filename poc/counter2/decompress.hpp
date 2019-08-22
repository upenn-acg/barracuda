#pragma once
#include <string>
#include <vector>
#include <zlib.h>
#include <lz4.h>
#include "except.hpp"

class Decompress
{
public:
    class DecompressError : public LibHookException
    {
    public:
        DecompressError(const char* msg, ...)
        {
            va_list ap;
            va_start(ap, msg);
            format_msg(msg, ap);
        }
    };

    static std::string zlib(const char* start, int comp_size, int uncomp_size)
    {
        std::vector<char> buffer(uncomp_size);
        uLongf destlen = uncomp_size;
        int result = uncompress((Bytef*)&buffer[0], &destlen, (const Bytef*)start, comp_size);
        if(result != Z_OK)
            throw DecompressError("Could not decompress zlib: %i\n", result);
        return std::string(&buffer[0], uncomp_size);
    }

    static std::string lz4(const char* start, int comp_size, int uncomp_size)
    {
        std::vector<char> buffer(uncomp_size);
        int result = LZ4_decompress_safe(start, &buffer[0], comp_size, uncomp_size);
        if(result < 0) // HACK 
        {
            for(--comp_size ; comp_size > 0 && start[comp_size] == 0; -- comp_size)
                ;
            comp_size += 1;
            result = LZ4_decompress_safe(start, &buffer[0], comp_size, uncomp_size);
        }

        if(result < 0)
            throw DecompressError("Could not decompress LZ4: %i\n", result);
        
        return std::string(&buffer[0], uncomp_size);
    }

};

