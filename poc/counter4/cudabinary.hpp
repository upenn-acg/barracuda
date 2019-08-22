#pragma once

#include <stdint.h>

namespace Types
{
    static const uint32_t CUDA_FAT_MAGIC = 0x466243b1;
    static const uint32_t CUDA_FAT_VERSION = 1;
    static const uint32_t CUDA_FATHDR_MAGIC = 0xba55ed50;
    static const uint32_t CUDA_FATHDR_VERSION = 1;

    struct CudaFatBinary
    {
        uint32_t magic;
        uint32_t version;
        void*    header;
        char*    name;
    };

    struct CudaFatHeader
    {
        uint32_t magic;
        uint16_t version;
        uint16_t hdr_size;
        uint64_t data_size;
    };
    
    struct CudaFatEntry
    {
        uint16_t type;
        uint16_t unk1;
        uint32_t bin_ofs;
        uint64_t bin_size;
        uint32_t unk2;
        uint32_t kind_ofs;
        uint32_t code_ver;
        uint32_t arch;
        uint32_t name_ofs;
        uint32_t name_size;
        uint64_t flags;
        uint64_t unk3;
        uint64_t uncomp_bin_size;
    };

    enum CUDA_ENTRY_FLAGS {
        CUDA_ENTRY_FLAG_64BIT            = 0x1,
        CUDA_ENTRY_FLAG_DEBUG            = 0x2,
        CUDA_ENTRY_FLAG_HOST_LINUX       = 0x10,
        CUDA_ENTRY_FLAG_HOST_MAC         = 0x20,
        CUDA_ENTRY_FLAG_HOST_WINDOWS     = 0x40,
        CUDA_ENTRY_FLAG_COMPRESSED_ZLIB  = 0x1000,
        CUDA_ENTRY_FLAG_COMPRESSED_LZ4   = 0x2000,
    };

    static const uint64_t CUDA_COMPRESS_MASK = CUDA_ENTRY_FLAG_COMPRESSED_LZ4 | CUDA_ENTRY_FLAG_COMPRESSED_ZLIB;

    enum CUDA_ENTRY_TYPES {
         CUDA_ENTRY_TYPE_PTX = 0x1,
         CUDA_ENTRY_TYPE_ELF = 0x2,
         CUDA_ENTRY_TYPE_CUBIN = 0x4,
    };

}

