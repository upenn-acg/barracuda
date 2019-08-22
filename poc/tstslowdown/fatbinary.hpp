#pragma once
#include <string>
#include <string.h>
#include <fstream>
#include <iostream>
#include "except.hpp"
#include "cudabinary.hpp"
#include "decompress.hpp"
#include "ptx_instrumentation.hpp"

static const inline int ALIGN_SIZE(int size, int alignment)
{
    return (size + alignment - 1) & ~(alignment -1);
}

class FatBinaryException: public LibHookException
{
public:
    FatBinaryException(const char* msg, ...)
    {
        va_list ap;
        va_start(ap, msg);
        format_msg(msg, ap);
    }
};

class CudaFatEntry
{
public:
    CudaFatEntry() : _entry(NULL)
    {
    }

    CudaFatEntry(Types::CudaFatEntry* entry) : _entry(entry)
    {
        switch(entry->type)
        {
        case Types::CUDA_ENTRY_TYPE_PTX:
        case Types::CUDA_ENTRY_TYPE_ELF:
        case Types::CUDA_ENTRY_TYPE_CUBIN:
            break;
        default:
            throw FatBinaryException("Unknown cuda type: %i\n", entry->type);
        }
    }

    operator Types::CudaFatEntry*() 
    {
        return _entry;
    }

    bool is_ptx() const
    {
        return _entry->type == Types::CUDA_ENTRY_TYPE_PTX;
    }

    std::string ptx_text()
    {
        char* bin_start = ((char*)_entry) + _entry->bin_ofs;
        int bin_size = (int)_entry->bin_size;
        int uncomp_size = (int)_entry->uncomp_bin_size;

        if(_entry->flags & Types::CUDA_ENTRY_FLAG_COMPRESSED_ZLIB)
            return Decompress::zlib(bin_start, bin_size, uncomp_size);
        if(_entry->flags & Types::CUDA_ENTRY_FLAG_COMPRESSED_LZ4)
            return Decompress::lz4(bin_start, bin_size, uncomp_size);
    
        return std::string(bin_start, bin_size);
    }

private:
    Types::CudaFatEntry* _entry;
};

class CudaFatIterator
{
public:
    CudaFatIterator() : _entry(NULL), _end(NULL)
    {
    }

    CudaFatIterator(Types::CudaFatEntry* entry, Types::CudaFatEntry* end) : _entry(entry), _end(end)
    {
    }

    CudaFatIterator& operator++()
    {
        if(_entry != _end)
        {
            _entry = (Types::CudaFatEntry*)(((char*)_entry) + _entry->bin_ofs + _entry->bin_size);
            if(_entry >= _end)
                _entry = _end;
        }
        return *this;
    }

    bool operator==(const CudaFatIterator& other) const
    {
        return _entry == other._entry;
    }
    bool operator!=(const CudaFatIterator& other) const
    {
        return _entry != other._entry;
    }

    CudaFatEntry operator*()
    {
        return CudaFatEntry(_entry);
    }
    CudaFatEntry operator->()
    {
        return CudaFatEntry(_entry);
    }

    CudaFatIterator(const CudaFatIterator& other) : _entry(other._entry), _end(other._end)
    {
    }

    CudaFatIterator& operator=(const CudaFatIterator& other)
    {
        _entry = other._entry;
        _end= other._end;
        return *this;
    }

private:
    Types::CudaFatEntry *_entry, *_end;
};

class CudaFatHeader
{
public:
    CudaFatHeader()
    {
    }

    CudaFatHeader(Types::CudaFatHeader* hdr) 
    {
        if(hdr->magic != Types::CUDA_FATHDR_MAGIC)
            throw FatBinaryException("Invalid cuda hdr magic!");
        char* base = (char*)hdr;
        Types::CudaFatEntry* begin = (Types::CudaFatEntry*)(base + hdr->hdr_size);
        Types::CudaFatEntry* end = (Types::CudaFatEntry*)(base + hdr->data_size);
        _begin = CudaFatIterator(begin, end);
        _end = CudaFatIterator(end, end);
    }

    CudaFatIterator begin() const { return _begin; }
    CudaFatIterator end() const { return _end; }
 
private:
    CudaFatIterator _begin, _end;
};

class FatBinary
{
public:
    FatBinary(void* bin) : 
        _newbin(NULL)
        ,_found_ptx(false)
    {
        parse_header(bin);
    }

    ~FatBinary()
    {
        if(_newbin != NULL)
           delete[] (char*)_newbin;
    }

public:
    void* instrument(bool disabled)
    {
       // XXX: leak
       if(!_found_ptx)
            throw FatBinaryException("Could not find PTX code in binary.\n");
    
        _newbin = create_instrumented_fat_binary(disabled);
        return _newbin;
    }

private:
    void parse_header(void* bin)
    {
        _bin = (Types::CudaFatBinary*)bin;
        if(_bin->magic != Types::CUDA_FAT_MAGIC)
            throw FatBinaryException("Invalid cuda bin magic!");

        _orig_hdr = (Types::CudaFatHeader*)_bin->header;        
        _hdr = CudaFatHeader(_orig_hdr);
        for(CudaFatIterator it = _hdr.begin(); it != _hdr.end(); ++ it)
        {
            CudaFatEntry entry = *it;
            if(entry.is_ptx())
            {
                _found_ptx = true;
                _ptx_entry = entry;
                _ptx_text = entry.ptx_text();
                break;
            }
        }
    }

    struct __attribute__((packed)) SingleEntryFatBin
    {
        Types::CudaFatBinary bin;
        Types::CudaFatHeader hdr;
        Types::CudaFatEntry  ptx;
    };

    std::string read_file(const std::string& filename)
    {
    	std::ifstream stream(filename);
    	std::string str;
    
    	if (!stream)
    	{
    		std::cerr << "Could not read: " << filename << std::endl;
    		exit(1);
    	}
    
    	stream.seekg(0, std::ios::end);
    	str.reserve((unsigned int)stream.tellg());
    	stream.seekg(0, std::ios::beg);
    
    	str.assign((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    	return str;
    }

	void write_file(const std::string& filename, const std::string& value)
	{
	    std::ofstream stream(filename);
		stream << value;	
	}

    Types::CudaFatBinary* create_instrumented_fat_binary(bool disabled)
    {
        size_t kind_size = sizeof(uint64_t);
	
        std::string instrumented_ptx;
		write_file("__before.ptx", _ptx_text);
        if(disabled)
            instrumented_ptx = _ptx_text;
        else
        {
    		if(!_ptx_instrument.instrument(_ptx_text, &instrumented_ptx))
    		{
    			fprintf(stderr, "Failed instrumenting binary!\n");
    			exit(1);
    		}   
        }
	    write_file("__after.ptx", instrumented_ptx);
            
        Types::CudaFatEntry* raw_entry = (Types::CudaFatEntry*)_ptx_entry;
        size_t aligned_name_size = ALIGN_SIZE(raw_entry->name_size, sizeof(uint64_t));
        size_t total_size = sizeof(SingleEntryFatBin) + kind_size + aligned_name_size + instrumented_ptx.size();
        SingleEntryFatBin* buffer = (SingleEntryFatBin*)(new char[ALIGN_SIZE(total_size, sizeof(uint64_t))]);
        
        buffer->bin.magic = Types::CUDA_FAT_MAGIC;
        buffer->bin.version = Types::CUDA_FAT_VERSION;
        buffer->bin.name = _bin->name;
        buffer->bin.header = &buffer->hdr;
        buffer->hdr.magic = Types::CUDA_FATHDR_MAGIC;
        buffer->hdr.version = Types::CUDA_FATHDR_VERSION;
        buffer->hdr.hdr_size = sizeof(buffer->hdr);
        
        char* src = (char*)raw_entry;
        char* dst = (char*)&buffer->ptx;
        memcpy(dst, src, sizeof(*raw_entry));
        buffer->ptx.flags &= ~Types::CUDA_COMPRESS_MASK;

        buffer->ptx.kind_ofs = sizeof(*raw_entry);
        memcpy(dst + buffer->ptx.kind_ofs, src + raw_entry->kind_ofs, kind_size);
        
        buffer->ptx.name_ofs = buffer->ptx.kind_ofs + kind_size;
        memcpy(dst + buffer->ptx.name_ofs, src + raw_entry->name_ofs, buffer->ptx.name_size);
        
        buffer->ptx.bin_ofs = buffer->ptx.name_ofs + aligned_name_size;
        buffer->ptx.bin_size = instrumented_ptx.size();
        buffer->ptx.uncomp_bin_size = instrumented_ptx.size();
        memcpy(dst + buffer->ptx.bin_ofs, instrumented_ptx.c_str(), buffer->ptx.bin_size);
        
        buffer->hdr.data_size = buffer->ptx.bin_ofs + buffer->ptx.bin_size;

        return &buffer->bin;
    }

private:
	PtxInstrumentation _ptx_instrument;
    Types::CudaFatBinary* _bin;
    Types::CudaFatBinary* _newbin; 
    Types::CudaFatHeader* _orig_hdr;

    CudaFatHeader _hdr;
    bool _found_ptx;
    CudaFatEntry _ptx_entry;
    std::string _ptx_text;
};
