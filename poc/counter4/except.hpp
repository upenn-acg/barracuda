#pragma once

#include <stdexcept>
#include <stdarg.h>
#include <string>

class LibHookException : public std::exception
{
public:
    LibHookException(const char* msg, ...)
    {
        va_list ap;
        va_start(ap, msg);
        format_msg(msg, ap);
    }

    virtual const char* what() const _GLIBCXX_USE_NOEXCEPT
    {
        return _buffer;
    }
    
protected:
    LibHookException()
    {
    }

    void format_msg(const char* msg, va_list& ap)
    {
        vsnprintf(_buffer, sizeof(_buffer) - 1, msg, ap);
        _buffer[sizeof(_buffer) - 1] = 0;
    }
    
private:
    char _buffer[1024];
};


