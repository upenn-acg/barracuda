#pragma once

#include "CV.hpp"
#include <map>

class ThreadBlockMap
{
public:
    static ThreadBlockMap* make()
    {
        return new ThreadBlockMap();
    }

    ~ThreadBlockMap()
    {
    }

private:
    ThreadBlockMap()
    {
    }

    std::map<tid_t, CV*> _threads;
};


