#pragma once

#include "threadblocks.hpp"
#include <map>

class StreamMap
{
public:
    static StreamMap* make()
    {
        return new StreamMap();
    }

    ~StreamMap()
    {
    }

private:
    StreamMap()
    {
    }

private:
    typedef std::map<unsigned short, ThreadBlockMap*> MapType;
    MapType _map;
};

