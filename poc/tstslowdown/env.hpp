#pragma once

#include <stdlib.h>

class Env
{
public:
    static inline bool is_disabled()
    {
        return getenv("SC_DISABLED") != NULL;
    }

    static inline bool dont_instrument_funcs()
    {
        return getenv("SC_NOFUNCS") != NULL;
    }
};


