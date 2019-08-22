#pragma once

#include <stdlib.h>

class Env
{
public:
    enum { DEFAULT_NUM_Q = 16 };

    static inline bool is_disabled()
    {
        return getenv("SC_DISABLED") != NULL;
    }

    static inline bool dont_instrument_funcs()
    {
        return getenv("SC_NOFUNCS") != NULL;
    }
    static inline bool gpu_only()
    {
        return getenv("SC_GPUONLY") != NULL;
    }
    static inline bool manually_instrumented()
    {
        return getenv("SC_MANUAL") != NULL;
    }
    static inline bool compile_time_ptx_instrumentation()
    {
        return getenv("SC_NOINS") != NULL;
    }

    static inline bool is_single_stream()
    {
        return true;
    }

    static inline int get_num_queues()
    {
        return get_int("SC_NUMQ", DEFAULT_NUM_Q, 1);
    }

    static int get_verbose_level(int def)
    {
        return get_int("SC_DBGLVL", def, 0);
    }

    static bool supress_prints()
    {
        return getenv("SC_QUIET") != NULL;
    }


private:
    static inline int get_int(const char* env, int def, int minvalue)
    {
        char* numstr = getenv(env);
        if(numstr != NULL)
        {
            int val = atoi(numstr);
            if(val >= minvalue)
                return val;
                
        }
        return def;
    }
};


