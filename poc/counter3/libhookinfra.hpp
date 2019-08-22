#pragma once

static inline void _failed(int err, const char* msg, const char* file, int line)
{
    fprintf(stderr,"Function failed at %s:%i: '%s', err=%i\n", file, line, msg, err); 
    if(getenv("DBGBREAK") != NULL) asm volatile("int $3");
    exit(1);
}

#define VERIFY(f) for(;;) { int val = (f); if(cudaSuccess != val) _failed(val, #f, __FILE__, __LINE__); break;}

class _Hook_BASE
{
public:
    static void load_hooks(void* handle)
    {
        static bool init = false;

        if(init)
            return;

        for(_Hook_BASE* cur = _first; cur != NULL; cur = cur->_next)
        {
            cur->load_orig(handle);
        }
        init = true;
    }
        
protected:
    static _Hook_BASE* _first;
    _Hook_BASE* _next;
    typedef void (*func_t)(void); 
    func_t func;
    const char* _name;

    _Hook_BASE(const char* name) : _name(name)
    {
        _next = _first;
        _first = this;
    }

    void load_orig(void* handle) 
    {
    	func = (func_t)dlsym(handle, _name);
    	if(func == NULL)
    	{
    		fprintf(stderr, "Failed loading function: '%s'\n", _name);
    		exit(1);
    	}
    }
};

_Hook_BASE* _Hook_BASE::_first = NULL;

#define DECLARE_HOOK_FUNC(name, args)                         \
    class _Hook_##name : public _Hook_BASE {                  \
    public:                                                   \
        typedef cudaError_t (*myfunc_t)args;                  \
        inline myfunc_t get() { return (myfunc_t)func; }      \
        _Hook_##name() : _Hook_BASE(#name) { }                \
    };                                                        \
    static _Hook_##name _theHook_##name;

#define DECLARE_HOOK_RETFUNC(name, ret, args)                 \
    class _Hook_##name : public _Hook_BASE {                  \
    public:                                                   \
        typedef ret (*myfunc_t)args;                          \
        inline myfunc_t get() { return (myfunc_t)func; }      \
        _Hook_##name() : _Hook_BASE(#name) { }                \
    };                                                        \
    static _Hook_##name _theHook_##name;

#define ORIG(name) _theHook_##name.get()
    
