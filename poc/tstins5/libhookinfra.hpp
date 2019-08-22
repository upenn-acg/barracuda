#pragma once

#define VERIFY(f) for(;;) { if(cudaSuccess != (f)) { fprintf(stderr,"Function failed: '%s'\n", #f); exit(1); } break;}

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
    
