#pragma once

#ifdef __CUDACC__

#ifndef BFUNC
#define BFUNC __host__ __device__
#endif

#ifndef NOINLINE
#define NOINLINE __attribute__((noline))
#endif

#ifndef FINLINE
#define FINLINE __forceinline__
#endif

#else // ! __CUDACC__

namespace std {

// Add make_unique to C++11
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}

#ifndef BFUNC
#define BFUNC 
#endif

#ifndef NOINLINE
#define NOINLINE __attribute__((noline))
#endif

#ifndef FINLINE
#define FINLINE inline
#endif

#endif //! __CUDACC__ 
