#pragma once

#ifdef __CUDACC__

#ifndef BFUNC
#define BFUNC __host__ __device__
#endif

#ifndef NOINLINE
#define NOINLINE __attribute__((noline))
#endif

#ifndef FINLINE
#define FINLINE __forceinline__ __attribute((always_inline))
#endif

#else // ! __CUDACC__

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

static void __debugbreak() { asm volatile("int $3"); }
