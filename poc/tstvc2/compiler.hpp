#pragma once

#ifdef __CUDACC__

#ifndef BFUNC
#define BFUNC __host__ __device__
#endif

#ifndef NOINLINE
#define NOINLINE __declspec(noinline)
#endif

#ifndef FINLINE
#define FINLINE inline
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

#ifdef _MSC_VER
uint32_t __sync_lock_test_and_set(unsigned *ptr, unsigned value)
{
	return InterlockedExchange(ptr, (unsigned)value);
}

void __sync_lock_release(unsigned *ptr)
{
	InterlockedExchange(ptr, 0);
}
#else
void __debugbreak() { asm volatile("int $3;\n"); }
#endif
