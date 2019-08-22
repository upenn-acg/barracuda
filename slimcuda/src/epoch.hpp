#pragma once

#include <stdint.h>
#include "debug.h"

class Epoch
{
public: 
	inline operator uint32_t() const
	{
		return value;
	}

	FINLINE uint32_t inc()
	{
		return ++value;
	}

	FINLINE void inc_from(const Epoch epoch)
	{
		value = epoch.value + 1;
	}

	FINLINE void max_of(const Epoch epoch)
	{
		if (value < epoch.value)
			value = epoch.value;
	}
	FINLINE void max_of(const uint32_t epoch)
	{
		if (value < epoch)
			value = epoch;
	}
	FINLINE void max_both(Epoch epoch)
	{
		if (value < epoch.value)
			value = epoch.value;
		else
			epoch.value = value;
	}

	uint32_t value;
};