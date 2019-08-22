#pragma once

#include <map>
#include <stdint.h>
#include <unordered_map>

typedef uint64_t tid_t;
typedef uint32_t epoch_t;


class mem_cv : public std::unordered_map<tid_t, epoch_t>
{
public:
	mem_cv()
	{
	}

	~mem_cv()
	{
	}
};


