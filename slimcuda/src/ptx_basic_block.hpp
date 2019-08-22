#pragma once

#include "ptx_program.hpp"

class PtxBasicBlock
{
public:
	PtxBasicBlock(const std::shared_ptr<PtxFunction>& func, 
				  const std::shared_ptr<PtxBlock>& code_block,
				  const PtxBlock::list_iterator& begin,
				  PtxBlock::list_iterator end
		) : _function(func)
		, _code_block(code_block)
		, _begin(begin)
		, _end(end)
	{
	}

	PtxBlock::list_iterator begin() { return _begin; }
	PtxBlock::list_iterator end() { return _end; }

	void add_instruction(PtxBlock::list_iterator where, const std::shared_ptr<PtxCodeElement>& line)
	{
		_code_block->elements().insert(where, line);
	}
	void replace_instructions(PtxBlock::list_iterator where, PtxBlock::list_iterator first, PtxBlock::list_iterator last)
	{
		_code_block->elements().insert(where, first, last);
        *where = std::shared_ptr<PtxCodeElement>();
	}
    std::shared_ptr<PtxBlock> block()
    {
        return _code_block;
    }

private:
	std::shared_ptr<PtxFunction> _function;
	std::shared_ptr<PtxBlock> _code_block;
	PtxBlock::list_iterator _begin, _end;
};

class PtxFunctionAnalyzer
{
public:
	PtxFunctionAnalyzer(const std::shared_ptr<PtxFunction>& func) :
		_func(func)
	{
		analyze(_func->main());
	}

	using list_type = std::list<std::shared_ptr<PtxBasicBlock>>;
	using list_iterator = list_type::iterator;

	list_iterator begin() { return _blocks.begin(); }
	list_iterator end() { return _blocks.end(); }

private:
	void analyze(const std::shared_ptr<PtxBlock>& block)
	{
		PtxBlock::list_iterator begin = block->begin(), end = block->end();
		PtxBlock::list_iterator current = begin;

		for(;current != end; ++ current)
		{
			if (is_leader(*current))
			{
				if (current == begin)
                    continue;
				_blocks.push_back(std::make_shared<PtxBasicBlock>(_func, block, begin, current));
				if ((*current)->type() == PtxBlock::Type::TypeBlock)
                {
					analyze(std::static_pointer_cast<PtxBlock>(*current));
                    ++ current;
                }
				begin = current;
			}
		}
		if(current != begin)
			_blocks.push_back(std::make_shared<PtxBasicBlock>(_func, block, begin, current));
	}

	bool is_leader(std::shared_ptr<PtxCodeElement>& ptr)
	{
		switch (ptr->type())
		{
		case PtxBlock::Type::TypeBlock:
		case PtxBlock::Type::TypeLabel:
			return true;
		case PtxBlock::Type::TypeVariable:
		case PtxBlock::Type::TypeDirective:
			return false;
		case PtxBlock::Type::TypeCode:
			return is_leader_instruction(std::static_pointer_cast<PtxCodeLine>(ptr));
		}
		assert(false);
		return false;
	}

	bool is_leader_instruction(const std::shared_ptr<PtxCodeLine>& ptr)
	{
		static const std::initializer_list<std::string> INSTRUCTIONS = {
			//"bar", "membar", "atom", "red", "vote",  // parallel
		//	"bar", "membar", "atom", "red", "vote",  // parallel
			"bra", "ret", "exit",
		};

		return find_if(INSTRUCTIONS.begin(), INSTRUCTIONS.end(), [&](auto& s) {return s == ptr->instruction(); }) != INSTRUCTIONS.end();
	}

private:
	std::shared_ptr<PtxFunction> _func;
	list_type _blocks;
};
