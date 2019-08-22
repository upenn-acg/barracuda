#pragma once

#include "ptx_program.hpp"
#include "ptx_parser.hpp"
#include "ptx_basic_block.hpp"
#include "protocol.hpp"
#include "ptx_stub.h"
#include "env.hpp"

#include <iostream>
#include <streambuf>
#include <sstream>
#include <memory>

extern "C"
{
    extern const char _binary_ptx_stub_ptx_start;
    extern const char _binary_ptx_stub_ptx_end;
    extern const int  _binary_ptx_stub_ptx_size;
}

class PtxInstrumentation
{
public:
	PtxInstrumentation()
	{
		load_stub(std::string(&_binary_ptx_stub_ptx_start, &_binary_ptx_stub_ptx_end));
	}

	~PtxInstrumentation()
	{

	}

	bool instrument(const std::string& ptx_text, std::string* out)
	{
		std::stringstream stream;

		PtxParser parser;
		PtxParser::ParseStatus status;
		if (!parser.parse_ptx(ptx_text, &status))
		{
			std::cerr << "Could not parse PTX at byte: " << status.parsed << std::endl;
			return false;
		}

		PtxProgram& ptx = parser.ptx();
		if (! (verify_target(ptx)
			   && instrument_funcs(ptx)
			   && add_stub_code(ptx)))
			return false;


		ptx.print(stream);
		*out = stream.str();

		return true;
	}

private:
	bool instrument_funcs(PtxProgram& ptx)
	{
        if(Env::dont_instrument_funcs())
            return true;

        int loc_id = 0;

		for (auto& d : ptx)
		{
			if (d->dir_type() == PtxDirective::DirType::Function)
			{
				if (!instrument_func(std::static_pointer_cast<PtxFunction>(d), &loc_id))
					return false;
			}
		}
		return true;
	}

	bool instrument_func(const std::shared_ptr<PtxFunction>& func, int* loc_id)
	{
        bool instrumented = false;
		PtxFunctionAnalyzer analyzer(func);
        
		for (auto& bb : analyzer)
		{
			for (auto it = bb->begin(); it != bb->end(); )
			{
                auto cur = it ++;
                auto proto = OP_UNKNOWN;
                std::shared_ptr<PtxCodeLine> codeline;
				switch((*cur)->type())
                {
                case PtxCodeElement::Type::TypeCode:
				    codeline = std::static_pointer_cast<PtxCodeLine>(*cur);
                    proto = get_proto(codeline);
                    break;
                case PtxCodeElement::Type::TypeLabel:
                    proto = OP_CONVERGE;
                    break;
                default:
                    break;
                }
                if(proto == OP_UNKNOWN || proto == OP_OTHER)
                    continue;
    			auto ins = create_instrumentation_instruction(proto, *cur, codeline, ++ *loc_id);
                if(ins)
                {
                    bb->replace_instructions(cur, ins->begin(), ins->end());
                    instrumented = true;
                }
			}
		}
        if(instrumented)
        {
    		add_tid_calc(func);
        }

		return true;
	}

	void add_tid_calc(const std::shared_ptr<PtxFunction>& func)
	{
		static const std::string pa_strm = "pa_strm";
		static const std::string pa_rval = "pa_rval";

		auto block = func->main();

		auto sub_block = std::make_shared<PtxBlock>();
		block->elements().push_front(sub_block);
		add_instrumented_param(sub_block, GETTID_STREAM_ARG_TYPE(), pa_strm, "0", InsArgument::Type::Immediate);
		add_instrumented_param(sub_block, TID_REG_TYPE(), pa_rval, "", InsArgument::Type::Reg);

		auto invoke = std::make_shared<PtxCodeLine>();
		invoke->set_instruction("call");
		invoke->set_func_name(NAMEOF_GETTID_FUNCTION_NAME);
		invoke->add_modifier(".uni");
		invoke->add_argument(std::make_shared<InsArgument>(pa_strm, InsArgument::Type::Variable));
		invoke->set_func_retval(std::make_shared<InsArgument>(pa_rval, InsArgument::Type::Variable));
		sub_block->add(std::static_pointer_cast<PtxCodeElement>(invoke));

		auto store = std::make_shared<PtxCodeLine>();
		store->set_instruction("ld");
		store->add_modifier(".param");
		store->add_modifier(TID_REG_TYPE());
		store->add_argument(std::make_shared<InsArgument>(TID_REG_NAME(), InsArgument::Type::Reg));
		auto arg1 = std::make_shared<InsArgument>(pa_rval, InsArgument::Type::Variable);
		arg1->set_indirect();
		store->add_argument(arg1);
		sub_block->add(std::static_pointer_cast<PtxCodeElement>(store));

		auto decl = std::make_shared<PtxVariable>(".reg", TID_REG_TYPE(), TID_REG_NAME());
		block->elements().push_front(std::static_pointer_cast<PtxCodeElement>(std::make_shared<PtxCodeVariable>(decl)));
		    
        auto rsa_decl = std::make_shared<PtxVariable>(".reg", ADDR_TYPE(), ADDRESS_REGISTER_NAME);
  		block->elements().push_front(std::static_pointer_cast<PtxCodeElement>(std::make_shared<PtxCodeVariable>(rsa_decl)));
	}

	std::shared_ptr<PtxBlock> create_instrumentation_instruction(PROTO_OP proto, std::shared_ptr<PtxCodeElement>& line, std::shared_ptr<PtxCodeLine> codeline, int loc_id)
	{
#if 1
        // debug code
        static int max_func = 0;
        if (++max_func >= 76)
            return std::shared_ptr<PtxBlock>();
#endif 

		static const std::string NULL_STR = "0";

		auto outerblock = std::make_shared<PtxBlock>();
        auto innerblock = std::make_shared<PtxBlock>();
		std::string addr_value_name;
        bool predicated = false;
        if(codeline && codeline->predicated())
        {
            predicated = true;
            auto locname = std::string("SP_") + std::to_string(loc_id);
		    auto bra = std::make_shared<PtxCodeLine>();
    		bra->set_instruction("bra");
    		bra->add_argument(std::make_shared<InsArgument>(locname, InsArgument::Type::Immediate));
            bra->set_predicate(codeline->predicate(), !codeline->is_neg_predicated());
            codeline->clear_predicate();
            outerblock->add(std::static_pointer_cast<PtxCodeElement>(bra));
            outerblock->add(std::static_pointer_cast<PtxCodeElement>(innerblock));
            auto label = std::make_shared<PtxCodeLabel>();
            label->set_name(locname);
            outerblock->add(std::static_pointer_cast<PtxCodeLabel>(label));
        }
        else
        {
            if(proto == OP_CONVERGE)
                outerblock->add(line);
            outerblock->add(std::static_pointer_cast<PtxCodeElement>(innerblock));
        }
            
		switch (proto)
		{
        PtxCodeLine::StateSpace ss;
		case OP_LOAD:
		case OP_ATOMIC:
		case OP_STORE:
            ss = codeline->get_state_space();
			if(!should_instrument(ss))
                return std::shared_ptr<PtxBlock>();
			add_instrumentation_target_addr(innerblock, ss, codeline->argument((proto == OP_STORE) ? 0 : 1));
			create_instrumentation_instruction_no_addr(innerblock, ADDRESS_REGISTER_NAME, InsArgument::Type::Reg, proto, loc_id);
            innerblock->add(line);
            break;
		case OP_SYNCTHREADS:
		case OP_FENCE_CTA:	
		case OP_FENCE_GL:	
		case OP_FENCE_SYS:  
			create_instrumentation_instruction_no_addr(innerblock, NULL_STR, InsArgument::Type::Immediate, proto, loc_id);
            innerblock->add(line);
            break;
        case OP_CONVERGE:
			create_instrumentation_instruction_no_addr(innerblock, NULL_STR, InsArgument::Type::Immediate, proto, loc_id);
            break;
		default:
			assert(false);
			return std::shared_ptr<PtxBlock>();
		}
        return outerblock;
	}

    static bool should_instrument(PtxCodeLine::StateSpace ss)
    {
        switch(ss)
        {
        case PtxCodeLine::StateSpace::SS_GLOBAL:
        //case PtxCodeLine::StateSpace::SS_LOCAL:
        case PtxCodeLine::StateSpace::SS_SHARED:
        case PtxCodeLine::StateSpace::SS_UNKNOWN:
            return true;
        default:
            return false;
        }
    }

	void add_instrumentation_target_addr(const std::shared_ptr<PtxBlock>& block, 
                                                PtxCodeLine::StateSpace ss, 
                                                const std::shared_ptr<InsArgument>& data_arg)
	{


		auto load_arg = std::make_shared<PtxCodeLine>();
		std::shared_ptr<InsArgument> src;
		
        switch (data_arg->type())
		{
		case InsArgument::Type::Reg:
            if(ss != PtxCodeLine::StateSpace::SS_UNKNOWN)
            {
    			load_arg->set_instruction("cvta");
    			load_arg->add_modifier(PtxCodeLine::SSMapping::to_string(ss));
    			load_arg->add_modifier(".u64");
    			//load_arg->add_modifier(ADDR_TYPE());
    			load_arg->add_argument(std::make_shared<InsArgument>(ADDRESS_REGISTER_NAME, InsArgument::Type::Reg));
    			src = std::make_shared<InsArgument>(data_arg->name(), data_arg->type());
    			src->set_ofs(data_arg->ofs());
    			load_arg->add_argument(src);
            }
            else
            {
    			load_arg->set_instruction("mov");
    			load_arg->add_modifier(".u64");
    			//load_arg->add_modifier(ADDR_TYPE());
    			load_arg->add_argument(std::make_shared<InsArgument>(ADDRESS_REGISTER_NAME, InsArgument::Type::Reg));
    			src = std::make_shared<InsArgument>(data_arg->name(), data_arg->type());
    			src->set_ofs(data_arg->ofs());
    			load_arg->add_argument(src);
            }
			break;
		case InsArgument::Type::Variable:
            if(ss != PtxCodeLine::StateSpace::SS_PARAM)
            {
    			load_arg->set_instruction("cvta");
    			load_arg->add_modifier(PtxCodeLine::SSMapping::to_string(ss));
    			load_arg->add_modifier(".u64");
    			//load_arg->add_modifier(ADDR_TYPE());
    			load_arg->add_argument(std::make_shared<InsArgument>(ADDRESS_REGISTER_NAME, InsArgument::Type::Reg));
    			src = std::make_shared<InsArgument>(data_arg->name(), data_arg->type());
    			src->set_ofs(data_arg->ofs());
    			load_arg->add_argument(src);
            }
            else
            {
    			load_arg->set_instruction("ld");
    			load_arg->add_modifier(PtxCodeLine::SSMapping::to_string(ss));
    			load_arg->add_modifier(ADDR_TYPE());
    			load_arg->add_argument(std::make_shared<InsArgument>(ADDRESS_REGISTER_NAME, InsArgument::Type::Reg));
    			src = std::make_shared<InsArgument>(data_arg->name(), data_arg->type());
    			src->set_ofs(data_arg->ofs());
    			src->set_indirect();
    			load_arg->add_argument(src);
            }
			break;
		default:
			assert(false);
			fprintf(stderr, "add_instrumentation_target_addr(data_arg->type=%i)\n", data_arg->type());
			exit(1);
		}

		block->add(std::static_pointer_cast<PtxCodeElement>(load_arg));
	}

	void create_instrumentation_instruction_no_addr(
                const std::shared_ptr<PtxBlock>& block, 
                const std::string& addr_name, 
                InsArgument::Type addr_type, int proto, int loc_id)
	{
		static const std::string param_slim_tid = "ps_tid";
		static const std::string param_slim_addr = "ps_addr";
		static const std::string param_slim_op = "ps_op";
		static const std::string param_slim_locid = "ps_lid";

		add_instrumented_param(block, TID_REG_TYPE(), param_slim_tid, TID_REG_NAME(), InsArgument::Type::Reg);
		//add_instrumented_param(block, ".u64", param_slim_addr, "0",InsArgument::Type::Immediate);
		add_instrumented_param(block, ".u64", param_slim_addr, addr_name, addr_type);
		add_instrumented_param(block, ".u32", param_slim_op, std::to_string(proto), InsArgument::Type::Immediate);
		add_instrumented_param(block, ".u32", param_slim_locid, std::to_string(loc_id), InsArgument::Type::Immediate);

		auto invoke = std::make_shared<PtxCodeLine>();
		invoke->set_instruction("call"); 
		invoke->set_func_name(NAMEOF_LOG_OP_FUNCTION_NAME);
		invoke->add_modifier(".uni");
		invoke->add_argument(std::make_shared<InsArgument>(param_slim_tid, InsArgument::Type::Variable));
		invoke->add_argument(std::make_shared<InsArgument>(param_slim_addr, InsArgument::Type::Variable));
		invoke->add_argument(std::make_shared<InsArgument>(param_slim_op, InsArgument::Type::Variable));
		invoke->add_argument(std::make_shared<InsArgument>(param_slim_locid, InsArgument::Type::Variable));
		block->add(std::static_pointer_cast<PtxCodeElement>(invoke));
	}

	void add_instrumented_param(const std::shared_ptr<PtxBlock>& block, 
                                const std::string& type, 
                                const std::string& name, 
                                const std::string& value, 
                                InsArgument::Type value_type)
	{
		auto decl = std::make_shared<PtxVariable>(".param", type, name);
		block->add(std::static_pointer_cast<PtxCodeElement>(std::make_shared<PtxCodeVariable>(decl)));

		if (value.size() > 0)
		{
			auto store = std::make_shared<PtxCodeLine>();
			store->set_instruction("st");
			store->add_modifier(".param");
			store->add_modifier(type);

			auto arg0 = std::make_shared<InsArgument>(name, InsArgument::Type::Variable);
			arg0->set_indirect();
			store->add_argument(arg0);

			auto arg1 = std::make_shared<InsArgument>(value, value_type);
			store->add_argument(arg1);

			block->add(std::static_pointer_cast<PtxCodeElement>(store));
		}
	}

	PROTO_OP get_proto(const std::shared_ptr<PtxCodeLine>& ptr)
	{
		struct InstructionProto
		{
			std::string str;
			PROTO_OP op;
		};
		static const std::initializer_list<InstructionProto> INSTRUCTIONS = {
			{ "ld", OP_LOAD },
			{ "st", OP_STORE },
			{ "atom", OP_ATOMIC },
			{ "bar", OP_SYNCTHREADS },
		};
		static const std::initializer_list<InstructionProto> FENCES = {
			{ ".cta", OP_FENCE_CTA },
			{ ".gl", OP_FENCE_GL },
			{ ".sys", OP_FENCE_SYS },
		};

		if (ptr->instruction() == "membar")
		{
			if (ptr->modifiers().begin() == ptr->modifiers().end())
			{
				std::cerr << "Invalid membar instruction!: " << *ptr << std::endl;
				exit(1);
			}
			std::string modifier = *ptr->modifiers().begin();
			auto it = std::find_if(FENCES.begin(), FENCES.end(), [&](auto& s) {return s.str == modifier; });
			if (it == FENCES.end())
			{
				std::cerr << "Invalid membar instruction!: " << *ptr << std::endl;
				exit(1);
			}
			return it->op;
		}
			
		auto it = std::find_if(INSTRUCTIONS.begin(), INSTRUCTIONS.end(), [&](auto& s) {return s.str == ptr->instruction(); });
		if (it == INSTRUCTIONS.end())
			return OP_OTHER;
		return it->op;
	}


	bool add_stub_code(PtxProgram& ptx)
	{
		auto it = std::find_if(ptx.begin(), ptx.end(), [&](auto& d){ return d->dir_type() != PtxDirective::DirType::Directive; });
		if (it == ptx.end())
		{
			std::cerr << "Not code in original PTX file!" << std::endl;
			return false;
		}

		ptx.directives().insert(it, _slim_devarea);
		ptx.directives().insert(it, _slim_init);
		ptx.directives().insert(it, _slim_gettid);
		ptx.directives().insert(it, _slim_log);
		return true;
	}

	bool verify_target(PtxProgram& ptx)
	{
		// XXX: check address_size==64, target, version
		return true;
	}


	void load_stub(const std::string& stub_ptx)
	{
		PtxParser parser;
		PtxParser::ParseStatus status;
		if (!parser.parse_ptx(stub_ptx, &status))
		{
			std::cerr << "Could not parse stub PTX at byte: " << status.parsed << std::endl;
			exit(1);
		}

		PtxProgram& program = parser.ptx();
		_slim_init = program.find(NAMEOF_INIT_FUNCTION_NAME);
		_slim_gettid = program.find(NAMEOF_GETTID_FUNCTION_NAME);
		_slim_log = program.find(NAMEOF_LOG_OP_FUNCTION_NAME);
		_slim_devarea = program.find(NAMEOF_DEVICE_AREA_GLOBAL_NAME);
	}

	static const std::string& ADDR_TYPE() { static const std::string value{ ".b64" }; return value; };
	static const std::string& GETTID_STREAM_ARG_TYPE() { static const std::string value{ ".b32" }; return value; };
	static const std::string& TID_REG_TYPE() { static const std::string value{ ".b64" }; return value; };
	static const std::string& TID_REG_NAME() { static const std::string value{ "%rd_slim_tid" }; return value;  };

private:
    const char* ADDRESS_REGISTER_NAME = "%rsa";
	std::shared_ptr<PtxDirective> _slim_init, _slim_gettid, _slim_log, _slim_devarea;
};
