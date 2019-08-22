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

// XXXX!
int NUM_BEFORE_PTX_INSTRUCTIONS = 0;
int NUM_AFTER_PTX_INSTRUCTIONS = 0;
int NUM_INSTRUMENTED_INSTRUCTIONS = 0;

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

        if(Env::manually_instrumented())
        {
            *out = ptx_text;
            return true;
        }
        PtxParser parser;
        PtxParser::ParseStatus status;
        if (!parser.parse_ptx(ptx_text, &status))
        {
            std::cerr << "Could not parse PTX at byte: " << status.parsed << std::endl;
            return false;
        }

        PtxProgram& ptx = parser.ptx();
        NUM_BEFORE_PTX_INSTRUCTIONS = ptx.num_instructions();

        if (! (verify_target(ptx)
               && instrument_funcs(ptx)
               && add_stub_code(ptx)))
            return false;
        NUM_AFTER_PTX_INSTRUCTIONS = ptx.num_instructions();


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
                auto func = std::static_pointer_cast<PtxFunction>(d);
                if(!func->has_main_block())
                    continue;
                if (!instrument_func(func, &loc_id))
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
            std::shared_ptr<PtxCodeLine> prev;
            for (auto it = bb->begin(); it != bb->end(); )
            {
                auto cur = it ++;
                auto proto = OP_OTHER;
                std::shared_ptr<PtxCodeLine> codeline, next;
                switch((*cur)->type())
                {
                case PtxCodeElement::Type::TypeCode:
                    codeline = std::static_pointer_cast<PtxCodeLine>(*cur);
                    if(it != bb->end() && (*it) && (*it)->type() == PtxCodeElement::Type::TypeCode)
                        next = std::static_pointer_cast<PtxCodeLine>(*it);
                    proto = get_proto(codeline, prev, next);
                    break;
                case PtxCodeElement::Type::TypeLabel:
                    proto = OP_CONVERGE;
                    break;
                default:
                    break;
                }
                prev = codeline;
                if(proto == OP_CONVERGE)
                    continue;
                auto ins = create_instrumentation_instruction(proto, *cur, codeline, ++ *loc_id);
                if(ins)
                {
                    bb->replace_instructions(cur, ins->begin(), ins->end());
                    instrumented = true;
                    ++ NUM_INSTRUMENTED_INSTRUCTIONS;
                }
            }
        }
        func->main()->elements().push_front(std::static_pointer_cast<PtxCodeElement>(std::make_shared<PtxCodeVariable>(std::make_shared<PtxVariable>(".reg", OP_ID_TYPE, OP_ID_REGISTER_NAME))));

        return true;
    }

    std::shared_ptr<PtxBlock> create_instrumentation_instruction(PROTO_OP proto, std::shared_ptr<PtxCodeElement>& line, std::shared_ptr<PtxCodeLine> codeline, int loc_id)
    {
        static const std::string NULL_STR = "0";

        auto outerblock = std::make_shared<PtxBlock>();
        auto innerblock = std::make_shared<PtxBlock>();
        std::string addr_value_name;
        bool add_inner = false;
        if(codeline && codeline->predicated())
        {
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
            add_inner = true;
        }
        else
        {
            outerblock->add(line);
            outerblock->add(std::static_pointer_cast<PtxCodeElement>(innerblock));
        }
           
        create_instrumentation_instruction_no_addr(innerblock, NULL_STR, InsArgument::Type::Immediate, proto != OP_OTHER, loc_id);
        if(add_inner)
            innerblock->add(line);
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

    void create_instrumentation_instruction_no_addr(
                const std::shared_ptr<PtxBlock>& block, 
                const std::string& addr_name, 
                InsArgument::Type addr_type, int proto, int loc_id)
    {
        // BUG in PTX JIT register allocator breaks when we use immediates
        // for slim_log() arguments rather then allocating registers ourselves -
        // it allocates a new pair for each invocation!

        auto mov = std::make_shared<PtxCodeLine>();
        mov->set_instruction("mov"); 
        mov->add_modifier(OP_ID_TYPE);
        mov->add_argument(std::make_shared<InsArgument>(OP_ID_REGISTER_NAME, InsArgument::Type::Reg));
        mov->add_argument(std::make_shared<InsArgument>(std::to_string(proto), InsArgument::Type::Immediate));
        block->add(std::static_pointer_cast<PtxCodeElement>(mov));

        add_instrumented_param(block, ".u32", param_slim_op, OP_ID_REGISTER_NAME, InsArgument::Type::Reg);

        auto invoke = std::make_shared<PtxCodeLine>();
        invoke->set_instruction("call"); 
        invoke->set_func_name(NAMEOF_LOG_OP_FUNCTION_NAME);
        invoke->add_modifier(".uni");
        invoke->add_argument(std::make_shared<InsArgument>(param_slim_op, InsArgument::Type::Variable));
        block->add(std::static_pointer_cast<PtxCodeElement>(invoke));
    }

    void declare_instrumented_param(const std::shared_ptr<PtxBlock>& block, const std::string& type, const std::string& name)
    {
        auto decl = std::make_shared<PtxVariable>(".param", type, name);
        block->elements().push_front(std::static_pointer_cast<PtxCodeElement>(std::make_shared<PtxCodeVariable>(decl)));
    }

    void add_instrumented_param(const std::shared_ptr<PtxBlock>& block, 
                                const std::string& type, 
                                const std::string& name, 
                                const std::string& value, 
                                InsArgument::Type value_type)
    {
        declare_instrumented_param(block, type, name);
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

    struct InstructionProto
    {
        std::string str;
        int op;
    };

    int get_fence_type(const std::shared_ptr<PtxCodeLine>& ptr)
    {
        static const std::initializer_list<InstructionProto> FENCES = {
            { ".cta", F_CTA },
            { ".gl", F_GL },
            { ".sys", F_SYS },
        };

        if (ptr->instruction() == "membar")
        {
            std::string modifier = ptr->modifier(0);
            auto it = std::find_if(FENCES.begin(), FENCES.end(), [&](auto& s) {return s.str == modifier; });
            if (it == FENCES.end())
            {
                std::cerr << "Invalid membar instruction!: " << *ptr << std::endl;
                exit(1);
            }
            return it->op;
        }
        return F_UNKNOWN;
    }

    PROTO_OP get_proto(const std::shared_ptr<PtxCodeLine>& ptr, const std::shared_ptr<PtxCodeLine>& prev, const std::shared_ptr<PtxCodeLine>& next)
    {
        static const std::initializer_list<InstructionProto> INSTRUCTIONS = {
            { "ld", OP_LOAD },
            { "st", OP_STORE },
            { "atom", OP_ATOMIC },
            { "bar", OP_SYNCTHREADS },
        };
        auto it = std::find_if(INSTRUCTIONS.begin(), INSTRUCTIONS.end(), [&](auto& s) {return s.str == ptr->instruction(); });
        if (it == INSTRUCTIONS.end())
            return OP_OTHER;

        int fnext = ((bool)next) ? get_fence_type(next) : F_UNKNOWN;
        int fprev = ((bool)prev) ? get_fence_type(prev) : F_UNKNOWN;
        int value = it->op;
        std::string atom_modifier;
        switch(it->op)
        {
        case OP_LOAD:
            if(fnext != F_UNKNOWN)
                value = OP_ACQUIRE | fnext;
            break;

        case OP_STORE:
            if(fprev != F_UNKNOWN)
                value = OP_RELEASE | fprev;
            break;

        case OP_ATOMIC:
            atom_modifier = ptr->modifier_from_list({".exch", ".cas"});
            if(atom_modifier == ".exch")
            {
                if(fprev != F_UNKNOWN)
                    value = OP_RELEASE | F_ATOMIC | fprev;
            } 
            else if(atom_modifier == ".cas")
            {
                if(fnext != F_UNKNOWN)
                    value = OP_ACQUIRE | F_ATOMIC | fnext;
            }
            else if(fnext != F_UNKNOWN || fprev != F_UNKNOWN)
                value = OP_ACQREL | F_ATOMIC | std::max(fnext, fprev);
        }

        return (PROTO_OP)value;
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
        _slim_log = program.find(NAMEOF_LOG_OP_FUNCTION_NAME);
        _slim_devarea = program.find(NAMEOF_DEVICE_AREA_GLOBAL_NAME);
    }

    const std::string OP_ID_TYPE = ".b32" ;
    const char* OP_ID_REGISTER_NAME = "%rso";

    const std::string param_slim_op = "ps_op";
private:
    std::shared_ptr<PtxDirective> _slim_init, _slim_gettid, _slim_log, _slim_devarea;
};
