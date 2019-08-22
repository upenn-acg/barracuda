#pragma once

#include "ptx_program.hpp"
#include "ptx_parser.hpp"
#include "ptx_basic_block.hpp"
#include "protocol.hpp"
#include "ptx_stub.h"
#include "env.hpp"
#include "ptx_maker.hpp"

#include <iostream>
#include <streambuf>
#include <sstream>
#include <memory>
#include <set>
#include <map>
#include <list>

// XXXX!
int NUM_BEFORE_PTX_INSTRUCTIONS = 0;
int NUM_AFTER_PTX_INSTRUCTIONS = 0;
int NUM_INSTRUMENTED_INSTRUCTIONS = 0;
int NUM_ACTUAL_INSTRUMENTED_INSTRUCTIONS = 0;

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
        std::set<std::string> extern_funcs;

        for (auto& d : ptx)
        {
            if (d->dir_type() == PtxDirective::DirType::Function)
            { 
                auto func = std::static_pointer_cast<PtxFunction>(d);
                if(func->decorator() == ".extern")
                {
                    extern_funcs.insert(func->name());
                    continue;
                }
                if(!func->is_kernel())
                    func->prepend_arg(std::make_shared<PtxVariable>(".param", TID_ARG_TYPE, TID_ARG_NAME, false));
                //else
                //    func->prepend_arg(std::make_shared<PtxVariable>(".param", SID_ARG_TYPE, SID_ARG_NAME, false));

                if(!func->has_main_block())
                    continue;
                if (!instrument_func(func, &loc_id, extern_funcs))
                    return false;
            }
        }
        return true;
    }

    struct instrumentation_point
    {
        instrumentation_point(const std::shared_ptr<PtxBasicBlock>& block_, 
                              const PtxBlock::list_iterator& where_, 
                              const std::shared_ptr<PtxBlock>& code_, 
                              PROTO_OP op_) : block(block_), where(where_), code(code_), op(op_) { }

        std::shared_ptr<PtxBasicBlock> block;
        PtxBlock::list_iterator where;
        std::shared_ptr<PtxBlock> code;
        PROTO_OP op;
    };
    typedef std::list<instrumentation_point> instrumentation_list;
    typedef std::map<std::string, instrumentation_list::iterator> instrumentation_map;

    bool instrument_func(const std::shared_ptr<PtxFunction>& func, int* loc_id, const std::set<std::string>& extern_funcs)
    {
        PtxFunctionAnalyzer analyzer(func);
        instrumentation_list tasks;

        for (auto& bb : analyzer)
        {
            instrumentation_map modifiers;

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
                    proto = get_proto(codeline, prev, next, func->is_kernel());
                    break;
                case PtxCodeElement::Type::TypeLabel:
                    proto = OP_CONVERGE;
                    break;
                default:
                    break;
                }
                prev = codeline;
                //std::cout << "Line: ";
                //if(codeline) codeline->print(std::cout);
                //std::cout << " " << proto << " " << func->is_kernel() << "\n";
                //if(proto == OP_END_KERNEL)
                //   printf("End kernel!\n");
                instrumentation_list::iterator insit = tasks.end();
                switch(proto)
                {
                case OP_CALL:
                    if(extern_funcs.find(codeline->func_name()) == extern_funcs.end())
                       prepend_tid_arg(bb->block(), cur, codeline);
                    break;
                
                case OP_OTHER:
                    break;

                default:
                    {
                        auto ins = create_instrumentation_instruction(proto, *cur, codeline, ++ *loc_id);
                        if(ins)
                        {
                            tasks.push_back(instrumentation_point(bb, cur, ins, proto));
                            insit = std::prev(tasks.end());
                            NUM_INSTRUMENTED_INSTRUCTIONS += 1;
                        }
                    }
                }
                inner_block_optimization(modifiers, tasks, proto, codeline, insit);
            }
        }

        if(func->is_kernel())
            instrument_kernel_entry_point(func->main(), ++ *loc_id);
      
        //printf("Tasks left: %i\n", tasks.size()); 
        if(func->is_kernel() || tasks.size() > 0)
        {
            for(auto it : tasks)
            {
               // std::cout << "Adding task:\n";
              //  it.code->print(std::cout);
               it.block->replace_instructions(it.where, it.code->begin(), it.code->end());
            }
            NUM_ACTUAL_INSTRUMENTED_INSTRUCTIONS += tasks.size();
            add_tid_calc(func);
        }

        return true;
    }

    void inner_block_optimization(instrumentation_map& insmap, instrumentation_list& tasks, 
                                    PROTO_OP proto, const std::shared_ptr<PtxCodeLine>& codeline, instrumentation_list::iterator insit)
    {
        if(!codeline)
            return;

        if(proto == OP_LOAD || proto == OP_STORE)
        {
            auto name = get_indirect_argument_name(codeline);
            if(name[0] == '%')
            {
                instrumentation_map::iterator prev_entry = insmap.find(name);
                if(prev_entry != insmap.end())
                {
                    auto old_op = prev_entry->second->op;
                    if(old_op == OP_STORE && proto != OP_STORE)
                    {
                        tasks.erase(insit);
                        //printf("Existing load or store from: %s, erasing new\n", name.c_str());
                    }
                    else
                    {
                        tasks.erase(prev_entry->second);
                        //printf("Existing load or store from: %s, erasing old\n", name.c_str());
                    }
                    prev_entry->second = insit;
                }
                else
                {
                    insmap.insert(instrumentation_map::value_type(name, insit));
                    //printf("New load or store from: %s\n", name.c_str());
                }
            }
        }
        else if(proto == OP_SYNCTHREADS || (proto >= OP_FIRST_SYNC && proto <= OP_LAST_SYNC))
        {
            //printf("sync, clearing all.\n");
            insmap.clear();
            return;
        }
            
        auto dst_register = codeline->dst_register();
        //std::cout << "Examining: ";
        //codeline->print(std::cout);
        //std::cout << "\n";
        
        if(dst_register.size() > 0)
        {
            dst_register = get_unique_register_name(dst_register, "");
            for(auto it = insmap.begin(); it != insmap.end();)
            {
                auto cur = it ++;
                if(strncmp(cur->first.c_str(), dst_register.c_str(), dst_register.size()) == 0)
                {
                    printf("sync, clearing %s.\n", cur->first.c_str());
                    insmap.erase(cur);
                }
            }
        }
                
    
    }

    PtxBlock::list_iterator func_start(const std::shared_ptr<PtxBlock>& block)
    {
        for(auto it = block->begin();
            it != block->end();
            ++ it)
        {
            switch((*it)->type())
            {
            case PtxCodeElement::Type::TypeDirective:
            case PtxCodeElement::Type::TypeVariable:
                break;
            default:
                return it;
            }
        }
        return block->begin();
        
    }

    void prepend_tid_arg(std::shared_ptr<PtxBlock> block, PtxBlock::list_iterator it, std::shared_ptr<PtxCodeLine>& line)
    {
        declare_instrumented_param(block, TID_REG_TYPE, param_slim_tid);
        block->elements().insert(it, PtxMaker::st_param(TID_REG_TYPE, param_slim_tid, InsArgument::Type::Reg, TID_REG_NAME));
        line->prepend_argument(std::make_shared<InsArgument>(param_slim_tid, InsArgument::Type::Variable));
    }

    void add_tid_calc(const std::shared_ptr<PtxFunction>& func)
    {
        static const std::string pa_rval = "pa_rval";
        static const std::string pa_sid = "pa_sid";
        static const std::string ra_sid = "%ra_sid";
        auto block = func->main();

        auto sub_block = std::make_shared<PtxBlock>();
        if(func->is_kernel())
        {
            auto decl = std::make_shared<PtxVariable>(".reg", SID_ARG_TYPE, ra_sid);
            sub_block->add(std::static_pointer_cast<PtxCodeElement>(std::make_shared<PtxCodeVariable>(decl)));

            declare_instrumented_param(sub_block, TID_REG_TYPE, pa_rval);
            declare_instrumented_param(sub_block, SID_ARG_TYPE, pa_sid);
            add_instrumented_param(sub_block, TID_REG_TYPE, pa_rval, "", InsArgument::Type::Reg);
    
            sub_block->add(PtxMaker::mov_to_reg(SID_ARG_TYPE, ra_sid, InsArgument::Type::Immediate, "0"));
            sub_block->add(PtxMaker::st_param(SID_ARG_TYPE, pa_sid, InsArgument::Type::Reg, ra_sid));

            auto invoke = PtxMaker::call(NAMEOF_GETTID_FUNCTION_NAME);
            invoke->add_argument(std::make_shared<InsArgument>(pa_sid, InsArgument::Type::Variable));
            invoke->set_func_retval(std::make_shared<InsArgument>(pa_rval, InsArgument::Type::Variable));
            sub_block->add(std::static_pointer_cast<PtxCodeElement>(invoke));

            sub_block->add(PtxMaker::ld_param(TID_REG_TYPE, TID_REG_NAME, InsArgument::Type::Variable, pa_rval));
        }
        else
        {
            sub_block->add(PtxMaker::ld_param(TID_REG_TYPE, TID_REG_NAME, InsArgument::Type::Variable, TID_ARG_NAME));
        }
        PtxBlock::list_type::iterator it = func_start(block);
        block->elements().insert(it, sub_block);

        block->elements().push_front(std::static_pointer_cast<PtxCodeElement>(std::make_shared<PtxCodeVariable>(std::make_shared<PtxVariable>(".reg", TID_REG_TYPE, TID_REG_NAME))));
        block->elements().push_front(std::static_pointer_cast<PtxCodeElement>(std::make_shared<PtxCodeVariable>(std::make_shared<PtxVariable>(".reg", ADDR_TYPE, ADDRESS_REGISTER_NAME))));
        block->elements().push_front(std::static_pointer_cast<PtxCodeElement>(std::make_shared<PtxCodeVariable>(std::make_shared<PtxVariable>(".reg", OP_ID_TYPE, OP_ID_REGISTER_NAME))));
        block->elements().push_front(std::static_pointer_cast<PtxCodeElement>(std::make_shared<PtxCodeVariable>(std::make_shared<PtxVariable>(".reg", LOC_ID_TYPE, LOC_ID_REGISTER_NAME))));
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
        else if(proto != OP_END_KERNEL)
        {
            outerblock->add(line);
            outerblock->add(std::static_pointer_cast<PtxCodeElement>(innerblock));
        } else {
            outerblock->add(std::static_pointer_cast<PtxCodeElement>(innerblock));
            outerblock->add(line);
        }
           
        switch (proto)
        {
        PtxCodeLine::StateSpace ss;
        case OP_CONVERGE:
        case OP_SYNCTHREADS:
            add_instrumentation_target_addr(innerblock, PtxCodeLine::StateSpace::SS_UNKNOWN, std::shared_ptr<InsArgument>());
            create_instrumentation_instruction_no_addr(innerblock, ADDRESS_REGISTER_NAME, InsArgument::Type::Immediate, proto, loc_id);
            break;
        
        case OP_END_KERNEL:
            add_instrumentation_target_addr(innerblock, PtxCodeLine::StateSpace::SS_UNKNOWN, std::shared_ptr<InsArgument>());
            create_instrumentation_instruction_no_addr(innerblock, ADDRESS_REGISTER_NAME, InsArgument::Type::Immediate, proto, loc_id);
            break;

        case OP_OTHER:
            assert(false);
            return std::shared_ptr<PtxBlock>();

        default:
            ss = codeline->get_state_space();
            if(!should_instrument(ss))
                return std::shared_ptr<PtxBlock>();
            auto arg = get_indirect_argument(codeline); 
            add_instrumentation_target_addr(innerblock, ss, arg);
            create_instrumentation_instruction_no_addr(innerblock, ADDRESS_REGISTER_NAME, InsArgument::Type::Reg, proto, loc_id);
            break;
        }
        if(add_inner)
            innerblock->add(line);
        return outerblock;
    }

    void instrument_kernel_entry_point(std::shared_ptr<PtxBlock>& main_block, int loc_id)
    {
        auto block = std::make_shared<PtxBlock>();
        add_instrumentation_target_addr(block, PtxCodeLine::StateSpace::SS_UNKNOWN, std::shared_ptr<InsArgument>());
        create_instrumentation_instruction_no_addr(block, ADDRESS_REGISTER_NAME, InsArgument::Type::Immediate, OP_START_KERNEL, loc_id);
        ++ NUM_INSTRUMENTED_INSTRUCTIONS;
        auto it = func_start(main_block);
        main_block->elements().insert(it, block);
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


        std::shared_ptr<PtxCodeElement> load_arg;
    
        if(data_arg)
        {
            switch (data_arg->type())
            {
            case InsArgument::Type::Reg:
                if(ss != PtxCodeLine::StateSpace::SS_UNKNOWN)
                    load_arg = PtxMaker::cvta(ss, ".u64", ADDRESS_REGISTER_NAME, data_arg);
                else
                    load_arg = PtxMaker::mov_to_reg(".u64", ADDRESS_REGISTER_NAME, data_arg);
                break;
            case InsArgument::Type::Variable:
                if(ss != PtxCodeLine::StateSpace::SS_PARAM)
                    load_arg = PtxMaker::cvta(ss, ".u64", ADDRESS_REGISTER_NAME, data_arg);
                else
                    load_arg = PtxMaker::ld(ss, ADDR_TYPE, ADDRESS_REGISTER_NAME, data_arg);
                break;
            default:
                assert(false);
                fprintf(stderr, "add_instrumentation_target_addr(data_arg->type=%i)\n", data_arg->type());
                exit(1);
            }
        }
        else
        {
            load_arg = PtxMaker::mov_to_reg(".u64", ADDRESS_REGISTER_NAME, InsArgument::Type::Immediate, "0");
        }

        block->add(std::static_pointer_cast<PtxCodeElement>(load_arg));
    }

    void create_instrumentation_instruction_no_addr(
                const std::shared_ptr<PtxBlock>& block, 
                const std::string& addr_name, 
                InsArgument::Type addr_type, int proto, int loc_id)
    {
        // BUG in PTX JIT register allocator breaks when we use immediates
        // for slim_log() arguments rather then allocating registers ourselves -
        // it allocates a new pair for each invocation!
        
        auto decl = std::make_shared<PtxVariable>(".reg", ".b32", "temp_reg_param"); // JIT wants this - undocumented
        block->add(std::static_pointer_cast<PtxCodeElement>(std::make_shared<PtxCodeVariable>(decl)));
        block->add(PtxMaker::mov_to_reg(OP_ID_TYPE, OP_ID_REGISTER_NAME, InsArgument::Type::Immediate, std::to_string(proto)));
        block->add(PtxMaker::mov_to_reg(LOC_ID_TYPE, LOC_ID_REGISTER_NAME, InsArgument::Type::Immediate, std::to_string(loc_id)));

        declare_instrumented_param(block, TID_REG_TYPE, param_slim_tid);
        add_instrumented_param(block, TID_REG_TYPE, param_slim_tid, TID_REG_NAME, InsArgument::Type::Reg);
        declare_instrumented_param(block, ".u64", param_slim_addr);
        add_instrumented_param(block, ".u64", param_slim_addr, addr_name, addr_type);
        declare_instrumented_param(block, ".u32", param_slim_op);
        add_instrumented_param(block, ".u32", param_slim_op, OP_ID_REGISTER_NAME, InsArgument::Type::Reg);
        declare_instrumented_param(block, ".u32", param_slim_locid);
        add_instrumented_param(block, ".u32", param_slim_locid, LOC_ID_REGISTER_NAME, InsArgument::Type::Reg);

        auto invoke = PtxMaker::call(NAMEOF_LOG_OP_FUNCTION_NAME);
        invoke->add_argument(std::make_shared<InsArgument>(param_slim_tid, InsArgument::Type::Variable));
        invoke->add_argument(std::make_shared<InsArgument>(param_slim_addr, InsArgument::Type::Variable));
        invoke->add_argument(std::make_shared<InsArgument>(param_slim_op, InsArgument::Type::Variable));
        invoke->add_argument(std::make_shared<InsArgument>(param_slim_locid, InsArgument::Type::Variable));
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
        if (value.size() > 0)
        {
            block->add(PtxMaker::st_param(type, name, value_type, value));
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

    std::string get_indirect_argument_name(const std::shared_ptr<PtxCodeLine>& ptr)
    {
        auto arg = get_indirect_argument(ptr); 
        return get_unique_register_name(arg->name(), std::to_string(arg->ofs()));
    }
    std::string get_unique_register_name(const std::string& name, const std::string& offset)
    {
        return name + "+" + offset;
    }

    std::shared_ptr<InsArgument> get_indirect_argument(const std::shared_ptr<PtxCodeLine>& ptr)
    {
        int idx = -1, cidx = 0;
        for(auto arg = ptr->arguments().begin(); arg != ptr->arguments().end(); ++ arg, ++ cidx)
        {
            if((*arg)->indirect())
            {
                if(idx != -1) 
                {
                    std::cerr << "Error, instruction contains multiple indirect arguments:\n";
                    ptr->print(std::cerr);
                    std::cerr << std::endl;
                    exit(1);
                }
                idx = cidx;
            }
        }
        if(idx == -1) 
        {
            std::cerr << "Error, instruction contains no indirect arguments:\n";
            ptr->print(std::cerr);
            std::cerr << std::endl;
            exit(1);
        }

        return ptr->argument(idx);
    }

    PROTO_OP get_proto(const std::shared_ptr<PtxCodeLine>& ptr, const std::shared_ptr<PtxCodeLine>& prev, const std::shared_ptr<PtxCodeLine>& next, bool is_kernel)
    {
        static const std::initializer_list<InstructionProto> INSTRUCTIONS = {
            { "ld", OP_LOAD },
            { "st", OP_STORE },
            { "atom", OP_ATOMIC },
            { "bar", OP_SYNCTHREADS },
            { "call", OP_CALL },
            { "ret", OP_END_KERNEL },
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
        case OP_END_KERNEL:
            if(!is_kernel)
                return OP_OTHER;
            break;

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

    const std::string ADDR_TYPE =  ".u64";
    const std::string GETTID_STREAM_ARG_TYPE = ".u32" ;
    const std::string SID_ARG_TYPE = ".u32";
    const std::string TID_ARG_TYPE = ".u64";
    const std::string TID_REG_TYPE = ".u64";
    const std::string OP_ID_TYPE = ".u32" ;
    const std::string LOC_ID_TYPE = ".u32" ;
    const char* ADDRESS_REGISTER_NAME = "%rsa";
    const char* OP_ID_REGISTER_NAME = "%rso";
    const char* LOC_ID_REGISTER_NAME = "%rsl";
    const std::string TID_REG_NAME = "%rdt";
    const std::string TID_ARG_NAME = "psa_tid";
    const std::string SID_ARG_NAME = "psa_sid";

    const std::string param_slim_tid = "ps_tid";
    const std::string param_slim_addr = "ps_addr";
    const std::string param_slim_op = "ps_op";
    const std::string param_slim_locid = "ps_lid";
private:
    std::shared_ptr<PtxDirective> _slim_init, _slim_gettid, _slim_log, _slim_devarea;
};
