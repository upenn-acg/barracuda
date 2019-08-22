#pragma once

#include "ptx_program.hpp"

class PtxMaker
{
public:
    static std::shared_ptr<PtxCodeElement> mov_to_reg(const std::string& arg_type, const std::string& name, 
                                                      InsArgument::Type type, const std::string& value, int offset = -1)
    {
        auto mov = std::make_shared<PtxCodeLine>();
        mov->set_instruction("mov");
        mov->add_modifier(arg_type);
        mov->add_argument(std::make_shared<InsArgument>(name, InsArgument::Type::Reg));
        auto arg = std::make_shared<InsArgument>(value, type);
        if(offset >= 0)
            arg->set_ofs(offset);
        mov->add_argument(arg);
        return std::static_pointer_cast<PtxCodeElement>(mov);
    }
    static std::shared_ptr<PtxCodeElement> mov_to_reg(const std::string& arg_type, const std::string& name, 
                                                      const std::shared_ptr<InsArgument>& data_arg)
    {
        return mov_to_reg(arg_type, name, data_arg->type(), data_arg->name(), data_arg->ofs());
    }

    static std::shared_ptr<PtxCodeElement> st_param(const std::string& arg_type, const std::string& name, 
                                                      InsArgument::Type type, const std::string& value)
    {
        auto store = std::make_shared<PtxCodeLine>();
        store->set_instruction("st");
        store->add_modifier(".param");
        store->add_modifier(arg_type);
        auto arg = std::make_shared<InsArgument>(name, InsArgument::Type::Variable);
        arg->set_indirect();
        store->add_argument(arg);
        store->add_argument(std::make_shared<InsArgument>(value, type));
        return std::static_pointer_cast<PtxCodeElement>(store);
    }

    static std::shared_ptr<PtxCodeElement> ld_param(const std::string& arg_type, const std::string& name, 
                                                      InsArgument::Type type, const std::string& value)
    {
        auto load = std::make_shared<PtxCodeLine>();
        load->set_instruction("ld");
        load->add_modifier(".param");
        load->add_modifier(arg_type);
        auto arg = std::make_shared<InsArgument>(name, InsArgument::Type::Variable);
        load->add_argument(arg);
        arg = std::make_shared<InsArgument>(value, type);
        arg->set_indirect();
        load->add_argument(arg);
        return std::static_pointer_cast<PtxCodeElement>(load);
    }
    
    static std::shared_ptr<PtxCodeElement> ld(PtxCodeLine::StateSpace ss, const std::string& arg_type, const std::string& name, 
                                              const std::shared_ptr<InsArgument>& data_arg)
    {
        auto load = std::make_shared<PtxCodeLine>();
        load->set_instruction("ld");
        load->add_modifier(PtxCodeLine::SSMapping::to_string(ss));
        load->add_modifier(arg_type);
        load->add_argument(std::make_shared<InsArgument>(name, InsArgument::Type::Reg));
        auto src = std::make_shared<InsArgument>(data_arg->name(), data_arg->type());
        src->set_ofs(data_arg->ofs());
        src->set_indirect();
        load->add_argument(src);
        return load;
    }

    static std::shared_ptr<PtxCodeElement> cvta(PtxCodeLine::StateSpace ss, const std::string& arg_type,
                                                const std::string& arg, InsArgument::Type type, const std::string& value, int offset = -1)
    {
        auto cvta = std::make_shared<PtxCodeLine>();
        cvta->set_instruction("cvta");
        cvta->add_modifier(PtxCodeLine::SSMapping::to_string(ss));
        cvta->add_modifier(arg_type);
        cvta->add_argument(std::make_shared<InsArgument>(arg, InsArgument::Type::Reg));
        auto src = std::make_shared<InsArgument>(value, type);
        if(offset >= 0)
            src->set_ofs(offset);
        cvta->add_argument(src);
        return std::static_pointer_cast<PtxCodeElement>(cvta);
    }
    
    static std::shared_ptr<PtxCodeElement> cvta(PtxCodeLine::StateSpace ss, const std::string& arg_type, 
                                                const std::string& arg, const std::shared_ptr<InsArgument>& data_arg)
    {
        return cvta(ss, arg_type, arg, data_arg->type(), data_arg->name(), data_arg->ofs());
    }

    static std::shared_ptr<PtxCodeLine> call(const std::string& func_name)
    {
        auto invoke = std::make_shared<PtxCodeLine>();
        invoke->set_instruction("call");
        invoke->set_func_name(func_name);
        invoke->add_modifier(".uni");
        return invoke;
    }
};


