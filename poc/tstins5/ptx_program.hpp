#pragma once

#include <string>
#include <list>
#include <memory>
#include <cassert>
#include <algorithm>

class PtxDirective
{
public:
	enum class DirType { Directive, Variable, Function };

	PtxDirective(const std::string& directive, const std::string& line) :
		  _dir_type(DirType::Directive)
		, _directive(directive)
		, _line(line) 
	{

	}

	PtxDirective(DirType dir_type) :
		_dir_type(dir_type)
	{

	}

	DirType dir_type() const { return _dir_type; }
	std::string get_directive() const  { return _directive; }
	std::string get_line() const { return _directive; }

	virtual void print(std::ostream& out) const
	{
		out << _directive << " " << _line;
	}

	virtual std::string get_id() const
	{
		return _directive;
	}

protected:
	DirType _dir_type;
	std::string _directive, _line;
};
static inline std::ostream& operator<<(std::ostream& out, const PtxDirective& dir) { dir.print(out); return out; }

class PtxSymbol : public PtxDirective
{
public:
	PtxSymbol(DirType dir_type) : PtxDirective(dir_type)
	{

	}
	PtxSymbol(DirType dir_type, const std::string& type, const std::string& name) : PtxDirective(dir_type), _type(type), _name(name)
	{

	}

	std::string decorator() const { return _decorator; }
	std::string type() const { return _type; }
	std::string name() const { return _name; }

	void set_decorator(const std::string& value) { _decorator = value; }
	void set_type(const std::string& value) { _type = value; }
	void set_name(const std::string& value) { _name = value; }

	virtual std::string get_id() const override
	{
		return _name;
	}

protected:
	std::string _decorator, _type, _name;
};

class PtxVariable : public PtxSymbol
{
public:
	PtxVariable() : PtxSymbol(DirType::Variable), _is_standalone(true), _array(-1)
	{

	}

	PtxVariable(const std::string& type, const std::string& data_type, const std::string& name) : 
		PtxSymbol(DirType::Variable, type, name)
		, _is_standalone(true)
		, _datatype(data_type)
        , _array(-1)
	{

	}

	std::string attributes() const { return _attributes;  }
	std::string datatype() const { return _datatype; }
	std::string value() const { return _value; }
	bool        is_array() const { return _array >= 0; }
	int         array() const { return _array; }
	std::list<std::string> more_names() const { return _more_names;  }

	void set_attributes(const std::string& value) { _attributes = value; }
	void set_datatype(const std::string& value) { _datatype = value; }
	void set_value(const std::string& value) { _value = value; }
	void set_array(int count) { _array = count; }
	void set_standalone(bool value) { _is_standalone = value;  }
	void add_name(const std::string& name) { _more_names.push_back(name); }

	virtual void print(std::ostream& out) const
	{
		out << _decorator
			<< " " << _type
			<< " " << _attributes
			<< " " << _datatype
			<< " " << _name;
		if (_more_names.size() > 0)
		{
			assert(!is_array());
			assert(_value.size() == 0);
			for (auto& s : _more_names)
			{
				out << ", " << s;
			}
		}
		else 
		{
			if (is_array())
			{
				if(_type == ".reg" || _type == ".pred")
					out << "<" << array() << ">" << _value;
				else
					out << "[" << array() << "]" << _value;
			}
			if (_value.size() > 0)
				out << " = " << _value;
		}
		if(_is_standalone)
			out << ";" << std::endl;
	}

protected:
	bool _is_standalone;
	std::string _attributes, _datatype, _value;
	int _array;
	std::list<std::string> _more_names;
};
static inline std::ostream& operator<<(std::ostream& out, const PtxVariable& variable) { variable.print(out); return out; }

class PtxCodeElement
{
public:
	virtual void print(std::ostream& out) const = 0;

	enum class Type { TypeBlock, TypeLabel, TypeVariable, TypeCode, TypeDirective };
	Type type() const { return _type; }

protected:
	PtxCodeElement(Type type) : _type(type)
	{

	}

	Type _type;
};

class PtxCodeLabel : public PtxCodeElement
{
public:
	PtxCodeLabel() : PtxCodeElement(Type::TypeLabel)
	{
	}

	bool set_name(const std::string& name) { _name = name; return true; }
	const std::string& name() const { return _name; }

	virtual void print(std::ostream& out) const override
	{
		out << _name << ":" << std::endl;
	}

private:
	std::string _name;
};

class PtxCodeVariable : public PtxCodeElement
{
public:
	PtxCodeVariable() : PtxCodeElement(Type::TypeVariable)
	{
	}

	PtxCodeVariable(const std::shared_ptr<PtxVariable>& value) : PtxCodeElement(Type::TypeVariable), _value(value)
	{
	}

	bool set(const std::shared_ptr<PtxVariable>& value) { _value = value; return true; }
	std::shared_ptr<PtxVariable> value() const { return _value; }

	virtual void print(std::ostream& out) const override
	{
		out << *_value;
	}

private:
	std::shared_ptr<PtxVariable> _value;
};

class InsArgument
{
public:
	enum class Type { Reg, Immediate, Variable, Vector };

	InsArgument(const std::string& name = std::string(), Type type = Type::Reg) : 
		_name(name)
		, _type(type)
		, _ofs(0)
		, _indirect(false)
		, _vector(nullptr)
	{
	}

	~InsArgument()
	{
		if (_vector != nullptr)
			delete _vector;
	}
	const std::string name() const { return _name;  }
	Type type() const { return _type;  }
	int ofs() const { return _ofs; }
	//int tex_coord() const { return _tex_coord; }
	//int tex_sampler() const { return _tex_sampler; }
	bool indirect() const { return _indirect; }

	bool set_name(const std::string& name) { _name = name; return true; }
	bool set_type(Type type) { _type = type; assert(type != Type::Vector); return true; }
	bool set_ofs(int ofs) { _ofs = ofs; return true;  }
	//bool set_tex_coord(int value) { return _tex_coord; }
	//bool set_tex_sampler(int value) { return _tex_sampler; }
	bool set_indirect() { _indirect = true; return true; }

	bool add_vector(const std::string& name)
	{
		if (_type != Type::Vector)
		{
			_type = Type::Vector;
			_vector = new std::list<std::string>();
		}
		_vector->push_back(name);
		return true;
	}

	void print(std::ostream& out) const
	{
		if (_type == Type::Vector)
		{
			print_vector(out);
			return;
		}

		if (_indirect)
			out << "[";
		
		out << _name;
		if (_ofs != 0)
			out << "+" << _ofs;

		if (_indirect)
			out << "]";
	}

	void print_vector(std::ostream& out) const
	{
		out << "{";
		bool comma = false;
		for (auto& v : *_vector)
		{
			if (comma)
				out << ",";
			else
				comma = true;
			out << v;
		}
		out << "}";
	}

private:
	std::string _name;
	Type _type;
	int _ofs;
	bool _indirect;
	//int _tex_coord, _tex_sampler;
	std::list<std::string> *_vector;
};
static inline std::ostream& operator<<(std::ostream& out, const InsArgument& variable) { variable.print(out); return out; }

class PtxCodeLine : public PtxCodeElement
{
public:
	PtxCodeLine() : PtxCodeElement(Type::TypeCode), _is_pred(false), _pred_neg(false), _is_func(false)
	{
	}

	bool set_predicate(const std::string& value, bool negative) { _is_pred = true; _pred = value; _pred_neg = negative;  return true; }
	std::string predicate() const { return _pred; }
    bool predicated() const { return _is_pred; }

	bool set_instruction(const std::string& value) { _ins = value; return true; }
	std::string instruction() const { return _ins; }

	bool add_modifier(const std::string& value) { _modifiers.push_back(value); return true; }
	const std::list<std::string>& modifiers() const { return _modifiers; }
	std::string modifier(int i)
	{
		auto it = _modifiers.begin();
		for (; i > 0; --i)
		{
			++it;
			assert(it != _modifiers.end());
		}
		return *it;
	}

	enum class StateSpace
	{
		SS_UNKNOWN, SS_GLOBAL, SS_LOCAL, SS_PARAM, SS_SHARED
	};

	struct SSMapping
	{
		std::string value;
		StateSpace ss;

		static const inline StateSpace to_enum(const std::string& value)
		{
			auto val = std::find_if(STATE_SPACES().begin(), STATE_SPACES().end(), [&](auto &ss) {return ss.value == value;  });
			if (val == STATE_SPACES().end())
				return StateSpace::SS_UNKNOWN;
			return val->ss;
		}

		static const inline std::string to_string(StateSpace value)
		{
			auto val = std::find_if(STATE_SPACES().begin(), STATE_SPACES().end(), [&](auto &ss) {return ss.ss == value;  });
			if (val == STATE_SPACES().end())
				return ".unknown";
			return val->value;
		}

		static const inline std::initializer_list<SSMapping>& STATE_SPACES()
		{
			static const std::initializer_list<SSMapping> STATE_SPACES = {
				{ ".global", StateSpace::SS_GLOBAL },{ ".local", StateSpace::SS_LOCAL },{ ".param", StateSpace::SS_PARAM },{ ".shared", StateSpace::SS_SHARED }
			};
			return STATE_SPACES;
		}
	};

	StateSpace get_state_space()
	{
		auto first = _modifiers.begin();
		if (first == _modifiers.end())
			return StateSpace::SS_UNKNOWN;

		static const std::string VOLATILE = ".volatile";

		if (*first == VOLATILE)
		{
			if (++first == _modifiers.end())
				return StateSpace::SS_UNKNOWN;
		}

		return SSMapping::to_enum(*first);
	}

	bool add_argument(const std::shared_ptr<InsArgument>& value) { _arguments.push_back(value); return true; }
	const std::list<std::shared_ptr<InsArgument>> arguments() const { return _arguments; }
	std::shared_ptr<InsArgument> argument(int i)
	{
		auto it = _arguments.begin();
		for (; i > 0; --i)
		{
			++it;
			assert(it != _arguments.end());
		}
		return *it;
	}

	bool set_func_name(const std::string& name) { _is_func = true;  _func_name = name; return true; }
	const std::string& func_name() const { return _func_name; }

	bool set_func_proto(const std::string& name) { _is_func = true;  _func_proto = name; return true; }
	const std::string& func_proto() const { return _func_proto; }

	bool set_func_retval(const std::shared_ptr<InsArgument>& arg) { _is_func = true;  _func_retval = arg; return true; }
	const std::shared_ptr<InsArgument> func_retval() const { return _func_retval; }

	virtual void print(std::ostream& out) const override
	{
		if (_pred.size() > 0)
			out << "@" << (_pred_neg ? "!" : "") << _pred << " ";
		out << _ins;
		for(auto s : _modifiers)
			out << s;
		out << " ";
		bool comma = false;
		if (_is_func)
		{
			if(_func_retval)
				out << "(" << *_func_retval << "),";
			out << _func_name;
			if (_arguments.size() > 0)
				out << ",(";
		}
		for (auto s : _arguments)
		{
			if (comma)
				out << ",";
			out << *s;
			comma = true;
		}
		if (_is_func)
		{
			out << ")";
			if (_func_proto.size() > 0)
				out << "," << _func_proto;
		}

		out << ";" << std::endl;
	}

private:
    bool _is_pred, _pred_neg;
	std::string _pred;
	std::string _ins;
	std::list<std::string> _modifiers;
	std::list<std::shared_ptr<InsArgument>> _arguments;
	bool _is_func;
	std::string _func_name;
	std::shared_ptr<InsArgument> _func_retval;
	std::string _func_proto;
};
static inline std::ostream& operator<<(std::ostream& out, const PtxCodeLine& line) { line.print(out); return out; }


class PtxCodeDirective : public PtxCodeElement
{
public:
	PtxCodeDirective(const std::shared_ptr<PtxDirective>& directive, bool need_semicolon) : 
		PtxCodeElement(Type::TypeDirective)
		, _directive(directive)
		, _need_semicolon(need_semicolon)
	{
	}

	bool set(const std::shared_ptr<PtxDirective>& value) { _directive = value; return true; }
	std::shared_ptr<PtxDirective> directive()  { return _directive; }

	virtual void print(std::ostream& out) const override
	{
		out << *_directive << (_need_semicolon ? ";" : "") << std::endl;
	}

private:
	std::shared_ptr<PtxDirective> _directive;
	bool _need_semicolon;
};

class PtxBlock : public PtxCodeElement
{
public:
	PtxBlock() : PtxCodeElement(Type::TypeBlock)
	{
	}

	virtual void print(std::ostream& out) const
	{
		out << "{" << std::endl;
		for (auto e : _elements)
			e->print(out);
		out << "}" << std::endl;
	}

	bool add(const std::shared_ptr<PtxCodeElement>& elm)
	{
		_elements.push_back(elm);
		return true;
	}

	using list_type = std::list<std::shared_ptr<PtxCodeElement>>;
	using list_iterator = list_type::iterator;

	inline list_iterator begin() { return _elements.begin(); }
	inline list_iterator end() { return _elements.end(); }

	inline list_type& elements() { return _elements;  }

protected:

	std::list<std::shared_ptr<PtxCodeElement>> _elements;
};
static inline std::ostream& operator<<(std::ostream& out, const PtxBlock& block) { block.print(out); return out; }


class PtxFunction : public PtxSymbol
{
public:
	PtxFunction() : PtxSymbol(DirType::Function)
	{
	}

	virtual void print(std::ostream& out) const override
	{
		out << _decorator << " " << _type << " ";
		if (_retval)
			out << "(" << *_retval << ")";
		out << _name << " (" << std::endl;
		bool comma = false;
		for (auto a : _args)
		{
			if (comma)
				out << ", " << std::endl;
			out << *a;
			comma = true;
		}
		out << std::endl << ")" << std::endl;
		if (has_main_block())
			out << *_main_block;
		else
			out << ";";
		out << std::endl;
	}

	std::shared_ptr<PtxVariable> retval() const { return _retval; }
	const std::list<std::shared_ptr<PtxVariable>>& args() const { return _args; }
	bool        has_main_block() const { return (bool)_main_block;  }

	void set_retval(const std::shared_ptr<PtxVariable>& value) { _retval = value; }
	void add_arg(const std::shared_ptr<PtxVariable>& value) { _args.push_back(value); }
	
	std::shared_ptr<PtxBlock>& main() { return _main_block; }
	void set_main(const std::shared_ptr<PtxBlock>& block) { _main_block = block; }

private:
	std::shared_ptr<PtxVariable> _retval;
	std::list<std::shared_ptr<PtxVariable>> _args;
	std::shared_ptr<PtxBlock> _main_block;
};

class PtxProgram
{
public:
	PtxProgram()
	{

	}

	bool add(const std::shared_ptr<PtxDirective>&& e)
	{
		_directives.push_back(e);
		return true;
	}

	virtual void print(std::ostream& out)
	{
		for (auto e : _directives)
			e->print(out);
	}

	std::shared_ptr<PtxDirective> find(const std::string& name)
	{
		auto it = std::find_if(_directives.begin(), _directives.end(), [&](auto& s) { return s->get_id() == name; });
		if (it == _directives.end())
			return std::shared_ptr<PtxDirective>();
		return *it;
	}

	using list_type = std::list<std::shared_ptr<PtxDirective>>;
	using list_iterator = list_type::iterator;

	list_type& directives()
	{
		return _directives;
	}

	inline list_iterator begin() { return _directives.begin(); }
	inline list_iterator end() { return _directives.end(); }

public:
	list_type _directives;
};

