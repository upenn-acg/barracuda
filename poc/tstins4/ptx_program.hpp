#pragma once

#include <string>
#include <list>
#include <memory>
#include <cassert>

class PtxDirective
{
public:
	PtxDirective(const std::string& line) : _line(line), _type(Directive)
	{

	}

	enum Type { Directive, Variable, Function };

	Type type() const
	{
		return _type;
	}

	virtual void print(std::ostream& out) const
	{
		out << _line;
	}

protected:
	std::string _line;
	Type _type;
};

class PtxSymbol : public PtxDirective
{
public:
	PtxSymbol() : PtxDirective("")
	{

	}

	std::string decorator() const { return _decorator; }
	std::string type() const { return _type; }
	std::string name() const { return _name; }

	void set_decorator(const std::string& value) { _decorator = value; }
	void set_type(const std::string& value) { _type = value; }
	void set_name(const std::string& value) { _name = value; }

protected:
	std::string _decorator, _type, _name;
};

class PtxVariable : public PtxSymbol
{
public:
	PtxVariable() : _array(-1), _is_standalone(true)
	{

	}

	std::string attributes() const { return _attributes;  }
	std::string datatype() const { return _datatype; }
	std::string value() const { return _value; }
	bool        is_array() const { return _array >= 0; }
	int         array() const { return _array; }
	
	void set_attributes(const std::string& value) { _attributes = value; }
	void set_datatype(const std::string& value) { _datatype = value; }
	void set_value(const std::string& value) { _value = value; }
	void set_array(int count) { _array = count; }
	void set_standalone(bool value) { _is_standalone = value;  }

	virtual void print(std::ostream& out) const
	{
		out << _decorator
			<< " " << _type
			<< " " << _attributes
			<< " " << _datatype
			<< " " << _name;
		if (is_array())
		{
			if(_type == ".reg" || _type == ".pred")
				out << "<" << array() << ">" << _value;
			else
				out << "[" << array() << "]" << _value;
		}
		if (_value.size() > 0)
			out << " = " << _value;
		if(_is_standalone)
			out << ";" << std::endl;
	}

protected:
	bool _is_standalone;
	std::string _attributes, _datatype, _value;
	int _array;
};
static inline std::ostream& operator<<(std::ostream& out, PtxVariable& variable) { variable.print(out); return out; }

class PtxCodeElement
{
public:
	virtual void print(std::ostream& out) const = 0;

	enum Type { TypeBlock, TypeLabel, TypeVariable, TypeCode };
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
	PtxCodeLabel() : PtxCodeElement(TypeLabel)
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
	PtxCodeVariable() : PtxCodeElement(TypeVariable)
	{
	}

	bool set(std::shared_ptr<PtxVariable>& value) { _value = value; return true; }
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
	enum Type { REG, IMM, VAR, VECTOR };

	InsArgument() : _type(REG), _ofs(0), _indirect(false), _vector(NULL)
	{
	}

	~InsArgument()
	{
		if (_vector != NULL)
			delete _vector;
	}
	const std::string name() const { return _name;  }
	Type type() const { return _type;  }
	int ofs() const { return _ofs; }
	bool indirect() const { return _indirect; }

	bool set_name(const std::string& name) { _name = name; return true; }
	bool set_type(Type type) { _type = type; assert(type != VECTOR); return true; }
	bool set_ofs(int ofs) { _ofs = ofs; return true;  }
	bool set_indirect() { _indirect = true; return true; }

	bool add_vector(const std::string& name)
	{
		if (_type != VECTOR)
		{
			_type = VECTOR;
			_vector = new std::list<std::string>();
		}
		_vector->push_back(name);
		return true;
	}

	void print(std::ostream& out) const
	{
		if (_type == VECTOR)
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
	std::list<std::string> *_vector;
};
static inline std::ostream& operator<<(std::ostream& out, InsArgument& variable) { variable.print(out); return out; }

class PtxCodeLine : public PtxCodeElement
{
public:
	PtxCodeLine() : PtxCodeElement(TypeCode), _is_func(false), _pred_neg(false)
	{
	}

	bool set_predicate(const std::string& value, bool negative) { _pred = value; _pred_neg = negative;  return true; }
	std::string predicate() const { return _pred; }

	bool set_instruction(const std::string& value) { _ins = value; return true; }
	std::string instruction() const { return _ins; }

	bool add_modifier(const std::string& value) { _modifiers.push_back(value); return true; }
	const std::list<std::string> modifiers() const { return _modifiers; }

	bool add_argument(const std::shared_ptr<InsArgument>& value) { _arguments.push_back(value); return true; }
	const std::list<std::shared_ptr<InsArgument>> arguments() const { return _arguments; }

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
	std::string _pred;
	std::string _ins;
	std::list<std::string> _modifiers;
	std::list<std::shared_ptr<InsArgument>> _arguments;
	bool _is_func, _pred_neg;
	std::string _func_name;
	std::shared_ptr<InsArgument> _func_retval;
	std::string _func_proto;
};

class PtxBlock : public PtxCodeElement
{
public:
	PtxBlock() : PtxCodeElement(TypeBlock)
	{
	}

	virtual void print(std::ostream& out) const
	{
		out << "{" << std::endl;
		for (auto e : _elements)
			e->print(out);
		out << "}" << std::endl;
	}

	bool add(std::shared_ptr<PtxCodeElement>& elm)
	{
		_elements.push_back(elm);
		return true;
	}

protected:
	std::list<std::shared_ptr<PtxCodeElement>> _elements;
};
static inline std::ostream& operator<<(std::ostream& out, const PtxBlock& block) { block.print(out); return out; }


class PtxFunction : public PtxSymbol
{
public:
	PtxFunction() : _has_main_block(false)
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
		if (_has_main_block)
			out << _main_block;
		else
			out << ";";
		out << std::endl;
	}

	std::shared_ptr<PtxVariable> retval() const { return _retval; }
	const std::list<std::shared_ptr<PtxVariable>>& args() const { return _args; }
	bool        has_main_block() const { return _has_main_block;  }

	void set_retval(std::shared_ptr<PtxVariable>& value) { _retval = value; }
	void add_arg(std::shared_ptr<PtxVariable>& value) { _args.push_back(value); }
	
	PtxBlock& main()
	{
		_has_main_block = true;
		return _main_block;
	}

private:
	std::shared_ptr<PtxVariable> _retval;
	std::list<std::shared_ptr<PtxVariable>> _args;
	bool _has_main_block;
	PtxBlock _main_block;
};

class PtxProgram
{
public:
	PtxProgram()
	{

	}

	bool add(std::shared_ptr<PtxDirective>&& e)
	{
		_directives.push_back(e);
		return true;
	}

	virtual void print(std::ostream& out)
	{
		for (auto e : _directives)
			e->print(out);
	}

public:
	std::list<std::shared_ptr<PtxDirective>> _directives;
};

