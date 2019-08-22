#pragma once

#include "text_parser.hpp"
#include "ptx_program.hpp"
#include <algorithm>
#include <cstdlib>
#include <cctype>

class PtxParser
{
public:
	PtxParser()
	{
	}

	bool parse_ptx(const std::string& text)
	{
		TextParser parser(text);

		parser.root().many_of({
			[&](auto& s)->bool { return parse_directive(_ptx, s);  },
			[&](auto& s)->bool { return parse_variable(_ptx, s);  },
			[&](auto& s)->bool { return parse_function(_ptx, s);  },
		});

		parser.root().skip_whitespace();
			
		if (!parser.eos())
		{
			fprintf(stderr, "PTX has unparsed code.\n");
			return false;
		}
		return true;
	}

	PtxProgram& ptx()
	{
		return _ptx;
	}

private:
	using Scope = TextParser::Scope;
	using ParseResult = Scope::ParseResult;

	bool parse_directive(PtxProgram& ptx, Scope& scope)
	{
		bool first = true;

		scope.skip_whitespace();

		if (!scope.sequence_of(
			[&](auto type, auto& tok)->ParseResult {
				if (first && (type != Scope::TOKEN_IDENT || !is_directive(tok)))
					return Scope::ResultFail;
				first = false;
				if (type == Scope::TOKEN_NEWLINE)
					return Scope::ResultEndInclude;
				return Scope::ResultCont;
			}))
			return false;

		ptx.add(std::make_shared<PtxDirective>(scope.apply()));;
		return true;
	}

	bool parse_variable(PtxProgram& ptx, Scope& scope)
	{
		scope.set_skip_ws(true);

		std::shared_ptr<PtxVariable> variable = std::make_shared<PtxVariable>();
		return parse_variable(*variable, scope)
			&& ptx.add(variable);
	}

	bool parse_variable(PtxVariable& variable, Scope& scope)
	{
		return parse_decorator(variable, scope)
			&& parse_type(variable, scope)
			&& is_value_type(variable.type())
			&& parse_attributes(variable, scope)
			&& parse_data_type(variable, scope)
			&& parse_identifier(scope, [&](auto& s)->bool {variable.set_name(s); return true; })
			&& parse_array_value(variable, scope)
			&& parse_variable_value(variable, scope)
			&& scope.expect(Scope::TOKEN_PUNCT, ";")
			&& scope.apply().size() > 0;
	}

	bool parse_decorator(PtxSymbol& symbol, Scope& s)
	{
		s.optional_expect(Scope::TOKEN_IDENT, [&](auto& s)->bool { 
			if (!is_decorator(s))
				return false;
			symbol.set_decorator(s);
			return true;
		});
		return true;
	}

	bool parse_type(PtxSymbol& symbol, Scope& s)
	{
		return s.expect(Scope::TOKEN_IDENT, [&](auto& s)->bool { symbol.set_type(s); return true; });
	}

	bool parse_identifier(Scope& s, std::function<bool(std::string& str)> filter)
	{
		s.skip_whitespace();
		Scope scope(s);
		scope.set_skip_ws(false);
		std::string tok;
		auto tok_type = scope.token(&tok);
		if (tok_type == Scope::TOKEN_PUNCT)
		{
			if (!(tok.compare("$") == 0 || tok.compare("%") == 0))
				return false;
			tok_type = scope.token(&tok);
		}
		if (tok_type != Scope::TOKEN_IDENT)
			return false;
		Scope extra(scope);
		std::string val;
		extra.close_and_apply_if(extra.token(&val) == Scope::TOKEN_IDENT && val.at(0) == '.');
		return filter(scope.text())
			&& scope.apply().size() > 0;
	}

	bool parse_array_value(PtxVariable& symbol, Scope& s)
	{
		const char *openers[] = { "[", "<", NULL };
		const char *closers[] = { "]", ">", NULL };

		for (int i = 0; openers[i] != 0; ++i)
		{
			Scope scope(s);
			if (scope.expect(Scope::TOKEN_PUNCT, openers[i]))
			{
				if (parse_integer(scope, [&](int64_t v)->bool { symbol.set_array((int)v); return true; })
					&& scope.expect(Scope::TOKEN_PUNCT, closers[i]))
				{
					scope.apply();
					return true;
				}
				return false;
			}
		}
		return true;
	}

	bool parse_integer(Scope& s, std::function<bool(int64_t)> func)
	{
		std::string tok, tok2;

		auto tok_type = s.token(&tok);
		if (tok_type != Scope::TOKEN_DEC_NUMBER && tok_type != Scope::TOKEN_HEX_NUMBER)
			return false;

		bool is_unsigned = false;
		do
		{
			Scope scope(s);
			scope.set_skip_ws(false);
			if (scope.token(&tok2) == Scope::TOKEN_IDENT && tok2.compare("U") == 0)
			{
				is_unsigned = true;
				scope.apply();
			}
		} while (false); 

		int radix = 10;
		if (tok_type == Scope::TOKEN_HEX_NUMBER)
		{
			tok = tok.substr(2);
			radix = 16;
		}
		int64_t value = 0;
		if (is_unsigned)
			value = strtoull(tok.c_str(), NULL, radix);
		else
			value = strtoul(tok.c_str(), NULL, radix);

		return func(value);
	}

	bool parse_immediate(Scope& s, std::string* val)
	{
		Scope tmp(s);
		tmp.skip_whitespace();
		tmp.set_skip_ws(false);
		switch (tmp.token(val))
		{
		case Scope::TOKEN_DEC_NUMBER:
		case Scope::TOKEN_HEX_NUMBER:
		case Scope::TOKEN_FLOAT_BINARY:
			break;
		default:
			return false;
		}
		return tmp.apply().size() > 0;
	}

	bool parse_attributes(PtxVariable& symbol, Scope& s)
	{
		Scope scope(s);
		if(!scope.sequence_of(
			[&](auto type, auto& tok)->ParseResult {
			if (type == Scope::TOKEN_IDENT && is_data_type(tok))
				return Scope::ResultEndExclude;
			return Scope::ResultCont;
			}))
			return false;
		symbol.set_attributes(scope.apply());
		return true;
	}

	bool parse_data_type(PtxVariable& symbol, Scope& s)
	{
		return s.expect(Scope::TOKEN_IDENT, [&](auto& s)->bool {
			if (!is_data_type(s)) 
				return false; 
			symbol.set_datatype(s); 
			return true; 
		});
	}

	bool parse_variable_value(PtxVariable& variable, Scope& s)
	{
		Scope scope(s);
		std::string tok;
		if (scope.token(&tok) != Scope::TOKEN_PUNCT)
			return false;
		if (tok.compare("=") != 0)
			return true;

		if (!scope.sequence_of(
			[&](auto type, auto& tok)->ParseResult {
				if (type == Scope::TOKEN_PUNCT && tok.compare(";") == 0)
					return Scope::ResultEndExclude;
				return Scope::ResultCont;
			}))
			return false;

		scope.apply();

		return true;
	}

	bool parse_function(PtxProgram& ptx, Scope& scope)
	{
		scope.set_skip_ws(true);
		std::shared_ptr<PtxFunction> func = std::make_shared<PtxFunction>();

		if (!(   parse_decorator(*func, scope)
			  && parse_type(*func, scope)
			  && is_function(func->type())))
			return false;

		Scope retval(scope);
		retval.close_and_apply_if(retval.expect(Scope::TOKEN_PUNCT, "(")
			&& parse_retval_list(func, retval)
			&& retval.expect(Scope::TOKEN_PUNCT, ")"));

		if (! (   scope.expect(Scope::TOKEN_IDENT, [&](auto& s)->bool {func->set_name(s); return true; })
			   && scope.expect(Scope::TOKEN_PUNCT, "(")
			   && parse_function_args(func, scope)
			   && scope.expect(Scope::TOKEN_PUNCT, ")")))
			return false;

		Scope eof(scope);
		if (eof.expect(Scope::TOKEN_PUNCT, ";"))
		{
			eof.apply();
		}
		else
		{
			eof.cancel();
			if (!parse_code_block(func->main(), scope))
				return false;
		}
		ptx.add(func);
		return true;
	}

	bool parse_code_block(PtxBlock& block, Scope& scope)
	{
		if (!scope.expect(Scope::TOKEN_PUNCT, "{"))
			return false;

		// TypeBlock, TypeLabel, TypeVariable, TypeCode, TypeInvoke
		scope.many_of({
			[&](auto& s)->bool { 
						std::shared_ptr<PtxBlock> new_block = std::make_shared<PtxBlock>(); 
						if (!parse_code_block(*new_block, s))
							return false;
						block.add(std::static_pointer_cast<PtxCodeElement>(new_block));
						return true;
				},
			[&](auto& s)->bool { return parse_code_label(block, s);  },
			[&](auto& s)->bool { return parse_code_variable(block, s);  },
			[&](auto& s)->bool { return parse_code_statment(block, s);  },
		});

		if (!scope.expect(Scope::TOKEN_PUNCT, "}"))
			return false;

		return true;
	}

	bool parse_code_label(PtxBlock& block, Scope& scope)
	{
		std::string tok;
		scope.set_skip_ws(false);
		scope.skip_whitespace();
		if (!(scope.token(&tok) == Scope::TOKEN_IDENT
			&& scope.expect(Scope::TOKEN_PUNCT, ":")))
		{
			return false;
		}

		auto ptr = std::make_shared<PtxCodeLabel>();
		ptr->set_name(tok);
		block.add(std::static_pointer_cast<PtxCodeElement>(ptr));
		return true;
	}

	bool parse_code_variable(PtxBlock& block, Scope& scope)
	{
		auto var = std::make_shared<PtxVariable>();
		var->set_standalone(true);
		auto blkvar = std::make_shared<PtxCodeVariable>();
		return parse_variable(*var, scope)
			&& blkvar->set(var)
			&& block.add(std::static_pointer_cast<PtxCodeElement>(blkvar));
	}

	bool parse_code_statment(PtxBlock& block, Scope& scope)
	{
		auto stmt = std::make_shared<PtxCodeLine>();
		std::string val;

		scope.set_skip_ws(false);
		return parse_predicate(stmt, scope)
			&& parse_instruction(stmt, scope)
			&& parse_instruction_modifiers(stmt, scope)
			&& whitespace_or_future_eos(scope)
			&& parse_instruction_arguments(stmt, scope)
			&& scope.expect(Scope::TOKEN_PUNCT, ";")
			&& block.add(std::static_pointer_cast<PtxCodeElement>(stmt));
	}

	bool whitespace_or_future_eos(Scope& scope)
	{
		if (scope.skip_whitespace())
			return true;
		Scope eos(scope); // optional EOS
		return eos.expect(Scope::TOKEN_PUNCT, ";");
	}

	bool parse_predicate(std::shared_ptr<PtxCodeLine>& stmt, Scope& scope)
	{
		bool negative = false;

		Scope pred(scope);
		pred.skip_whitespace();
		pred.close_and_apply_if(pred.expect(Scope::TOKEN_PUNCT, "@")
			&& pred.optional_expect(Scope::TOKEN_PUNCT, [&](auto& s)->bool { 
				if (s == "!") {
					negative = true; 
					return true;
				}
				return false;
			})
			&& parse_identifier(pred, [&](auto& s)->bool {
				stmt->set_predicate(s, negative);
				return true;
		}));

		return true;
	}

	bool parse_instruction(std::shared_ptr<PtxCodeLine>& stmt, Scope& scope)
	{
		scope.skip_whitespace();

		std::string val;
		switch (scope.token(&val))
		{
		case Scope::TOKEN_PUNCT:
			if (val != ".")
				return false;
			if (scope.token(&val) != Scope::TOKEN_IDENT)
				return false;
			val = std::string(".") + val;
			if (!is_directive(val))
				return false;
			break;

		case Scope::TOKEN_IDENT:
			break;

		default:
			return false;
		}
		stmt->set_instruction(val);
		return true;
	}

	bool parse_instruction_modifiers(std::shared_ptr<PtxCodeLine>& stmt, Scope& scope)
	{
		std::string val;
		for (;;)
		{
			Scope mod(scope);
			auto tok = mod.token(&val);
			if (tok != Scope::TOKEN_IDENT || val[0] != '.')
			{
				if (tok != Scope::TOKEN_WHITESPACE && tok != Scope::TOKEN_NEWLINE && (tok != Scope::TOKEN_PUNCT || val != ";") )
					return false;
				break;
			}

			stmt->add_modifier(val);
			mod.apply();
		}
		return true;
	}

	bool parse_instruction_arguments(std::shared_ptr<PtxCodeLine>& stmt, Scope& scope)
	{
		scope.set_skip_ws(true);

		if (stmt->instruction() == "call")
			return parse_instruction_call_arguments(stmt, scope);
		else
			return parse_instruction_regular_arguments(stmt, scope);

/*		return scope.one_of({
			[&](auto& s)->bool { return parse_instruction_call_arguments(stmt, s); },
			[&](auto& s)->bool { return parse_instruction_regular_arguments(stmt, s); }
		});
		*/
	}

	bool parse_instruction_call_arguments(std::shared_ptr<PtxCodeLine>& stmt, Scope& scope)
	{
		std::shared_ptr<PtxCodeLine> tmpstmt = std::make_shared<PtxCodeLine>();
		if (!(parse_instruction_call_optional_retval(tmpstmt, scope)
			&& parse_instruction_call_function(tmpstmt, scope)
			&& parse_instruction_call_optional_args(tmpstmt, scope)
			&& parse_instruction_call_optional_proto(tmpstmt, scope)))
			return false;
		stmt->set_func_retval(tmpstmt->func_retval());
		stmt->set_func_name(tmpstmt->func_name());
		stmt->set_func_proto(tmpstmt->func_proto());
		for (auto& s : tmpstmt->arguments())
			stmt->add_argument(s);
		return true;
	}
	bool parse_instruction_regular_arguments(std::shared_ptr<PtxCodeLine>& stmt, Scope& scope)
	{
		Scope eoa(scope); // optional EOS
		std::string val;
		if (eoa.expect(Scope::TOKEN_PUNCT, ";"))
			return true;
		eoa.cancel();

		for (;;)
		{
			if (!parse_regular_argument(stmt, scope))
				return false;

			Scope eoa2(scope); // optional EOS
			std::string val;
			if (eoa2.token(&val) != Scope::TOKEN_PUNCT)
				return false;
			if (val == ";")
				break;
			if (val != ",")
				return false;
			eoa2.apply();
		}
		return true;
	}

	bool parse_regular_argument(std::shared_ptr<PtxCodeLine>& stmt, Scope& scope)
	{
		return scope.one_of({
			[&](auto& s)->bool {
				std::string val;
				if (!parse_immediate(s, &val))
					return false;
				std::shared_ptr<InsArgument> argument = std::make_shared<InsArgument>();
				argument->set_type(InsArgument::IMM);
				argument->set_name(val);
				return stmt->add_argument(argument); },
			[&](auto& s)->bool { return parse_vector_argument(stmt, s); },
			[&](auto& s)->bool { return parse_regular_indirect_argument(stmt, s); },
			[&](auto& s)->bool { return parse_identifier(s, [&](auto&s)->bool {
				std::shared_ptr<InsArgument> argument = std::make_shared<InsArgument>();
				argument->set_type((s.at(0) == '%') ? InsArgument::REG : InsArgument::VAR);
				argument->set_name(s);
				return stmt->add_argument(argument); }); 
			} 
		});
	}

	bool parse_vector_argument(std::shared_ptr<PtxCodeLine>& stmt, Scope& scope)
	{
		if (!scope.expect(Scope::TOKEN_PUNCT, "{"))
			return false;

		std::shared_ptr<InsArgument> argument = std::make_shared<InsArgument>();
		for (;;)
		{
			if (!parse_identifier(scope, [&](auto&s)->bool {
				return argument->add_vector(s);
				}))
				return false;

			Scope eoa(scope); // optional EOS
			std::string val;
			if (eoa.token(&val) != Scope::TOKEN_PUNCT)
				return false;
			if (val != ",")
			{
				if (val == "}")
				{
					eoa.apply();
					break;
				}
				return false;
			}
			eoa.apply();
		}

		stmt->add_argument(argument);

		return true;

	}

	bool parse_regular_indirect_argument(std::shared_ptr<PtxCodeLine>& stmt, Scope& scope)
	{
		if (!scope.expect(Scope::TOKEN_PUNCT, "["))
			return false;

		std::string name;
		int ofs = 0;

		std::shared_ptr<InsArgument> argument = std::make_shared<InsArgument>();
		argument->set_indirect();
		if (!parse_identifier(scope, [&](auto&s)->bool { name = s; return true; }))
			return false;

		Scope ofscope(scope);
		std::string val;
		ofscope.close_and_apply_if(ofscope.expect(Scope::TOKEN_PUNCT, "+")
			&& parse_integer(ofscope, [&](int64_t v)->bool { ofs = (int)v; return true; }));

		if (!scope.expect(Scope::TOKEN_PUNCT, "]"))
			return false;

		argument->set_type((name.at(0) == '%') ? InsArgument::REG : InsArgument::VAR);
		argument->set_name(name);
		argument->set_ofs(ofs);
		stmt->add_argument(argument);

		return true;
	}

	bool parse_instruction_call_optional_retval(std::shared_ptr<PtxCodeLine>& stmt, Scope& scope)
	{
		std::shared_ptr<InsArgument> argument;
		
		Scope retval(scope);
		if(retval.expect(Scope::TOKEN_PUNCT, "(")
			&& parse_identifier(retval, [&](auto& s)->bool { argument = std::make_shared<InsArgument>(); argument->set_name(s); return true; })
			&& retval.expect(Scope::TOKEN_PUNCT, ")")
			&& retval.expect(Scope::TOKEN_PUNCT, ",")
			&& retval.apply().size() > 0
			&& argument)
			stmt->set_func_retval(argument);
		return true;
	}

	bool parse_instruction_call_function(std::shared_ptr<PtxCodeLine>& stmt, Scope& scope)
	{
		return parse_identifier(scope, [&](auto& s)->bool {return stmt->set_func_name(s); });
	}

	bool parse_instruction_call_optional_args(std::shared_ptr<PtxCodeLine>& stmt, Scope& scope)
	{
		Scope args(scope);
		if (args.expect(Scope::TOKEN_PUNCT, ",")
			&& args.expect(Scope::TOKEN_PUNCT, "("))
		{
			for (;;)
			{
				if (!parse_regular_argument(stmt, args))
					return false;

				Scope eoa(args); // optional EOS
				std::string val;
				if (eoa.token(&val) != Scope::TOKEN_PUNCT)
					return false;
				if (val != ",")
				{
					if (val == ")")
					{
						eoa.apply();
						break;
					}
					return false;
				}
				eoa.apply();
			}
			args.apply();
		}
		return true;
	}

	bool parse_instruction_call_optional_proto(std::shared_ptr<PtxCodeLine>& stmt, Scope& scope)
	{
		Scope proto(scope);
		proto.close_and_apply_if(parse_identifier(proto, [&](auto& s)->bool {return stmt->set_func_proto(s); }));
		return true;
	}

	bool parse_retval_list(std::shared_ptr<PtxFunction>& func, Scope& s)
	{
		return parse_argument(s, [&](auto& s)->bool {func->set_retval(s); return true; });
	}

	bool parse_function_args(std::shared_ptr<PtxFunction>& func, Scope& s)
	{
		{
			Scope scope(s);
			std::string val;
			if (scope.expect(Scope::TOKEN_PUNCT, ")"))
				return true;
		}
		for (bool done = false;!done;)
		{
			auto variable = std::make_shared<PtxVariable>();
			if (!(parse_decorator(*variable, s)
				&& parse_type(*variable, s)
				&& is_value_type(variable->type())
				&& parse_attributes(*variable, s)
				&& parse_data_type(*variable, s)
				&& parse_identifier(s, [&](auto& s)->bool {variable->set_name(s); return true; })
				&& parse_array_value(*variable, s)
				&& parse_variable_value(*variable, s)))
				return false;
			variable->set_standalone(false);
			func->add_arg(variable);
			Scope scope(s);
			std::string val;
			if (scope.token(&val) != Scope::TOKEN_PUNCT)
				return false;
			if (val != ",")
				break;
			scope.apply();
		}
		
		return true;
	}

	bool parse_argument(Scope& s, std::function<bool(std::shared_ptr<PtxVariable>& var)> func)
	{
		auto var = std::make_shared<PtxVariable>();
		var->set_standalone(false);
		return parse_type(*var, s)
			&& is_value_type(var->type())
			&& parse_data_type(*var, s)
			&& parse_identifier(s, [&](auto& s)->bool {var->set_name(s); return true; })
			&& func(var);
	}

	bool is_directive(const std::string& str)
	{
		static std::initializer_list<std::string> DIRECTIVES = {
			".address_size", ".file", ".minnctapersm", ".target", ".align", ".branchtargets", ".pragma", ".version", ".callprototype", ".loc", ".calltargets",
			".reqntid", ".maxnctapersm", ".section", ".maxnreg", ".maxntid"
		};
		return std::find(DIRECTIVES.begin(), DIRECTIVES.end(), str) != DIRECTIVES.end();
	}

	bool is_function(const std::string& str)
	{
		static std::initializer_list<std::string> FUNC_DIRECTIVES = {
			".func", ".entry"
		};
		return std::find(FUNC_DIRECTIVES.begin(), FUNC_DIRECTIVES.end(), str) != FUNC_DIRECTIVES.end();
	}

	bool is_value_type(const std::string& str)
	{
		static std::initializer_list<std::string> VALUE_DIRECTIVES = {
			".global", ".local", ".const", ".reg", ".param", ".shared", ".tex", ".sreg"
		};
		return std::find(VALUE_DIRECTIVES.begin(), VALUE_DIRECTIVES.end(), str) != VALUE_DIRECTIVES.end();
	}

	bool is_decorator(const std::string& str)
	{
		static std::initializer_list<std::string> DECORATORS = {
			".weak", ".visible", ".extern"
		};
		return std::find(DECORATORS.begin(), DECORATORS.end(), str) != DECORATORS.end();
	}

	struct Type
	{
		std::string name;
		enum BasicType { Signed, Unsigned, Untyped, Float, Predicate };
		BasicType type;
		int width;
	};
	bool is_data_type(const std::string& str, Type* type = NULL)
	{
		static std::initializer_list<Type> TYPES = {
			{ ".s8",  Type::Signed,     8 },
			{ ".s16", Type::Signed,    16 },
			{ ".s32", Type::Signed,    32 },
			{ ".s64", Type::Signed,    64 },
			{ ".u8",  Type::Unsigned,   8 },
			{ ".u16", Type::Unsigned,  16 },
			{ ".u32", Type::Unsigned,  32 },
			{ ".u64", Type::Unsigned,  64 },
			{ ".b8",  Type::Untyped,    8 },
			{ ".b16", Type::Untyped,   16 },
			{ ".b32", Type::Untyped,   32 },
			{ ".b64", Type::Untyped,   64 },
			{ ".f8",  Type::Float,      8 },
			{ ".f16", Type::Float,     16 },
			{ ".f32", Type::Float,     32 },
			{ ".f64", Type::Float,     64 },
			{ ".pred", Type::Predicate, 0 }
		};
		auto it = std::find_if(TYPES.begin(), TYPES.end(), [&](auto& s)->bool {return s.name == str; });
		if (it == TYPES.end())
			return false;
		if (type != NULL)
			*type = *it;
		return true;
	}

private:
	PtxProgram _ptx;

};
