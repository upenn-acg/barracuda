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

	struct ParseStatus
	{
		int parsed;
		int left;
		int line;
	};
	int parse_ptx(const std::string& text, ParseStatus* status = nullptr)
	{
		TextParser parser(text);

		parser.root().many_of({
			[&](auto& s)->bool { return parse_program_directive(_ptx, s);  },
			[&](auto& s)->bool { return parse_variable(_ptx, s);  },
			[&](auto& s)->bool { return parse_function(_ptx, s);  },
		});

		parser.root().skip_whitespace();
			
		if (status != nullptr)
		{
			status->left = parser.left();
			status->parsed = text.size() - status->left;
			status->line = parser.line();
		}
		return parser.eos();
	}

	PtxProgram& ptx()
	{
		return _ptx;
	}

	enum class TypeClass { Signed, Unsigned, Untyped, Float, Predicate, TextureRef };

	struct Type
	{
		std::string name;
		TypeClass type;
		int width;
	};

	static bool is_data_type(const std::string& str, Type* type = nullptr)
	{
		static const std::initializer_list<Type> TYPES = {
			{ ".s8",  TypeClass::Signed,     8 },
			{ ".s16", TypeClass::Signed,    16 },
			{ ".s32", TypeClass::Signed,    32 },
			{ ".s64", TypeClass::Signed,    64 },
			{ ".u8",  TypeClass::Unsigned,   8 },
			{ ".u16", TypeClass::Unsigned,  16 },
			{ ".u32", TypeClass::Unsigned,  32 },
			{ ".u64", TypeClass::Unsigned,  64 },
			{ ".b8",  TypeClass::Untyped,    8 },
			{ ".b16", TypeClass::Untyped,   16 },
			{ ".b32", TypeClass::Untyped,   32 },
			{ ".b64", TypeClass::Untyped,   64 },
			{ ".f8",  TypeClass::Float,      8 },
			{ ".f16", TypeClass::Float,     16 },
			{ ".f32", TypeClass::Float,     32 },
			{ ".f64", TypeClass::Float,     64 },
			{ ".pred", TypeClass::Predicate, 0 },
			{ ".texref", TypeClass::TextureRef, 0 }
		};
		auto it = std::find_if(TYPES.begin(), TYPES.end(), [&](auto& s)->bool {return s.name == str; });
		if (it == TYPES.end())
			return false;
		if (type != nullptr)
			*type = *it;
		return true;
	}

private:
	using Scope = TextParser::Scope;
	using TokenType = Scope::TokenType;
	using ParseResult = Scope::ParseResult;

	bool parse_program_directive(PtxProgram& ptx, Scope& scope)
	{
		std::shared_ptr<PtxDirective> ptx_dir;
		if (!parse_directive(scope, &ptx_dir))
			return false;
		ptx.add(ptx_dir);
		scope.apply();
		return true;
	}

	bool parse_directive(Scope& scope, std::shared_ptr<PtxDirective>* ptx_dir)
	{
		scope.skip_whitespace();
		std::string first;
		Scope first_scope(scope);
		auto tok = first_scope.token(&first);
		if (tok != TokenType::Identifier || !is_directive(first))
			return false;
		first_scope.apply();
		scope.skip_whitespace();

		Scope subscope(scope);
		if (!subscope.sequence_of(
			[&](auto type, auto& tok)->ParseResult {
				if (type == TokenType::Newline)
					return ParseResult::SuccessInclusive;
				return ParseResult::Continue;
			}))
			return false;

		*ptx_dir = std::make_shared<PtxDirective>(first, subscope.apply());
		scope.apply();
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
			&& scope.expect(TokenType::Punctuation, ";")
			&& scope.apply().size() > 0;
	}

	bool parse_decorator(PtxSymbol& symbol, Scope& s)
	{
		s.optional_expect(TokenType::Identifier, [&](auto& s)->bool {
			if (!is_decorator(s))
				return false;
			symbol.set_decorator(s);
			return true;
		});
		return true;
	}

	bool parse_type(PtxSymbol& symbol, Scope& s)
	{
		return s.expect(TokenType::Identifier, [&](auto& s)->bool { symbol.set_type(s); return true; });
	}

	bool parse_identifier(Scope& s, std::function<bool(const std::string& str)> filter)
	{
		s.skip_whitespace();
		Scope scope(s);
		scope.set_skip_ws(false);
		std::string tok;
		auto tok_type = scope.token(&tok);
		if (tok_type == TokenType::Punctuation)
		{
			if (!(tok.compare("$") == 0 || tok.compare("%") == 0))
				return false;
			tok_type = scope.token(&tok);
		}
		if (tok_type != TokenType::Identifier)
			return false;
		Scope extra(scope);
		std::string val;
		extra.close_and_apply_if(extra.token(&val) == TokenType::Identifier && val.at(0) == '.');
		return filter(scope.text())
			&& scope.apply().size() > 0;
	}

	bool parse_array_value(PtxVariable& symbol, Scope& s)
	{
		static const std::initializer_list<std::pair<std::string, std::string>> BRACES =
		{ {"[", "]"}, { "<", ">" } };

		for(const auto& brace : BRACES)
		{
			Scope scope(s);
			if (scope.expect(TokenType::Punctuation, brace.first))
			{
				Scope val(scope);
                symbol.set_unknown_array();
				val.close_and_apply_if(parse_integer(val, [&](int64_t v)->bool { symbol.set_array((int)v); return true; }));

				if(scope.expect(TokenType::Punctuation, brace.second))
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
		if (tok_type != TokenType::DecNumber && tok_type != TokenType::HexNumber)
			return false;

		bool is_unsigned = false;
		do
		{
			Scope scope(s);
			scope.set_skip_ws(false);
			if (scope.token(&tok2) == TokenType::Identifier && tok2.compare("U") == 0)
			{
				is_unsigned = true;
				scope.apply();
			}
		} while (false); 

		int radix = 10;
		if (tok_type == TokenType::HexNumber)
		{
			tok = tok.substr(2);
			radix = 16;
		}
		int64_t value = 0;
		if (is_unsigned)
			value = strtoull(tok.c_str(), nullptr, radix);
		else
			value = strtoul(tok.c_str(), nullptr, radix);

		return func(value);
	}

	bool parse_immediate(Scope& s, std::string* val)
	{
		Scope tmp(s);
		tmp.skip_whitespace();
		tmp.set_skip_ws(false);
		switch (tmp.token(val))
		{
		case TokenType::DecNumber:
		case TokenType::HexNumber:
		case TokenType::FloatBinary:
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
			if (type == TokenType::Identifier && is_data_type(tok))
				return ParseResult::SuccessExclusive;
			return ParseResult::Continue;
			}))
			return false;
		symbol.set_attributes(scope.apply());
		return true;
	}

	bool parse_data_type(PtxVariable& symbol, Scope& s)
	{
		return s.expect(TokenType::Identifier, [&](auto& s)->bool {
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
		if (scope.token(&tok) != TokenType::Punctuation)
			return false;
		if (tok.compare("=") != 0)
			return true;

		if (!scope.sequence_of(
			[&](auto type, auto& tok)->ParseResult {
				if (type == TokenType::Punctuation && tok.compare(";") == 0)
					return ParseResult::SuccessExclusive;
				return ParseResult::Continue;
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
		retval.close_and_apply_if(retval.expect(TokenType::Punctuation, "(")
			&& parse_retval_list(func, retval)
			&& retval.expect(TokenType::Punctuation, ")"));

		if (! (   scope.expect(TokenType::Identifier, [&](auto& s)->bool {func->set_name(s); return true; })
			   && scope.expect(TokenType::Punctuation, "(")
			   && parse_function_args(func, scope)
			   && scope.expect(TokenType::Punctuation, ")")))
			return false;

		Scope eof(scope);
		if (eof.expect(TokenType::Punctuation, ";"))
		{
			eof.apply();
		}
		else
		{
			eof.cancel();
			for (;;)
			{
				Scope dirscope(scope);
				dirscope.set_skip_ws(false);
				std::shared_ptr<PtxDirective> ptx_dir;
				if (!parse_directive(dirscope, &ptx_dir))
					break;
				func->add_header_directive(ptx_dir);
				dirscope.apply();
			}

			func->set_main(std::make_shared<PtxBlock>());
			if (!parse_code_block(func->main(), scope))
				return false;
		}
		ptx.add(func);
		return true;
	}

	bool parse_code_block(std::shared_ptr<PtxBlock>& block, Scope& scope)
	{
		if (!scope.expect(TokenType::Punctuation, "{"))
			return false;

		// TypeBlock, TypeLabel, TypeVariable, TypeCode, TypeInvoke
		scope.many_of({
			[&](auto& s)->bool { 
						std::shared_ptr<PtxBlock> new_block = std::make_shared<PtxBlock>(); 
						if (!parse_code_block(new_block, s))
							return false;
						block->add(std::static_pointer_cast<PtxCodeElement>(new_block));
						return true;
				},
			[&](auto& s)->bool { return parse_code_label(block, s);  },
			[&](auto& s)->bool { return parse_code_variable(block, s);  },
			[&](auto& s)->bool { return parse_code_directive(block, s); },
			[&](auto& s)->bool { return parse_code_statment(block, s);  },
		});

		if (!scope.expect(TokenType::Punctuation, "}"))
		{
			printf("Function parse error at %i:%s\n", scope.line(), scope.next_text().c_str());
			return false;
		}

		return true;
	}

	bool parse_code_label(std::shared_ptr<PtxBlock>& block, Scope& scope)
	{
		std::string tok;
		scope.set_skip_ws(false);
		scope.skip_whitespace();
		if (!(parse_identifier(scope, [&](auto& s)->bool { tok = s; return true; })
			&& scope.expect(TokenType::Punctuation, ":")))
		{
			return false;
		}

		auto ptr = std::make_shared<PtxCodeLabel>();
		ptr->set_name(tok);
		block->add(std::static_pointer_cast<PtxCodeElement>(ptr));
		return true;
	}

	bool parse_code_variable(std::shared_ptr<PtxBlock>& block, Scope& scope)
	{
		auto var = std::make_shared<PtxVariable>();
		var->set_standalone(true);
		auto blkvar = std::make_shared<PtxCodeVariable>();
		return parse_variable(*var, scope)
			&& blkvar->set(var)
			&& block->add(std::static_pointer_cast<PtxCodeElement>(blkvar));
	}

	bool parse_code_statment(std::shared_ptr<PtxBlock>& block, Scope& scope)
	{
		auto stmt = std::make_shared<PtxCodeLine>();
		std::string val;

		scope.set_skip_ws(false);
		return parse_predicate(stmt, scope)
			&& parse_instruction(stmt, scope)
			&& parse_instruction_modifiers(stmt, scope)
			&& whitespace_or_future_eos(scope)
			&& parse_instruction_arguments(stmt, scope)
			&& scope.expect(TokenType::Punctuation, ";")
			&& block->add(std::static_pointer_cast<PtxCodeElement>(stmt));
	}

	bool parse_code_directive(std::shared_ptr<PtxBlock>& block, Scope& scope)
	{
		bool skip = false;

		scope.skip_whitespace();
		std::string first;
		Scope first_scope(scope);
		auto tok = first_scope.token(&first);
		if (tok != TokenType::Identifier || !is_directive(first))
			return false;
		first_scope.apply();
		scope.skip_whitespace();

		Scope subscope(scope);
		subscope.set_skip_ws(false);
		if (!subscope.sequence_of(
			[&](auto type, auto& tok)->ParseResult {
			if (type == TokenType::Newline)
				return ParseResult::SuccessExclusive;
			if (type == TokenType::Punctuation && tok == ";")
			{
				skip = true;
				return ParseResult::SuccessExclusive;
			}
			return ParseResult::Continue;
		}))
			return false;

		auto dir = std::make_shared<PtxDirective>(first, subscope.apply());
		auto codedir = std::make_shared<PtxCodeDirective>(dir, skip);
		block->add(std::static_pointer_cast<PtxCodeElement>(codedir));
		if (skip)
		{
			std::string unused;
			scope.token(&unused);
		}
		scope.apply();
		return true;
	}

	bool whitespace_or_future_eos(Scope& scope)
	{
		if (scope.skip_whitespace())
			return true;
		Scope eos(scope); // optional EOS
		return eos.expect(TokenType::Punctuation, ";");
	}

	bool parse_predicate(std::shared_ptr<PtxCodeLine>& stmt, Scope& scope)
	{
		bool negative = false;

		Scope pred(scope);
		pred.skip_whitespace();
		pred.close_and_apply_if(pred.expect(TokenType::Punctuation, "@")
			&& pred.optional_expect(TokenType::Punctuation, [&](auto& s)->bool { 
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
		case TokenType::Punctuation:
			if (val != ".")
				return false;
			if (scope.token(&val) != TokenType::Identifier)
				return false;
			val = std::string(".") + val;
			if (!is_directive(val))
				return false;
			break;

		case TokenType::Identifier:
			break;

		default:
			return false;
		}
		stmt->set_instruction(val);
		return true;
	}

	bool parse_instruction_modifiers(std::shared_ptr<PtxCodeLine>& stmt, Scope& scope)
	{
		bool done = false;
		while(!done)
		{
			// some modifiers can begin with numbers, such as tex.3d.... 
			// so we cannot just parse identifiers as modifier strings
			Scope mod(scope);
			Scope first(mod);
			std::string val;
			auto tok = first.token(&val);

			if ((tok != TokenType::Punctuation && tok != TokenType::Identifier) || val[0] != '.')
				return true;
			first.apply();

			for (;;)
			{
				std::string next_val;
				Scope next(mod);
				tok = next.token(&next_val);
				if (tok == TokenType::Newline || tok == TokenType::Whitespace || next_val[0] == ';')
				{
					done = true;
					break;
				}
				if ((tok == TokenType::Punctuation || tok == TokenType::Identifier) && next_val[0] == '.')
					break;
				next.apply();
			}

			stmt->add_modifier(mod.apply());
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
		if (eoa.expect(TokenType::Punctuation, ";"))
			return true;
		eoa.cancel();

		for (;;)
		{
			if (!parse_regular_argument(stmt, scope))
				return false;

			Scope eoa2(scope); // optional EOS
			std::string val;
			if (eoa2.token(&val) != TokenType::Punctuation)
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
				argument->set_type(InsArgument::Type::Immediate);
				argument->set_name(val);
				return stmt->add_argument(argument); },
			[&](auto& s)->bool { return parse_vector_argument(stmt, s); },
			[&](auto& s)->bool { return parse_regular_indirect_argument(stmt, s); },
			[&](auto& s)->bool { 
				std::shared_ptr<InsArgument> argument;
                if(!parse_identifier(s, [&](auto&s)->bool {
				    argument = std::make_shared<InsArgument>();
    				argument->set_type((s.at(0) == '%') ? InsArgument::Type::Reg : InsArgument::Type::Variable);
    				argument->set_name(s);
    				return stmt->add_argument(argument); }))
                {
                    return false;;
                }
                Scope subscope(s);
                subscope.close_and_apply_if(
                    subscope.expect(TokenType::Punctuation, "|")
                    && parse_identifier(subscope, [&](auto&s)->bool {
                        return argument->set_optional(s);
                    })
                );
                    
                return true;
			} 
		});
	}

	bool parse_vector_argument(std::shared_ptr<PtxCodeLine>& stmt, Scope& scope)
	{
		if (!scope.expect(TokenType::Punctuation, "{"))
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
			if (eoa.token(&val) != TokenType::Punctuation)
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
		if (!scope.expect(TokenType::Punctuation, "["))
			return false;

		std::string name;
		int ofs = 0;

		std::shared_ptr<InsArgument> argument = std::make_shared<InsArgument>();
		argument->set_indirect();
		if (!parse_identifier(scope, [&](auto&s)->bool { name = s; return true; }))
			return false;

		Scope ofscope(scope);
		std::string val;
		ofscope.close_and_apply_if(ofscope.expect(TokenType::Punctuation, "+")
			&& parse_integer(ofscope, [&](int64_t v)->bool { ofs = (int)v; return true; }));

		if (!optional_parse_texture(/*texarg,*/ scope))
			return false;

		scope.skip_whitespace();
		if (!scope.expect(TokenType::Punctuation, "]"))
			return false;

		argument->set_type((name.at(0) == '%') ? InsArgument::Type::Reg : InsArgument::Type::Variable);
		argument->set_name(name);
		argument->set_ofs(ofs);
		stmt->add_argument(argument);

		return true;
	}

	bool optional_parse_texture(/*TexArgument& tex,*/ Scope& s)
	{
		// not implemented for now, just verifies parsing!
		Scope scope(s);
		scope.skip_whitespace(); 
		if (!scope.expect(TokenType::Punctuation, ","))
			return true;

		scope.skip_whitespace();
		Scope sampler(scope);
		sampler.close_and_apply_if(parse_identifier(sampler, [&](auto& s)->bool {return true; })
			&& sampler.expect(TokenType::Punctuation, ","));
		scope.skip_whitespace();
		if (!scope.expect(TokenType::Punctuation, "{"))
			return true;
			
		for (;;)
		{
			scope.skip_whitespace();
			if (!parse_identifier(scope, [&](auto&s)->bool {return true; }))
				return false;
			scope.skip_whitespace();
			std::string val;
			if (scope.token(&val) != TokenType::Punctuation)
				return false;
			if (val != ",")
			{
				if (val != "}")
					return false;
				break;
			}
		}
		scope.apply();
		return true;
	}

	bool parse_instruction_call_optional_retval(std::shared_ptr<PtxCodeLine>& stmt, Scope& scope)
	{
		std::shared_ptr<InsArgument> argument;
		
		Scope retval(scope);
		if(retval.expect(TokenType::Punctuation, "(")
			&& parse_identifier(retval, [&](auto& s)->bool { argument = std::make_shared<InsArgument>(); argument->set_name(s); return true; })
			&& retval.expect(TokenType::Punctuation, ")")
			&& retval.expect(TokenType::Punctuation, ",")
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
		if (args.expect(TokenType::Punctuation, ",")
			&& args.expect(TokenType::Punctuation, "("))
		{
			for (;;)
			{
				if (!parse_regular_argument(stmt, args))
					return false;

				Scope eoa(args); // optional EOS
				std::string val;
				if (eoa.token(&val) != TokenType::Punctuation)
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
			if (scope.expect(TokenType::Punctuation, ")"))
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
			if (scope.token(&val) != TokenType::Punctuation)
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
		static const std::initializer_list<std::string> DIRECTIVES = {
			".address_size", ".file", ".minnctapersm", ".target", ".align", ".branchtargets", ".pragma", ".version", ".callprototype", ".loc", ".calltargets",
			".reqntid", ".maxnctapersm", ".section", ".maxnreg", ".maxntid"
		};
		return std::find(DIRECTIVES.begin(), DIRECTIVES.end(), str) != DIRECTIVES.end();
	}

	bool is_function(const std::string& str)
	{
		static const std::initializer_list<std::string> FUNC_DIRECTIVES = {
			".func", ".entry"
		};
		return std::find(FUNC_DIRECTIVES.begin(), FUNC_DIRECTIVES.end(), str) != FUNC_DIRECTIVES.end();
	}

	bool is_value_type(const std::string& str)
	{
		static const std::initializer_list<std::string> VALUE_DIRECTIVES = {
			".global", ".local", ".const", ".reg", ".param", ".shared", ".tex", ".sreg"
		};
		return std::find(VALUE_DIRECTIVES.begin(), VALUE_DIRECTIVES.end(), str) != VALUE_DIRECTIVES.end();
	}

	bool is_decorator(const std::string& str)
	{
		static const std::initializer_list<std::string> DECORATORS = {
			".weak", ".visible", ".extern"
		};
		return std::find(DECORATORS.begin(), DECORATORS.end(), str) != DECORATORS.end();
	}

private:
	PtxProgram _ptx;

};
