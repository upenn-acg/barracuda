#include "text_parser.hpp"

class TestCase
{
public:
	TestCase(const char* name) : _name(name), _status(true)
	{

	}
	~TestCase()
	{
		if (_status)
			_success += 1;
		else
			_failed += 1;
	}

	void test(bool condition)
	{
		if (!condition)
			failed();
	}

	void failed()
	{
		_status = false;
		fprintf(stderr, "Failed: %s\n", _name);
	}

	static void print_stats()
	{
		printf("Succeeded: %i/%i\n", _success, _success + _failed);
	}

private:
	static int _failed;
	static int _success;
	const char* _name;
	bool _status;

};

int TestCase::_failed = 0;
int TestCase::_success = 0;

void test_parsing_helper(TestCase& tc, const std::string& text, 
		std::initializer_list<TextParser::Scope::TokenType> types,
		std::initializer_list<std::string> values)
{
	TextParser parser(text);
	TextParser::Scope& root = parser.root();

	std::string tok;
	auto types_it = types.begin();
	auto values_it = values.begin();
	for(;types_it != types.end() && values_it != values.end(); ++ types_it, ++ values_it)
	{
		TextParser::Scope::TokenType toktype;

		toktype = root.token(&tok);
		if (toktype != *types_it || tok.compare(*values_it) != 0)
		{
			tc.failed();
			return;
		}
	}
}

void test_parsing()
{
	test_parsing_helper(TestCase("number"), "1234", { TextParser::Scope::TOKEN_DEC_NUMBER, }, { "1234" });
	test_parsing_helper(TestCase("negative-number"), "-1234", { TextParser::Scope::TOKEN_DEC_NUMBER, }, { "-1234" });
	test_parsing_helper(TestCase("hexnumber"), "0x1234", { TextParser::Scope::TOKEN_HEX_NUMBER, }, { "0x1234" });
	test_parsing_helper(TestCase("number_punct"), "1234.", { TextParser::Scope::TOKEN_DEC_NUMBER, TextParser::Scope::TOKEN_PUNCT }, { "1234", "." });
	test_parsing_helper(TestCase("float"), "1234.5678", { TextParser::Scope::TOKEN_FLOAT, }, { "1234.5678" });
	test_parsing_helper(TestCase("negative-float"), "-1234.5678", { TextParser::Scope::TOKEN_FLOAT, }, { "-1234.5678" });
	test_parsing_helper(TestCase("float-fail"), "1234.-5678", { TextParser::Scope::TOKEN_DEC_NUMBER, TextParser::Scope::TOKEN_PUNCT, TextParser::Scope::TOKEN_DEC_NUMBER }, { "1234", ".", "-5678" });
	test_parsing_helper(TestCase("number_punct"), "1234.", { TextParser::Scope::TOKEN_DEC_NUMBER, TextParser::Scope::TOKEN_PUNCT }, { "1234", "." });
	test_parsing_helper(TestCase("ident"), "a1234", { TextParser::Scope::TOKEN_IDENT, }, { "a1234" });
	test_parsing_helper(TestCase("ident2"), "_a1234", { TextParser::Scope::TOKEN_IDENT, }, { "_a1234" });
	test_parsing_helper(TestCase("ident3"), ".a1234", { TextParser::Scope::TOKEN_IDENT, }, { ".a1234" } );
	test_parsing_helper(TestCase("ident4"), ".a_1234", { TextParser::Scope::TOKEN_IDENT, }, { ".a_1234" });

	test_parsing_helper(TestCase("ident_x4"), "cvta.to.global.u64",
		{ TextParser::Scope::TOKEN_IDENT, TextParser::Scope::TOKEN_IDENT, TextParser::Scope::TOKEN_IDENT,  TextParser::Scope::TOKEN_IDENT, },
		{ "cvta", ".to", ".global", ".u64" });
	test_parsing_helper(TestCase("ident-ws-ident"), "a1234  \t\r\nblabla;\r\n",
		{ TextParser::Scope::TOKEN_IDENT, TextParser::Scope::TOKEN_WHITESPACE, TextParser::Scope::TOKEN_NEWLINE, TextParser::Scope::TOKEN_IDENT, TextParser::Scope::TOKEN_PUNCT, TextParser::Scope::TOKEN_WHITESPACE, TextParser::Scope::TOKEN_NEWLINE },
		{ "a1234", "  \t\r", "\n", "blabla", ";", "\r", "\n" });
	test_parsing_helper(TestCase("register"), "%a_1234",
		{ TextParser::Scope::TOKEN_PUNCT, TextParser::Scope::TOKEN_IDENT, },
		{ "%", "a_1234" });

	test_parsing_helper(TestCase("ptx1"), "	ld.param.u64 %rd1, [_Z6tstfunPj_param_0];\n",
		{ TextParser::Scope::TOKEN_WHITESPACE, TextParser::Scope::TOKEN_IDENT, TextParser::Scope::TOKEN_IDENT, TextParser::Scope::TOKEN_IDENT, TextParser::Scope::TOKEN_WHITESPACE, TextParser::Scope::TOKEN_PUNCT, TextParser::Scope::TOKEN_IDENT, TextParser::Scope::TOKEN_PUNCT,
		TextParser::Scope::TOKEN_WHITESPACE, TextParser::Scope::TOKEN_PUNCT, TextParser::Scope::TOKEN_IDENT, TextParser::Scope::TOKEN_PUNCT, TextParser::Scope::TOKEN_PUNCT, TextParser::Scope::TOKEN_NEWLINE },
		{ "\t", "ld", ".param", ".u64", " ", "%", "rd1" ,"," , " ", "[", "_Z6tstfunPj_param_0", "]", ";", "\n" });
}

void test_scopes()
{
	const char* lines = "	mov.u32 %r1, %ntid.x;\n"
	                 	"	mov.u32 %r2, %ctaid.x;\n";

	TextParser parser(lines);
	{
		std::string tok;
		TextParser::Scope subscope(parser.root());
		subscope.token(&tok);
		subscope.token(&tok);
		subscope.token(&tok);
	}

	{
		std::string tok;
		TextParser::Scope subscope1(parser.root());
		subscope1.token(&tok);
		subscope1.token(&tok);
		subscope1.token(&tok);
		subscope1.token(&tok);
		auto result = subscope1.apply();
		TestCase("scopes1").test(result.compare("	mov.u32 ") == 0);
	}
	
	{
		std::string tok;
		TextParser::Scope subscope2(parser.root());
		subscope2.token(&tok);
		subscope2.token(&tok);
		auto result = subscope2.apply();
		TestCase("scopes2").test(result.compare("%r1") == 0);
	}

	{
		std::string tok;
		auto result = parser.root().apply();
		TestCase("scopes3").test(result.compare("	mov.u32 %r1") == 0);
	}
}

void main()
{
	test_parsing();
	test_scopes();

	TestCase::print_stats();
}