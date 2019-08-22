#include "text_parser.hpp"

using TokenType = TextParser::Scope::TokenType;

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

	void failed() const
	{
		_status = false;
		fprintf(stderr, "Failed: %s\n", _name);
	}

	static void print_stats()
	{
		printf("Succeeded: %i/%i\n", _success, _success + _failed);
	}

    static bool has_failed()
    {
        return _failed > 0;
    }

private:
	static int _failed;
	static int _success;
	const char* _name;
	mutable bool _status;

};

int TestCase::_failed = 0;
int TestCase::_success = 0;

void test_parsing_helper(const TestCase& tc, const std::string& text, 
		std::initializer_list<TokenType> types,
		std::initializer_list<std::string> values)
{
	TextParser parser(text);
	TextParser::Scope& root = parser.root();

	std::string tok;
	auto types_it = types.begin();
	auto values_it = values.begin();
	for(;types_it != types.end() && values_it != values.end(); ++ types_it, ++ values_it)
	{
		TokenType toktype;

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

	test_parsing_helper(TestCase("number"), "1234", { TokenType::DecNumber, }, { "1234" });
	test_parsing_helper(TestCase("negative-number"), "-1234", { TokenType::DecNumber, }, { "-1234" });
	test_parsing_helper(TestCase("hexnumber"), "0x1234", { TokenType::HexNumber, }, { "0x1234" });
	test_parsing_helper(TestCase("number_punct"), "1234.", { TokenType::DecNumber, TokenType::Punctuation }, { "1234", "." });
	test_parsing_helper(TestCase("float"), "1234.5678", { TokenType::Float, }, { "1234.5678" });
	test_parsing_helper(TestCase("negative-float"), "-1234.5678", { TokenType::Float, }, { "-1234.5678" });
	test_parsing_helper(TestCase("float-fail"), "1234.-5678", { TokenType::DecNumber, TokenType::Punctuation, TokenType::DecNumber }, { "1234", ".", "-5678" });
	test_parsing_helper(TestCase("number_punct"), "1234.", { TokenType::DecNumber, TokenType::Punctuation }, { "1234", "." });
	test_parsing_helper(TestCase("ident"), "a1234", { TokenType::Identifier, }, { "a1234" });
	test_parsing_helper(TestCase("ident2"), "_a1234", { TokenType::Identifier, }, { "_a1234" });
	test_parsing_helper(TestCase("ident3"), ".a1234", { TokenType::Identifier, }, { ".a1234" } );
	test_parsing_helper(TestCase("ident4"), ".a_1234", { TokenType::Identifier, }, { ".a_1234" });

	test_parsing_helper(TestCase("ident_x4"), "cvta.to.global.u64",
		{ TokenType::Identifier, TokenType::Identifier, TokenType::Identifier,  TokenType::Identifier, },
		{ "cvta", ".to", ".global", ".u64" });
	test_parsing_helper(TestCase("ident-ws-ident"), "a1234  \t\r\nblabla;\r\n",
		{ TokenType::Identifier, TokenType::Whitespace, TokenType::Newline, TokenType::Identifier, TokenType::Punctuation, TokenType::Whitespace, TokenType::Newline },
		{ "a1234", "  \t\r", "\n", "blabla", ";", "\r", "\n" });
	test_parsing_helper(TestCase("register"), "%a_1234",
		{ TokenType::Punctuation, TokenType::Identifier, },
		{ "%", "a_1234" });

	test_parsing_helper(TestCase("ptx1"), "	ld.param.u64 %rd1, [_Z6tstfunPj_param_0];\n",
		{ TokenType::Whitespace, TokenType::Identifier, TokenType::Identifier, TokenType::Identifier, TokenType::Whitespace, TokenType::Punctuation, TokenType::Identifier, TokenType::Punctuation,
		TokenType::Whitespace, TokenType::Punctuation, TokenType::Identifier, TokenType::Punctuation, TokenType::Punctuation, TokenType::Newline },
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

int main()
{
	test_parsing();
	test_scopes();

	TestCase::print_stats();
    return TestCase::has_failed() ? 1 : 0;
}
