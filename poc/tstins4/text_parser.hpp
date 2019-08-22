#pragma once

#include <string>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <functional>

class TextParser
{
public:
	TextParser(const std::string& text) : _text(text), _root(*this)
	{
	}

	class Scope
	{
	public:
		Scope(Scope& parent) : 
			  _parser(parent._parser)
			, _index(parent._index)
			, _start(parent._index)
			, _end(_parser.text().size())
			, _parent(&parent)
			, _child(NULL)
			, _str(_parser.text().c_str())
			, _skip_ws(parent._skip_ws)
		{
			parent._child = this;
		}

		~Scope()
		{
			cancel();
		}

	public: // high level
		std::string apply()
		{
			if (_parent != NULL)
				_parent->_index = _index;

			cancel();
			return text();
		}

		bool close_and_apply_if(bool pred)
		{
			if (pred)
				apply();
			else
				cancel();
			return pred;
		}

		void cancel()
		{
			assert(_child == NULL);
			if (_parent != NULL)
			{
				assert(_parent->_child == this);
				_parent->_child = NULL;
				_parent = NULL;
			}
		}

		std::string text()
		{
			return std::string(_parser.text(), _start, _index - _start);
		}

		enum TokenType { TOKEN_INVALID, TOKEN_WHITESPACE, TOKEN_NEWLINE, TOKEN_PUNCT, TOKEN_IDENT, TOKEN_HEX_NUMBER, TOKEN_DEC_NUMBER, TOKEN_FLOAT, TOKEN_FLOAT_BINARY, TOKEN_EOS};

	public:
		bool many_of(std::initializer_list<std::function<bool(Scope&)>> parsers)
		{
			bool any = false;
			while (one_of(parsers))
				any = true;
			return any;
		}

		bool one_of(std::initializer_list<std::function<bool(Scope&)>> parsers)
		{
			for (auto parser : parsers)
			{
				Scope subscope(*this);
				if (parser(subscope))
				{
					subscope.apply();
					return true;
				}
			}
			return false;
		}

		enum ParseResult { ResultCont, ResultFail, ResultEndInclude, ResultEndExclude };

		bool sequence_of(std::function<ParseResult(TokenType type, std::string& token)> filter)
		{
			for (;;)
			{
				Scope subscope(*this);

				std::string tok;
				auto type = subscope.token(&tok);

				switch (type)
				{
				case TOKEN_EOS:
				case TOKEN_INVALID:
					return false;
				}
				switch (filter(type, tok))
				{
				case ResultCont: break;
				case ResultFail: return false;
				case ResultEndInclude: subscope.apply(); return true;
				case ResultEndExclude: return true;
				}
				subscope.apply();
			}
		}

		bool expect(TokenType type, const std::string& value)
		{
			std::string val;
			return type == token(&val) && val.compare(value) == 0;
		}

		bool optional_expect(TokenType type, std::function<bool(const std::string& value)> func)
		{
			std::string val;
			Scope scope(*this);
			if(scope.expect(type, func))
				scope.apply();
			return true;
		}

		bool expect(TokenType type, std::function<bool(const std::string& value)> func)
		{
			std::string val;
			return type == token(&val) && func(val);
		}

		void set_skip_ws(bool enable)
		{
			_skip_ws = enable;
		}

	public: // low-level
		TokenType token(std::string* out)
		{
			for (;;)
			{
				if (_index >= (int)_parser._text.size())
					return TOKEN_EOS;

				int prev = _index;
				TokenType token = TOKEN_INVALID;
				if (!(whitespace(&token)
					|| newline(&token)
					|| ident(&token)
					|| number(&token)
					|| punct(&token)
					))
				{
					return TOKEN_INVALID;
				}

				if (!(_skip_ws && (token == TOKEN_WHITESPACE || token == TOKEN_NEWLINE)))
				{
					*out = _parser.text().substr(prev, _index - prev);
					return token;
				}
			}
		}

		bool skip_whitespace()
		{
			bool found = false;
			for (;;)
			{
				Scope s(*this);
				if (_index >= (int)_parser._text.size())
					break;

				int prev = _index;
				TokenType token = TOKEN_INVALID;
				if (!(s.whitespace(&token)
					|| s.newline(&token)
					))
					break;

				s.apply();
				found = true;
			}
			return found;
		}

	private:
		Scope(TextParser& parser) :
			_parser(parser), _child(NULL), _parent(NULL), _index(0), _start(0), _end(_parser.text().size()), _str(parser.text().c_str())
		{
		}

	private:
		bool whitespace(TokenType* type)
		{
			Index index(this);
			for (;;)
			{
				char c = nc();
				if (!(c == ' ' || c == '\t' || c == '\r'))
					break;
			}
			return index.try_apply(type, TOKEN_WHITESPACE);
		}

		bool newline(TokenType* type)
		{
			Index index(this);
			for (;;)
			{
				char c = nc();
				if (!(c == '\n'))
					break;
			}
			return index.try_apply(type, TOKEN_NEWLINE);
		}

		bool punct(TokenType* type)
		{
			Index index(this);
			char c = nc();
			switch(c)
			{
			case '~': case '!': case '#': case '$': case '%': case '^': case '&': case '*': case '=': case '_': case '-': case '+': case '/': case '@': 
			case '(': case ')': case '[': case ']': case '{': case '}': case '<': case '>': case '?': case ';': case ':': case '.': case ',':
				return index.try_apply(type, TOKEN_PUNCT, 1, 0);
			}
			return false;
		}

		bool ident(TokenType* type)
		{
			Index index(this);
			char c = nc();
			int min_expected = 1;
			if (c == '.')
			{
				c = nc();
				min_expected += 1;
			}
			bool nonfirst = false;
			for (;;)
			{
				if (!((c == '_') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (nonfirst && (c == '$' /* CUDA PTX */ || (c >= '0' && c <= '9')))))
					break;
				nonfirst = true;
				c = nc();
			}
			return index.try_apply(type, TOKEN_IDENT, min_expected);
		}

		bool number(TokenType* type)
		{
			return float_number(type) || hex_number(type) || float_binary(type) || dec_number(type);
		}

		bool float_binary(TokenType* type)
		{
			Index index(this);
			if (!(lookahead("0f") || lookahead("0F") || lookahead("0d") || lookahead("0D")))
				return false;

			char c;
			for (;;)
			{
				c = nc();
				if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')))
					break;
			}

			return index.try_apply(type, TOKEN_FLOAT_BINARY, 3);
		}

		bool hex_number(TokenType* type)
		{
			Index index(this);
			if (!lookahead("0x"))
				return false;
			char c;
			for (;;)
			{
				c = nc();
				if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')))
					break;
			}

			return index.try_apply(type, TOKEN_HEX_NUMBER, 3);
		}

		bool dec_number(TokenType* type, bool allow_negative = true)
		{
			Index index(this);
			int first = true;
			char c = nc();
			int min_expected = 1;
			if (allow_negative && c == '-')
			{
				min_expected += 1;
				c = nc();
			}
			for (;;)
			{
				if (!(c >= '0' && c <= '9'))
					break;
				c = nc();
			}

			return index.try_apply(type, TOKEN_DEC_NUMBER, min_expected);
		}

		bool float_number(TokenType* type)
		{
			Index index(this);
			if (!(dec_number(type) && nc() == '.' && dec_number(type, false)))
				return false;
			index.try_apply(type, TOKEN_FLOAT, 3, 0);

			return true;
		}

		bool lookahead(const char* what)
		{
			int len = strlen(what);
			if (std::strncmp(_str + _index, what, len) != 0)
				return false;
			_index += len;
			return true;
		}

		char nc()
		{
			if (_index > _end)
				return 0;
			return _str[_index++];
		}

	private:
		class Index
		{
		public:
			Index(Scope* scope) : _scope(scope), _orig(scope->_index)
			{

			}

			~Index()
			{
				if (_scope != NULL)
					_scope->_index = _orig;
			}

			bool try_apply(TokenType* type, TokenType thetype, int min_expected = 1, int backtrack = 1)
			{
				if (_scope->_index - _orig < (min_expected + backtrack))
					return false;
				_scope->_index -= backtrack;
				_scope = NULL;
				*type = thetype;
				return true;
			}

		private:
			Scope* _scope;
			int _orig;
		};

	private:
		friend class TextParser;
		TextParser& _parser;
		const char* _str;
		int _start;
		int _index;
		int _end;
		Scope* _child;
		Scope* _parent;
		bool _skip_ws;
	};

	Scope& root()
	{
		return _root;
	}

	bool eos() const
	{
		return _root._index == _text.size();
	}

public:
	const std::string& text() const
	{
		return _text;
	}

private:
	std::string _text;
	Scope _root;
};


