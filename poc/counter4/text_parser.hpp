#pragma once

#include <string>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <functional>
#include <algorithm>

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
			, _str(_parser.text().c_str())
			, _start(parent._index)
			, _index(parent._index)
			, _end(_parser.text().size())
			, _child(nullptr)
			, _parent(&parent)
			, _skip_ws(parent._skip_ws)
			, _line(1)
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
			if (_parent != nullptr)
			{
				_parent->_index = _index;
				_parent->_line = _line;
			}

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
			assert(_child == nullptr);
			if (_parent != nullptr)
			{
				assert(_parent->_child == this);
				_parent->_child = nullptr;
				_parent = nullptr;
			}
		}

		std::string text()
		{
			return std::string(_parser.text(), _start, _index - _start);
		}

		enum class TokenType { Invalid, Whitespace, Newline, Punctuation, Identifier, HexNumber, DecNumber, Float, FloatBinary, EOS};

		int line() const { return _line; }
		std::string next_text() const
		{
			auto pos = _parser.text().find_first_of('\n');
			return _parser.text().substr(_index, pos);
		}
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

		enum class ParseResult { Continue, Fail, SuccessInclusive, SuccessExclusive };

		bool sequence_of(std::function<ParseResult(TokenType type, std::string& token)> filter)
		{
			for (;;)
			{
				Scope subscope(*this);

				std::string tok;
				auto type = subscope.token(&tok);

				switch (type)
				{
				case TokenType::EOS:
				case TokenType::Invalid:
					return false;
                default:
                    break;
				}
				switch (filter(type, tok))
				{
				case ParseResult::Continue: break;
				case ParseResult::Fail: return false;
				case ParseResult::SuccessInclusive: subscope.apply(); return true;
				case ParseResult::SuccessExclusive: return true;
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
					return TokenType::EOS;

				int prev = _index;
				TokenType token = TokenType::Invalid;
				if (!(whitespace(&token)
					|| newline(&token)
					|| ident(&token)
					|| number(&token)
					|| punct(&token)
					))
				{
					return TokenType::Invalid;
				}

				if (!(_skip_ws && (token == TokenType::Whitespace || token == TokenType::Newline)))
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

				TokenType token = TokenType::Invalid;
				if (!(s.whitespace(&token)
					|| s.newline(&token)
					))
					break;

				s.apply();
				found = true;
			}
			return found;
		}

		std::string raw(unsigned int numchars)
		{
			numchars = std::min(numchars, (unsigned int)(_parser._text.size() - _index));
			std::string result{ _str + _index, numchars };
			_index += numchars;
			return result;
		}

	private:
		Scope(TextParser& parser) :
			_parser(parser)
            , _str(parser.text().c_str())
            , _start(0), _index(0), _end(_parser.text().size())
            , _child(nullptr)
            , _parent(nullptr)
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
			return index.try_apply(type, TokenType::Whitespace);
		}

		bool newline(TokenType* type)
		{
			Index index(this);
			for (;;)
			{
				char c = nc();
				if (!(c == '\n'))
					break;
				++_line;
			}
			return index.try_apply(type, TokenType::Newline);
		}

		bool punct(TokenType* type)
		{
			Index index(this);
			char c = nc();
			switch(c)
			{
			case '~': case '!': case '#': case '$': case '%': case '^': case '&': case '*': case '=': case '_': case '-': case '+': case '/': case '@': 
			case '(': case ')': case '[': case ']': case '{': case '}': case '<': case '>': case '?': case ';': case ':': case '.': case ',': case '"': case '\'':
            case '|':
				return index.try_apply(type, TokenType::Punctuation, 1, 0);
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
			return index.try_apply(type, TokenType::Identifier, min_expected);
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

			return index.try_apply(type, TokenType::FloatBinary, 3);
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

			return index.try_apply(type, TokenType::HexNumber, 3);
		}

		bool dec_number(TokenType* type, bool allow_negative = true)
		{
			Index index(this);
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

			return index.try_apply(type, TokenType::DecNumber, min_expected);
		}

		bool float_number(TokenType* type)
		{
			Index index(this);
			if (!(dec_number(type) && nc() == '.' && dec_number(type, false)))
				return false;
			index.try_apply(type, TokenType::Float, 3, 0);

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
				if (_scope != nullptr)
					_scope->_index = _orig;
			}

			bool try_apply(TokenType* type, TokenType thetype, int min_expected = 1, int backtrack = 1)
			{
				if (_scope->_index - _orig < (min_expected + backtrack))
					return false;
				_scope->_index -= backtrack;
				_scope = nullptr;
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
		int _line;
	};

	Scope& root()
	{
		return _root;
	}

	bool eos() const
	{
		return _root._index == (int)_text.size();
	}

	int left() const
	{
		return _text.size() - _root._index;
	}

	int line() const
	{
		return _root._line;
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


