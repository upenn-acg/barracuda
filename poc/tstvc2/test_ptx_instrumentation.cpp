#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>

#include "ptx_program.hpp"
#include "ptx_parser.hpp"
#include "ptx_instrumentation.hpp"

std::string read_file(const std::string& filename)
{
	std::ifstream stream(filename);
	std::string str;

	if (!stream)
	{
		std::cerr << "Could not read: " << filename << std::endl;
		exit(1);
	}

	stream.seekg(0, std::ios::end);
	str.reserve((unsigned int)stream.tellg());
	stream.seekg(0, std::ios::beg);

	str.assign((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
	return str;
}

int main(int argc, char* argv[])
{
	if(argc != 2)
	{
		fprintf(stderr, "Syntax: %s file.ptx\n", argv[0]);
		return 1;
	}
	std::string ptx_text = read_file(argv[1]);

	PtxInstrumentation instrumentation;
	std::string instrumented;
	if (!instrumentation.instrument(ptx_text, &instrumented))
	{
		std::cerr << "Error instrumenting PTX" << std::endl;
	}

	std::cout << instrumented;
	return 0;
}

