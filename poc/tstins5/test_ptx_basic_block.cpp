#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>

#include "ptx_program.hpp"
#include "ptx_parser.hpp"
#include "ptx_instrumentation.hpp"
#include "ptx_basic_block.hpp"

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

    PtxParser parser;
    PtxParser::ParseStatus status;
    if (!parser.parse_ptx(ptx_text, &status))
    {
        std::cerr << "Could not parse PTX at byte: " << status.parsed << std::endl;
        return false;
    }

    PtxProgram& ptx = parser.ptx();
    for (auto& d : ptx)
    {
        if (d->dir_type() == PtxDirective::DirType::Function)
        {
            auto func = std::static_pointer_cast<PtxFunction>(d);
            std::cout << "====Function: " << func->name() << std::endl;
            PtxFunctionAnalyzer analyzer(func);
            for (auto& bb : analyzer)
            {
                std::cout << "---Block: " << std::endl;
                for (auto it = bb->begin(); it != bb->end(); ++ it)
                {
                    (*it)->print(std::cout);
                }
            }
        }
    }
    

    return 0;
}

