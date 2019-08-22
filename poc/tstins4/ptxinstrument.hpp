#pragma once

#include <string>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

class PTXInstrumenter
{
public:
    static std::string instrument(std::string ptx)
    {
        static const char* FILENAME = "tst.ptx";

        FILE* tmp = fopen(FILENAME, "w");
        if(tmp == NULL)
        {
            fprintf(stderr, "Failed opening %s\n", FILENAME);
            exit(1);
        }

        fwrite(ptx.c_str(), ptx.size(), 1, tmp);
        fclose(tmp);

        fprintf(stdout, "Wrote tst.ptx, press any key to reload.\n");
        getc(stdin);

        tmp = fopen(FILENAME, "r");
        if(tmp == NULL)
        {
            fprintf(stderr, "Failed opening %s\n", FILENAME);
            exit(1);
        }
        struct stat b;      
        if(stat(FILENAME, &b) != 0)
        {
            fprintf(stderr, "Failed stat on %s\n", FILENAME);
            exit(1);
        }
        char* buf = new char[b.st_size];
        fread(buf, b.st_size, 1, tmp);
        fclose(tmp);
        std::string val = std::string(buf, b.st_size);
        delete[] buf;
        return val; 
    }

};
