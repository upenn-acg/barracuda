#include <stdio.h>

int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        fprintf(stderr, "%s file.ptx\n", argv[0]);
        return 1;
    }

    int f = open(argv[1], _O_READ);
    void* m = mmap(ar
