TARGETS=tst1 libhook.so
BASE=/opt/nvidia/cuda
SAMPLES_BASE=$(BASE)/samples
CC=clang++
CFLAGS=-I$(BASE)/include -Wall -g -std=c++11 -lz -llz4
#CFLAGS=-I$(BASE)/include -Wall -Werror -g -std=c++11 -stdlib=libc++ -lz -llz4
NVCC=$(BASE)/bin/nvcc
#NVCFLAGS=-g -O2 -I$(SAMPLES_BASE)/common/inc -lpthread --cudart=shared -arch=sm_30
NVCFLAGS=-I$(SAMPLES_BASE)/common/inc --cudart=shared -arch=sm_30 -g  -D_FORCE_INLINES 
STUBNVCFLAGS=-O3 -I$(SAMPLES_BASE)/common/inc --cudart=shared -arch=sm_30  -D_FORCE_INLINES 

all: $(TARGETS)

tst1: tst1.cu
	$(NVCC) $(NVCFLAGS)  $< -o $@
	objdump -M intel -d $@ | c++filt > $@.s

libhook.so: libhook.cpp *.hpp ptxstubs.ptx
	$(CC) $(CFLAGS) -fPIC -shared $< -o $@

ptxstubs.ptx: ptxstubs.elf
	cuobjdump -ptx $< > $@

ptxstubs.elf: ptxstubs.cu
	$(NVCC) $(STUBNVCFLAGS)  $< -o $@

test: $(TARGETS)
	./tstasyncpc

clean:
	\rm -f $(TARGETS) tstasyncpc.s ptxstubs.ptx ptxstubs.elf

