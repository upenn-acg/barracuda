TARGETS=axpy libhook.so
BASE=/opt/nvidia/cuda
SAMPLES_BASE=/data/cuda/samples/
CC=g++
CFLAGS=-I$(BASE)/include -Wall -Werror -g 
NVCC=$(BASE)/bin/nvcc
NVCFLAGS=-g -O2 -I$(SAMPLES_BASE)/common/inc --cudart=shared

all: $(TARGETS)

axpy: axpy.cu libhook.so
	$(NVCC) $(NVCFLAGS) -L./ -l hook -Xlinker "-rpath ." $^ -o $@
	objdump -M intel -d $@ | c++filt > $@.s

libhook.so: libhook.c
	$(CC) $(CFLAGS) -fPIC -shared $^ -o $@

test: $(TARGETS)
	./axpy

clean:
	\rm -f $(TARGETS) axpy.s


