TARGETS=tstshadow
BASE=/opt/nvidia/cuda
SAMPLES_BASE=/data/cuda/samples/
CC=g++
CFLAGS=-I$(BASE)/include -Wall -Werror -g 
NVCC=$(BASE)/bin/nvcc
NVCFLAGS=-g -O2 -I$(SAMPLES_BASE)/common/inc --cudart=shared

all: $(TARGETS)

tstshadow: tstshadow.cu
	$(NVCC) $(NVCFLAGS) $^ -o $@
	objdump -M intel -d $@ | c++filt > $@.s

test: $(TARGETS)
	./tstshadow

clean:
	\rm -f $(TARGETS) 


