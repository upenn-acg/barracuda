TARGETS=tstasyncpc libhook.so
BASE=/opt/nvidia/cuda
SAMPLES_BASE=/data/cuda/samples/
CC=g++
CFLAGS=-I$(BASE)/include -Wall -Werror -g 
NVCC=$(BASE)/bin/nvcc
#NVCFLAGS=-g -O2 -I$(SAMPLES_BASE)/common/inc -lpthread --cudart=shared -arch=sm_30
NVCFLAGS=-g -I$(SAMPLES_BASE)/common/inc -lpthread --cudart=shared -arch=sm_30

all: $(TARGETS)

tstasyncpc: tstasyncpc.cu libhook.so consumer.hpp protocol.hpp slimfast.hpp impl.hpp devlogger.hpp CV.hpp threadblocks.hpp threadmap.hpp streammap.hpp
	$(NVCC) $(NVCFLAGS) -L./ -l hook -Xlinker "-rpath ." $< -o $@
	objdump -M intel -d $@ | c++filt > $@.s

libhook.so: libhook.cpp libhookinfra.hpp hooklib.h
	$(CC) $(CFLAGS) -fPIC -shared $< -o $@

test: $(TARGETS)
	./tstasyncpc

clean:
	\rm -f $(TARGETS) tstasyncpc.s


