TARGETS=tstasyncpc libhook.so
BASE=/opt/nvidia/cuda
SAMPLES_BASE=/data/cuda/samples/
CC=g++
CFLAGS=-I$(BASE)/include -Wall -Werror -g 
NVCC=$(BASE)/bin/nvcc
#NVCFLAGS=-g -O2 -I$(SAMPLES_BASE)/common/inc -lpthread --cudart=shared -arch=sm_30
NVCFLAGS=-g -O2 -I$(SAMPLES_BASE)/common/inc -lpthread --cudart=shared -arch=sm_30
#NVCFLAGS=-g -I$(SAMPLES_BASE)/common/inc -lpthread --cudart=shared -arch=sm_30

all: $(TARGETS)

tstasyncpc tstasyncpcdbg: tstasyncpc.cu libhook.so consumer.hpp protocol.hpp slimfast.hpp impl.hpp devlogger.hpp 
	$(NVCC) $(NVCFLAGS) -L./ -l hook -Xlinker "-rpath ." $< -o $@
	$(NVCC) -DDEBUG $(NVCFLAGS) -L./ -l hookdbg -Xlinker "-rpath ." $< -o tstasyncpcdbg
	objdump -M intel -d $@ | c++filt > $@.s

libhook.so libhookdbg.so: libhook.cpp libhookinfra.hpp hooklib.h
	$(CC) $(CFLAGS) -fPIC -shared $< -o $@
	$(CC) -DDEBUG $(CFLAGS) -fPIC -shared $< -o libhookdbg.so


test: $(TARGETS)
	./tstasyncpc

clean:
	\rm -f $(TARGETS) tstasyncpc.s libhookdbg.so tstasyncpcdbg


