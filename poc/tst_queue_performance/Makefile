TARGETS=tstasyncpc libhook.so
BASE=/opt/nvidia/cuda
SAMPLES_BASE=/data/cuda/samples/
CC=g++
CFLAGS=-I$(BASE)/include -Wall -Werror -g  -D_FORCE_INLINES
NVCC=$(BASE)/bin/nvcc
#NVCFLAGS=-g -O2 -I$(SAMPLES_BASE)/common/inc -lpthread --cudart=shared -arch=sm_30
NVCFLAGS=-g -I$(SAMPLES_BASE)/common/inc -lpthread --cudart=shared -arch=sm_30 -D_FORCE_INLINES
#NVCFLAGS=-g -I$(SAMPLES_BASE)/common/inc -lpthread --cudart=shared -arch=sm_30

all: $(TARGETS)

tstasyncpc tstasyncpcdbg: tstasyncpc.cu libhook.so consumer.hpp protocol.hpp slimfast.hpp impl.hpp devlogger.hpp 
	$(NVCC) -O2 $(NVCFLAGS) -L./ -l hook -Xlinker "-rpath ." $< -o $@
	$(NVCC) -G $(NVCFLAGS) -L./ -l hookdbg -Xlinker "-rpath ." $< -o tstasyncpcdbg
	#$(NVCC) -G -DDEBUG $(NVCFLAGS) -L./ -l hookdbg -Xlinker "-rpath ." $< -o tstasyncpcdbg
	objdump -M intel -d $@ | c++filt > $@.s

libhook.so libhookdbg.so: libhook.cpp libhookinfra.hpp hooklib.h
	$(CC) -O2 $(CFLAGS) -fPIC -shared $< -o $@
	$(CC) -DDEBUG $(CFLAGS) -fPIC -shared $< -o libhookdbg.so


test: $(TARGETS)
	./tstasyncpc

clean:
	\rm -f $(TARGETS) tstasyncpc.s libhookdbg.so tstasyncpcdbg


