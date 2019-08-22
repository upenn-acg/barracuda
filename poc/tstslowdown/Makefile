TARGETS=libhook.so \
		test_ptx_instrumentation test_ptx_parser test_text_parser test_ptx_basic_block \
        tst1 tst2 tst3 tst4
RAW_PTX_STUB=ptx_stub.raw.o
LIB_HOOK_O=libhook.o

BASE=/opt/nvidia/cuda
SAMPLES_BASE=$(BASE)/samples
CC=clang++
CFLAGS=-I$(BASE)/include -Wall -g -std=c++14 -D_FORCE_INLINES
LDFLAGS=-lz -llz4 -g -Wall
#CFLAGS=-I$(BASE)/include -Wall -Werror -g -std=c++11 -stdlib=libc++ -lz -llz4
NVCC=$(BASE)/bin/nvcc
#NVCFLAGS=-g -O2 -I$(SAMPLES_BASE)/common/inc -lpthread --cudart=shared -arch=sm_30
NVCFLAGS=-I$(SAMPLES_BASE)/common/inc --cudart=shared -arch=sm_30  -g
#NVCFLAGS=-I$(SAMPLES_BASE)/common/inc --cudart=shared -arch=sm_30  -lineinfo -g
STUBNVCFLAGS=-O3 -I$(SAMPLES_BASE)/common/inc --cudart=shared -arch=sm_30 -D_FORCE_INLINES 


all: $(TARGETS)

tst1: tst1.cu
	$(NVCC) $(NVCFLAGS)  $< -o $@
	objdump -M intel -d $@ | c++filt > $@.s
	cuobjdump -ptx $@ | sed -n '/^.version.*/,$$p' > $@.ptx

tst2: tst2.cu
	$(NVCC) $(NVCFLAGS)  $< -o $@
	objdump -M intel -d $@ | c++filt > $@.s
	cuobjdump -ptx $@ | sed -n '/^.version.*/,$$p' > $@.ptx


tst3: tst3.cu
	$(NVCC) $(NVCFLAGS)  $< -o $@
	objdump -M intel -d $@ | c++filt > $@.s
	cuobjdump -ptx $@ | sed -n '/^.version.*/,$$p' > $@.ptx

tst4: tst4.cu
	$(NVCC) $(NVCFLAGS)  $< -o $@
	objdump -M intel -d $@ | c++filt > $@.s
	cuobjdump -ptx $@ | sed -n '/^.version.*/,$$p' > $@.ptx

libhook.so: $(LIB_HOOK_O) $(RAW_PTX_STUB)
	$(CC) $(LDFLAGS) -fPIC -shared $^ -o $@

$(LIB_HOOK_O): libhook.cpp *.hpp
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

ptx_stub.ptx: ptx_stub.elf
	cuobjdump -ptx $< | sed -n '/^.version.*/,$$p' > $@

ptx_stub.elf: ptx_stub.cu
	$(NVCC) $(STUBNVCFLAGS)  $< -o $@
	objdump -M intel -d $@ | c++filt > $@.s
	./ptx_stub.elf

$(RAW_PTX_STUB): ptx_stub.ptx 
	objcopy -I binary -O elf64-x86-64 --binary-architecture i386 ptx_stub.ptx $(RAW_PTX_STUB)

test_ptx_instrumentation: test_ptx_instrumentation.cpp $(RAW_PTX_STUB)  *.hpp
	$(CC) $(CFLAGS) $< $(RAW_PTX_STUB)  -o $@

test_ptx_parser: test_ptx_parser.cpp *.hpp
	$(CC) $(CFLAGS) $< -o $@

test_text_parser: test_text_parser.cpp  *.hpp
	$(CC) $(CFLAGS) $< -o $@

test_ptx_basic_block: test_ptx_basic_block.cpp  *.hpp
	$(CC) $(CFLAGS) $< -o $@

test: $(TARGETS)
	./tstasyncpc

clean:
	\rm -f $(TARGETS) tstasyncpc.s ptx_stub.ptx ptx_stub.elf $(LIB_HOOK_O) *.ptx *.o *.s

