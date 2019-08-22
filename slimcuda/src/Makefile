SM=35
TARGETS=libhook.so \
		test_ptx_instrumentation2 test_ptx_instrumentation test_ptx_parser test_text_parser test_ptx_basic_block \
        tst1_basic_works tst1_pre_instrumented tst2 tst3 tst4 tst5  tst5a tst6
RAW_PTX_STUB=ptx_stub.raw.o
LIB_HOOK_O=libhook.o

ifneq ($(DEBUG),)
#EXTRA_CFLAGS += -DSLIM_DEBUG
EXTRA_CFLAGS += -DSLIM_VERBOSE -DSLIM_DEBUG 
EXTRA_NVCC_CFLAGS += -DSLIM_DEBUG 
endif

ifneq ($(ASAN),)
EXTRA_CFLAGS += "-fsanitize=address"
EXTRA_LDFLAGS += "-fsanitize=address -shared-libasan"
endif

BASE=/opt/nvidia$(NVIDIA_SDK_VERSION)/cuda
SAMPLES_BASE=$(BASE)/samples
TBB=/opt/tbb
TBB_LIB=$(TBB)/lib/intel64/gcc4.4/
CC=clang++
CFLAGS=-I$(BASE)/include -I$(TBB)/include -Wall -g -std=c++14 -D_FORCE_INLINES $(EXTRA_CFLAGS) #-fsanitize=thread
LDFLAGS=-lz -llz4 -g -Wall -lpthread -ldl  -fPIC -L$(TBB_LIB) -ltbb -rdynamic #-fsanitize=thread #-Wl,--no-undefined 
NVCC=$(BASE)/bin/nvcc
NVCFLAGS=-I$(SAMPLES_BASE)/common/inc --cudart=shared -arch=sm_$(SM)  -g $(EXTRA_NVCC_CFLAGS)
STUBNVCFLAGS=-O3 -I$(SAMPLES_BASE)/common/inc --cudart=shared -arch=sm_$(SM) -D_FORCE_INLINES  $(EXTRA_NVCC_CFLAGS)


all: $(TARGETS)

tst1_basic_works: tst1_basic_works.cu
	$(NVCC) $(NVCFLAGS)  $< -o $@
	objdump -M intel -d $@ | c++filt > $@.s
	cuobjdump -ptx $@ | sed -n '/^.version.*/,$$p' > $@.ptx

tst1_pre_instrumented: tst1_pre_instrumented.cu
	$(NVCC) -g $(NVCFLAGS)  $< -o $@
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

tst5: tst5.cu
	$(NVCC) $(NVCFLAGS)  $< -o $@
	objdump -M intel -d $@ | c++filt > $@.s
	cuobjdump -ptx $@ | sed -n '/^.version.*/,$$p' > $@.ptx

tst5a: tst5a.cu
	$(NVCC) $(NVCFLAGS)  $< -o $@
	objdump -M intel -d $@ | c++filt > $@.s
	cuobjdump -ptx $@ | sed -n '/^.version.*/,$$p' > $@.ptx

tst6: tst6.cu
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

test_ptx_instrumentation2: test_ptx_instrumentation2.cpp $(RAW_PTX_STUB)  *.hpp
	$(CC) -I$(SAMPLES_BASE)/common/inc $(CFLAGS) $(RAW_PTX_STUB) $<  -o $@ -lcuda -L/opt/nvidia/cuda/lib64/ -lcudart
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

