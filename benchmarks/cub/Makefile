.PHONY: all run cudarace clean

BENCHMARKS=example_block_radix_sort example_block_reduce example_block_scan example_device_partition_flagged example_device_partition_if example_device_radix_sort example_device_reduce example_device_scan example_device_select_flagged example_device_select_if example_device_select_unique example_device_sort_find_non_trivial_runs
BENCHMARKS_DONT_COMPILE=example_device_histogram 

CCFLAGS=-DSM350 -D_FORCE_INLINES -Xcompiler -ffloat-store -m64 -I./ -O3 -cudart=shared -gencode=arch=compute_35,code=\"sm_35,compute_35\"   -g

all: $(BENCHMARKS)

$(BENCHMARKS): *.cu
	nvcc $(CCFLAGS) -o $@ $@.cu
	nvcc $(CCFLAGS) -lineinfo -o $@_linfo $@.cu
	cuobjdump -ptx $@_linfo > $@.ptx 

run:
	bash ./run
	
cudarace:
	bash ./cudarace
	
clean:
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o block_radix_sort block_reduce block_scan $(BENCHMARKS)

