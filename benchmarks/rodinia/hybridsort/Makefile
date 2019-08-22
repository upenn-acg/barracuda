SM=35
.PHONY: all clean run cudarace


all:
	nvcc -O2 -D_FORCE_INLINES -I/opt/nvidia/cuda/samples/common/inc/ -arch=sm_$(SM) --cudart=shared main.cu -o hybridsort
	nvcc -lineinfo -D_FORCE_INLINES -I/opt/nvidia/cuda/samples/common/inc/ -arch=sm_$(SM) --cudart=shared main.cu -o hybridsort_linfo
	cuobjdump -ptx hybridsort > hybridsort.ptx

clean:
	rm -f result.txt hybridsort

run:
	bash ./run

cudarace:
	bash ./cudarace

