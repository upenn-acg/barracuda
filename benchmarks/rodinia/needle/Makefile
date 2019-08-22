.PHONY: all clean run cudarace

all:
	nvcc -O2 -D_FORCE_INLINES -arch=sm_35 --cudart=shared needle.cu -o needle
	nvcc -O2 -lineinfo -D_FORCE_INLINES -arch=sm_35 --cudart=shared needle.cu -o needle_linfo
	cuobjdump -ptx needle > needle.ptx

clean:
	rm -f result.txt needle

run:
	bash ./run

cudarace:
	bash ./cudarace

