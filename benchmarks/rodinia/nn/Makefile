.PHONY: all clean run cudarace

all:
	nvcc -O2 -D_FORCE_INLINES -arch=sm_35 --cudart=shared nn.cu -o nn
	nvcc -O2 -lineinfo -D_FORCE_INLINES -arch=sm_35 --cudart=shared nn.cu -o nn_linfo
	cuobjdump -ptx nn > nn.ptx

clean:
	rm -f result.txt nn

run:
	bash ./run

cudarace:
	bash ./cudarace

