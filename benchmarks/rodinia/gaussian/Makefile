
.PHONY: all clean run cudarace

all:
	nvcc -O2 -D_FORCE_INLINES -arch=sm_35 --cudart=shared gaussian.cu -o gaussian
	nvcc -O2 -lineinfo -D_FORCE_INLINES -arch=sm_35 --cudart=shared gaussian.cu -o gaussian_linfo
	cuobjdump -ptx gaussian > gaussian.ptx

clean:
	rm -f result.txt gaussian

run:
	bash ./run

cudarace:
	bash ./cudarace
