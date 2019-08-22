.PHONY: all clean run cudarace


all:
	nvcc -O2 -D_FORCE_INLINES -arch=sm_35 --cudart=shared main.cu -o dwt2d
	nvcc -lineinfo -D_FORCE_INLINES -arch=sm_35 --cudart=shared main.cu -o dwt2d_linfo
	cuobjdump -ptx dwt2d > dwt2d.ptx

clean:
	rm -f result.txt dwt2d

run:
	bash ./run

cudarace:
	bash ./cudarace

