.PHONY: all clean run cudarace

all:
	nvcc -O2 -D_FORCE_INLINES -arch=sm_35 --cudart=shared streamcluster_cuda_cpu.cu -o streamcluster
	nvcc -O2 -lineinfo -D_FORCE_INLINES -arch=sm_35 --cudart=shared streamcluster_cuda_cpu.cu -o streamcluster_linfo
	cuobjdump -ptx streamcluster > streamcluster.ptx

clean:
	rm -f result.txt streamcluster

run:
	bash ./run

cudarace:
	bash ./cudarace

