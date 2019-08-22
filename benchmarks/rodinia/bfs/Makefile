SM=52
.PHONY: all clean run cudarace


all:
	nvcc -O2 -D_FORCE_INLINES -arch=sm_$(SM) --cudart=shared bfs.cu -o bfs -g
	nvcc -lineinfo -D_FORCE_INLINES -arch=sm_$(SM) --cudart=shared bfs.cu -o bfs_linfo
	cuobjdump -ptx bfs > bfs.ptx

clean:
	rm -f result.txt bfs

run:
	bash ./run

cudarace:
	bash ./cudarace

