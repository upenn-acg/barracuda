.PHONY: all clean run cudarace


all:
	nvcc -O2 -D_FORCE_INLINES -arch=sm_35 --cudart=shared pathfinder.cu -o pathfinder
	nvcc -O2 -lineinfo -D_FORCE_INLINES -arch=sm_35 --cudart=shared pathfinder.cu -o pathfinder_linfo
	cuobjdump -ptx pathfinder > pathfinder.ptx

clean:
	rm -f result.txt pathfinder

run:
	bash ./run

cudarace:
	bash ./cudarace

