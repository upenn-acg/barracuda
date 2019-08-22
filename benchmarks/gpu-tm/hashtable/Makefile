
.PHONY: all clean run cudarace

all:
	nvcc -O -arch sm_35 -I ../../common/inc/ -cudart=shared hashtable.cu ../../common/lib/libcutil_x86_64.a -o hashtable
	nvcc -lineinfo -O -arch sm_35 -I ../../common/inc/ -cudart=shared hashtable.cu ../../common/lib/libcutil_x86_64.a -o hashtable_linfo
	cuobjdump -ptx hashtable > hashtable.ptx

clean:
	rm -f hashtable

run:
	bash ./run

cudarace:
	bash ./cudarace

