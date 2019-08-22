.PHONY: all clean run cudarace


all:
	nvcc -O -arch sm_35 -I ../../common/inc/ -cudart=shared interac.cu ../../common/lib/libcutil_x86_64.a -o interac
	nvcc -lineinfo -O -arch sm_35 -I ../../common/inc/ -cudart=shared interac.cu ../../common/lib/libcutil_x86_64.a -o interac_linfo
	cuobjdump -ptx interac > interac.ptx

clean:
	rm -f interac

run:
	bash ./run

cudarace:
	bash ./cudarace

