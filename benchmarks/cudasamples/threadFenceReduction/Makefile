.PHONY: all clean run cudarace

all:
	nvcc -O -arch sm_35 -I../../common/inc -cudart=shared threadFenceReduction.cu -o threadFenceReduction
	nvcc -lineinfo -O -arch sm_35 -I../../common/inc -cudart=shared threadFenceReduction.cu -o threadFenceReduction_linfo
	cuobjdump -ptx threadFenceReduction > threadFenceReduction.ptx

clean:
	rm -f threadFenceReduction

run:
	bash ./run

cudarace:
	bash ./cudarace

