SM=52
.PHONY: all clean run cudarace
OPT_LEVEL=-O3

all:
	nvcc $(OPT_LEVEL) -D_FORCE_INLINES -arch=sm_$(SM) --cudart=shared hotspot.cu -o hotspot
	nvcc $(OPT_LEVEL) -lineinfo -D_FORCE_INLINES -arch=sm_$(SM) --cudart=shared hotspot.cu -o hotspot_linfo
	cuobjdump -ptx hotspot > hotspot.ptx

clean:
	rm -f result.txt hotspot

run:
	bash ./run

cudarace:
	bash ./cudarace

