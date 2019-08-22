.PHONY: all clean run cudarace


all:
	nvcc  -O2 -D_FORCE_INLINES -arch=sm_35 --cudart=shared srad.cu -o srad
	nvcc -lineinfo -D_FORCE_INLINES -arch=sm_35 --cudart=shared srad.cu -o srad_linfo
	cuobjdump -ptx srad > srad.ptx

clean:
	rm -f result.txt srad

run:
	bash ./run

cudarace:
	bash ./cudarace

