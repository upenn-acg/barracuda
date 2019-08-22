.PHONY: all clean run cudarace


all:
	nvcc -O2 -D_FORCE_INLINES -arch=sm_35 --cudart=shared -o kmeans cluster.c getopt.c kmeans.cpp kmeans_clustering.c kmeans_cuda.cu rmse.c
	nvcc -O2 -lineinfo -D_FORCE_INLINES -arch=sm_35 --cudart=shared -o kmeans_linfo cluster.c getopt.c kmeans.cpp kmeans_clustering.c kmeans_cuda.cu rmse.c
	cuobjdump -ptx kmeans > kmeans.ptx

clean:
	rm -f result.txt kmeans

run:
	bash ./run

cudarace:
	bash ./cudarace

