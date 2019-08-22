.PHONY: all clean run cudarace


FILES=main.cu Graph.cpp Timer.cpp OptionParser.cpp ResultDatabase.cpp Option.cpp 

all:
	nvcc -arch sm_35 -o shoc_bfs $(FILES) --cudart=shared
	nvcc -arch sm_35 -o shoc_bfs_linfo $(FILES) -lineinfo --cudart=shared
	cuobjdump -ptx shoc_bfs_linfo > shoc_bfs.ptx

clean:
	rm -f result.txt shoc_bfs

run:
	bash ./run

cudarace:
	bash ./cudarace

