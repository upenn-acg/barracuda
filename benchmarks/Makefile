CUDA_SAMPLES=cudasamples/threadFenceReduction
GPU_TM_SAMPLES=gpu-tm/hashtable #gpu-tm/interac 
RODINIA_SAMPLES=rodinia/bfs rodinia/hotspot rodinia/needle rodinia/nn rodinia/pathfinder rodinia/gaussian rodinia/kmeans rodinia/streamcluster rodinia/dwt2d
CUB_SAMPLES=cub
SHOC=shoc/shocbfs
SUBDIRS=$(CUDA_SAMPLES) $(CUB_SAMPLES) $(RODINIA_SAMPLES) $(GPU_TM_SAMPLES) $(SHOC)

.PHONY: run clean all cudarace $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	make -C $@
	
run:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir run; \
	done

cudarace:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir cudarace; \
	done

clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
