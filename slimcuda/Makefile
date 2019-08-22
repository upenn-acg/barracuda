TARGETS=slim

.PHONY: slim

all: $(TARGETS)

slim:
	make -C src
	mkdir -p bin
	cp src/libhook.so bin/

clean:
	(cd src; make clean)
