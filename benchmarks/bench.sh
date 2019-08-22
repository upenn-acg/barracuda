#!/bin/sh
export HOOKLIB=/home/arieleiz/gpu-race-detection-ae/slimcuda/src/libhook.so
mv result.log result.log.old
mv result.raw result.raw.old
for i in `seq 1 1 3`
do
    # Instrumented run
    SC_GPUONLY=1 make run >> result.raw 2>&1
    # Non instrumented run
    SC_GPUONLY=1 SC_NOFUNCS=1 make run >> result.raw 2>&1
done
grep "^TIME MS" result.raw > result.log
