OUTDIR=out.a6
OUTNAME=out
mkdir $OUTDIR
for q in 1 2 3
do
for r in `seq 3 1 17`
do
    for j in 4 8 16 24 32 48
    do
        for i in 32 64
        do
            echo Benchmark $q - i=$i j=$j
#            timeout 20m ./tstasyncpc $j $q 0 1024 $i 64 >& $OUTDIR/$OUTNAME.d.$q.$r.$j.$i
            timeout 20m ./tstasyncpc $j $q 1 1024 $i 64 >& $OUTDIR/$OUTNAME.e.$q.$r.$j.$i
        done
    done
done
done
(for i in $OUTDIR/$OUTNAME.*; do awk '/^@/ { printf "%s,",$2 } END { printf "\n" }' $i; done) > ${OUTDIR}.csv
