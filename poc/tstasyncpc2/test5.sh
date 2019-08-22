#mkdir out.a3
#for j in `seq 13 1 25`
#do
#    for i in 1 2 4 8 16 32 64 128
#    do
#        echo ./tstasyncpc 3 0 256 $i 512 
#        ./tstasyncpc.old 3 0 256 $i 512 > out.a3/gout3.d.$j.$i
#        echo ./tstasyncpc 3 1 256 $i 512 
#        ./tstasyncpc.old 3 1 256 $i 512 > out.a3/gout3.e.$j.$i
#    done
#done
#(for i in out.a3/gout*; do awk '/^@/ { printf "%s,",$2 } END { printf "\n" }' $i; done) > out3.csv
#mkdir out.a4
#for j in `seq 16 1 26`
#do
#    for i in 128 256 512 1024
#    do
#        echo ./tstasyncpc 1 0 256 $i 512 
#        ./tstasyncpc.old 1 0 $i $((32768/i)) 512 > out.a4/gout1.d.$j.$i
#        echo ./tstasyncpc 1 1 256 $i 512 
#        ./tstasyncpc.old 1 1 $i $((32768/i)) 512 > out.a4/gout1.e.$j.$i
#        echo ./tstasyncpc 2 0 256 $i 512 
#        ./tstasyncpc.old 2 0 $i $((32768/i)) 512 > out.a4/gout2.d.$j.$i
#        echo ./tstasyncpc 2 1 256 $i 512 
#        ./tstasyncpc.old 2 1 $i $((32768/i)) 512 > out.a4/gout2.e.$j.$i
#        echo ./tstasyncpc 3 0 256 $i 512 
#        ./tstasyncpc.old 3 0 $i $((32768/i)) 512 > out.a4/gout3.d.$j.$i
#        echo ./tstasyncpc 3 1 256 $i 512 
#        ./tstasyncpc.old 3 1 $i $((32768/i)) 512 > out.a4/gout3.e.$j.$i
#    done
#done
#(for i in out.a4/gout*; do awk '/^@/ { printf "%s,",$2 } END { printf "\n" }' $i; done) > out3.csv

mkdir out.a6
for j in `seq 1 1 12000`
do
    echo ./tstasyncpc 1 0 256 $i 512 
    ./tstasyncpc.old 1 0 256 4 512 > out.a6/gout1.d1.$j
    ./tstasyncpc.old 1 0 256 4 512 > out.a6/gout1.d2.$j
    sleep 5
done


