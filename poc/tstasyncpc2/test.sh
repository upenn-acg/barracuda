mkdir out
for j in 3 4 5 6 7 8 9 10 11 12
do
    for i in 1 2 4 8 16 32 64 128
    do
        echo ./tstasyncpc 1 0 256 $i 512 
        ./tstasyncpc 1 0 256 $i 512 > out/out1.d.$j.$i
        echo ./tstasyncpc 1 1 256 $i 512 
        ./tstasyncpc 1 1 256 $i 512 > out/out1.e.$j.$i
        echo cuda-memcheck --tool racecheck --racecheck-report all ./tstasyncpc 1 -1 256 $i 512 
        cuda-memcheck --tool racecheck --racecheck-report all ./tstasyncpc 1 -1 256 $i 512 > out/out2.r.$j.$i
        echo ./tstasyncpc 2 0 256 $i 512 
        ./tstasyncpc 2 0 256 $i 512 > out/out2.d.$j.$i
        echo ./tstasyncpc 2 1 256 $i 512 
        ./tstasyncpc 2 1 256 $i 512 > out/out2.e.$j.$i
        echo cuda-memcheck --tool racecheck --racecheck-report all ./tstasyncpc 2 -1 256 $i 512 
        cuda-memcheck --tool racecheck --racecheck-report all ./tstasyncpc 2 -1 256 $i 512 > out/out2.r.$j.$i
        echo ./tstasyncpc 3 0 256 $i 512 
        ./tstasyncpc 3 0 256 $i 512 > out/out3.d.$j.$i
        echo ./tstasyncpc 3 1 256 $i 512 
        ./tstasyncpc 3 1 256 $i 512 > out/out3.e.$j.$i
        echo cuda-memcheck --tool racecheck --racecheck-report all ./tstasyncpc 3 -1 256 $i 512 
        cuda-memcheck --tool racecheck --racecheck-report all ./tstasyncpc 3 -1 256 $i 512 > out/out3.r.$j.$i
    done
done

