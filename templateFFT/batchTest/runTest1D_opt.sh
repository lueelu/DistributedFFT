#!/bin/bash
num_iter=1000
printResult=0
echo 'X,Y,Z,Buffer,hip_time,GFlops,num_iter,bandwidth,max error' > batch_result1D.csv
for ((X=256; X<=131072; X=X*2))
do
    ./Test_1D $X 1 1 $num_iter $printResult
    # ./rocFFT_1d $X 1 $printResult
done
for ((X=3; X<=14348907; X=X*3))
do
    ./Test_1D $X 1 1 $num_iter $printResult
done
for ((X=5; X<=48828125; X=X*5))
do
    ./Test_1D $X 1 1 $num_iter $printResult
done
for ((X=7; X<=40353607; X=X*7))
do
    ./Test_1D $X 1 1 $num_iter $printResult
done