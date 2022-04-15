#!/bin/bash
num_iter=1000
printResult=0
echo 'X,Y,Z,Buffer,hip_time,GFlops,num_iter,bandwidth,max error' > batch_rocResult2D.csv
for ((X=2048; X>=128; X=X/2))
do
  for((Y=2048; Y>=128; Y=Y/2))
do
    ./rocFFT_2d $X $Y 0
done
done