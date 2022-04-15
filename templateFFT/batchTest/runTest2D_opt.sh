#!/bin/bash
num_iter=1000
printResult=0
echo 'X,Y,Z,Buffer,hip_time,GFlops,num_iter,bandwidth,max error' > batch_result2D.csv
for ((X=2048; X>=128; X=X/2))
do
  for((Y=2048; Y>=128; Y=Y/2))
do
    ./Test_2D $X $Y 1 $num_iter $printResult
done
done