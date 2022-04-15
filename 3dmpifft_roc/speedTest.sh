#!/bin/bash
MPI_DIR=$HOME/ompi
hostfile = "../nodelist"

# single nodes (comment if not needed)
${MPI_DIR}/bin/mpirun -np $1 -mca btl '^openib' --mca pml ucx ./distFFTRoc $2 $3 $4 1

# multiple nodes (comment if not needed)
# ${MPI_DIR}/bin/mpirun -np $1 --hostfile ${hostfile} -mca btl '^openib' --mca pml ucx ./distFFTOpt $2 $3 $4 1
