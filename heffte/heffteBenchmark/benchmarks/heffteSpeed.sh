#!/bin/bash

export LD_LIBRARY_PATH=/home/lulu/hyc/DistibutedFFTArtifact/heffte/heffte/lib
MPI_DIR=$HOME/binbin/new-library/ompi

${MPI_DIR}/bin/mpirun -np $1 -mca btl '^openib' --mca pml ucx ./speed3d_c2c rocfft double $2 $3 $4 -p2p_pl -slabs

