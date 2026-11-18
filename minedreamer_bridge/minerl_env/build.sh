#!/bin/bash
# Build the MineRL 1.0 image for the MineDreamer track (fakeroot, no root needed).
unset SINGULARITY_BIND APPTAINER_BIND
export APPTAINER_CACHEDIR=/scratch/jl16678/.sing_cache APPTAINER_TMPDIR=/scratch/jl16678/.sing_tmp
cd /scratch/jl16678/env/minerl
date; time /share/apps/apptainer/1.5.3/bin/apptainer build --fakeroot minerl-mp5.sif src/minerl-mp5.def
echo "build exit: $?"; date
