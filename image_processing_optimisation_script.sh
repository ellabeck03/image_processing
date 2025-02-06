#!/bin/bash

#SBATCH -p scarf
#SBATCH -n 32
#SBATCH -t 24:00:00
#SBATCH -o %J.log
#SBATCH -e %J.err

module load python/3.7
module load numba/0.58.1-foss-2023a

cd "/home/vol08/scarf1378"

python image_processing_optimisation.py 11_01_H_170726081325.avi
