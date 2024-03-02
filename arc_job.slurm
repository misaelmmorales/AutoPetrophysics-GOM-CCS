#!/bin/bash

#SBATCH -J AutoClass     # Job name
#SBATCH -o out.%j        # Name of stdout output file
#SBATCH -e err.%j        # Name of stderr error file
#SBATCH -p gpu-a100      # Queue (partition) name
#SBATCH -N 1             # Total # of nodes (must be 1 for serial)
#SBATCH -n 1             # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 48:00:00      # Run time (hh:mm:ss)
#SBATCH -A EAR23030      # Project/Allocation name
#SBATCH --mail-type=all  # Send email at begin and end of job
#SBATCH --mail-user=misaelmorales@utexas.edu

# Startup environment
pwd
date

# Run the program
python3 auto_rock_classification.py --well_number 411 --n_classes 3 --method 'gmm' 2>&1 | tee arc.out