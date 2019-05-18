#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=7-0
#SBATCH --mem=100GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=um367@nyu.edu
#SBATCH --job-name=slurm_%j 
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge
module load python/intel/2.7.12
source /scratch/um367/urwa-env/py2.7.12/bin/activate


python vaeSiamese.py  > vaesiamese.txt


