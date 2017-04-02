#!/bin/bash
#
#SBATCH --job-name=task1
#SBATCH --output=task1.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=32GB

# Sample script to run the model on prince.

module purge
module load python/intel
module load faster-rcnn-pytorch/intel/20170308

python -u main.py sample_config.json


