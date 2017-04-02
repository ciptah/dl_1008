#!/bin/bash
#
#SBATCH --job-name=test_lr
#SBATCH --output=test_lr.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=32GB

module purge
module load python/intel
module load faster-rcnn-pytorch/intel/20170308




python main.py test_lr.json


