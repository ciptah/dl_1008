#!/bin/bash
#
#SBATCH --job-name=test_best
#SBATCH --output=test_best.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=32GB

module purge
module load python/intel
module load faster-rcnn-pytorch/intel/20170308




python test_sgd.py test_best.json


