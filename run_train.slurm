#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:1
#SBATCH --job-name=esm2
#SBATCH --output=logs/esm2.log
#SBATCH --mem=128G

set -e

/hai/scratch/fangwu97/miniconda3/envs/sim/bin/python /hai/scratch/fangwu97/xu/Personal/train_esm2.py