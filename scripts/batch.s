#!/bin/bash
#SBATCH --job-name=yg390
#SBATCH --mail-user=yg390@nyu.edu
#SBATCH --mail-type=END
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
##SBATCH --gres=gpu:k80:1
##SBATCH --array=0-5
#SBATCH -o distilling_batch.log

module purge
module load python3/intel/3.6.3
source ~/distiller/env/bin/activate

python3 src/many_lstms_main.py
