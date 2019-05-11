#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=global_train
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --time=100:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=yzs208@nyu.edu
#SBATCH --output=global_train.out.txt

# Load modules and activate environment
module purge
module load python3/intel/3.6.3
module load cuda/9.0.176
module load cudnn/9.0v7.3.0.29
source /scratch/yzs208/learned-inverted-index/lii_env/bin/activate

# Set python variables
MODE="index2doc"
RNN="GRU"
HIDDEN=100
LAYERS=2
EPOCHS=10000
LOG=1
LOGFILE="results_"$MODE"_"$RNN"_h"$HIDDEN"_l"$LAYERS"_e"$EPOCHS".log"

echo $LOGFILE

# Run script
export PYTHONPATH=/scratch/yzs208/learned-inverted-index/lii_env/lib
python3 /scratch/yzs208/learned-inverted-index/src/global_model_train.py --mode $MODE --rnn $RNN --hidden $HIDDEN --layers $LAYERS --epochs $EPOCHS --log $LOG --log-file $LOGFILE 

# Deactivate environment and purge modules
deactivate
module purge
