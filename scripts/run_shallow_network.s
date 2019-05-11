#!/bin/bash
#SBATCH --job-name=yg390
#SBATCH --mail-user=yg390@nyu.edu
#SBATCH --mail-type=END
#SBATCH --time=10:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
##SBATCH --gres=gpu:2
##SBATCH --gres=gpu:k80:1
##SBATCH --array=0-5
##SBATCH -o shallow_batch.log

module purge
module load python3/intel/3.6.3
source ~/env-gpu/py3.6.3/bin/activate

EPOCHS=10000
LOG_INTERVAL=1000
BATCH_SIZE=32
POSTING_LIST_SIZE=4096
LR=0.01
LOSS_STOP_THRESHOLD=1e-3
LOSS_SCRAP=10000000

python3 src/shallow_network_train.py --epochs $EPOCHS --log-interval $LOG_INTERVAL --batch_size $BATCH_SIZE --posting_list_size $POSTING_LIST_SIZE --lr $LR --loss_stop_threshold $LOSS_STOP_THRESHOLD --loss_scrap $LOSS_SCRAP

# Deactivate environment and purge modules
deactivate
module purge