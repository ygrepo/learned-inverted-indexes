#!/bin/bash

module purge
module load python3/intel/3.6.3
source ~/distiller/env/bin/activate

python3 src/many_lstms_main.py --epochs 10
#python3 many_models.py --summary png
#python3 many_models.py --summary sparsity
