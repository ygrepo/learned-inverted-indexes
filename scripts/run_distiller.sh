#!/bin/bash

source ~/distiller/env/bin/activate

python3 code/many_lstms_main.py --nhid 20 --nlayers 2
#python3 many_models.py --summary png
#python3 many_models.py --summary sparsity
