#!/bin/bash

source ~/distiller/env/bin/activate

python3 src/many_lstms_main.py --resume golden_data/quantizer.checkpoint.pth.tar --summary sparsity
#python3 src/many_models.py --nhid 20 --nlayers 2
#python3 many_models.py --summary png
#python3 many_models.py --summary sparsity
