#!/bin/bash

python multi_label/surveillance/run.py \
    --size 10000 \
    --lr 1e-3 \
    --epochs 1200 \
    --is_privacy_preserve \
    # --hidden_units "128,128" \
    
    
    
    
# Training size 1000, best results
# Without DPSGD, binary_acc=1.0 : lr=1e-2, hidden_size=100, epochs=1000
# With DPSGD, binary_acc=0.9412: lr=1e-3, hidden_size=100, epochs=2000, batch_size=10



# Training size 10000, best results
# Without DPSGD, binary_acc=0.9716: lr=1e-3, hidden_size=128,128, epochs=500, loss=0.1075 
# With DPSGD, binary_acc=0.9413: lr=1e-3, hidden_size=100, epochs=1200