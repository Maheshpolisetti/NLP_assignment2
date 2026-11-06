#!/bin/bash
# Stage A: compare setups (seed 42, 20 epochs). Stage B: 3 seeds of the best one.
cd "$(dirname "$0")"

# Stage A
python task2.py --seed 42                                      # (a) baseline
python task2.py --seed 42 --lowercase --small_init             # (b) + lowercase + small init
python task2.py --seed 42 --lowercase --small_init --kl_warmup 4   # (c) + KL annealing
python summarize.py

# Stage B: set BEST to the flags of the setup with the best val F1 from Stage A
BEST="--lowercase --small_init --kl_warmup 4"
for s in 1 2 3; do
    python task2.py --seed $s $BEST
done
python summarize.py
