# NLP Assignment 2 - Project 3

This repository contains the implementation for Project 3: From CKY to Neural Grammar Induction (Partition B).

## Environment Setup
1. Open the folder `NLP_assignment2` in VS Code.
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

### Data Preprocessing
Run the preprocessing script to download the dataset, create the dataset splits (indices [1305:2610)), and strip functional tags:
```bash
python code/data_preprocessing.py
```
This will generate `partition_b_splits.json` and `partition_b_trees.json` in the `code/data` folder.

### Task 1: CKY with a Count-Based PCFG
Run the baseline implementation which trains the model on the training trees and evaluates it on the test trees:
```bash
python code/task1/task1.py
```
The script will output Labeled and Unlabeled Bracketing F1 scores for both 0th-order and 1st-order markovization.

### Task 2: Compound PCFG (Unsupervised Grammar Induction)
Reproduction of Kim, Dyer & Rush (ACL 2019), *Compound Probabilistic Context-Free Grammars for Grammar Induction*. The model is trained only on the raw word sequences of the Partition B training split; gold trees are used only to score the test set.

**Model (as per the project spec):**
- 30 nonterminals, 60 preterminals, sentence-level latent `z` in R^64, word embedding dim 256
- Inference network: 1-layer BiLSTM (hidden 256) giving `q(z|x)`
- Rule, root and emission probabilities: 2-layer MLPs (hidden 256, ReLU) conditioned on `z`
- Loss: negative ELBO = −log p(x|z) (inside algorithm) + KL(q(z|x) || N(0, I))
- Adam, lr 1e-3, batch size 8, 20 epochs; trees decoded with Viterbi

**Preprocessing:** punctuation (`, . : `` '' -LRB- -RRB-`) is removed from both the sentences and the gold trees, following `process_ptb.py` in the official repo. Evaluation is unlabeled bracketing F1, averaged over sentences, excluding single-word and whole-sentence spans.

Run a single setup:
```bash
python code/task2/task2.py --seed 42 --lowercase --small_init --kl_warmup 4
```

| Option | Default | Meaning |
|---|---|---|
| `--epochs` | 20 | Training epochs |
| `--batch_size` | 8 | Sentences per batch (grouped by length) |
| `--lr` | 1e-3 | Adam learning rate |
| `--max_len` | 40 | Max training sentence length |
| `--seed` | 42 | Random seed |
| `--lowercase` | off | Lowercase all words |
| `--small_init` | off | Initialise all parameters uniform(−0.01, 0.01), as in the official code |
| `--kl_warmup N` | 0 (off) | Ramp the KL weight linearly from 0 to 1 over the first N epochs |

Each run writes to `code/task2/outputs/lc{0/1}_init{0/1}_kl{N}_seed{S}/`: periodic checkpoints, `best.pt` (best validation F1) and `results.json`. At the end it prints the test F1 for the **final epoch** and the **best-validation epoch**, alongside right-branching, left-branching and random-tree baselines.

> If an output folder already exists, the script resumes from its latest checkpoint. Delete the folder to start a run from scratch.

**Full experiment (setup comparison + 3 seeds):**
```bash
bash code/task2/run_experiments.sh
```
- Stage A trains three setups with seed 42: (a) baseline, (b) + lowercase + small init, (c) + KL annealing.
- Stage B trains the best setup (set `BEST` in the script) with seeds 1, 2 and 3.

Summarise all runs (mean ± std over seeds):
```bash
python code/task2/summarize.py
```

**Notes**
- With only ~1,000 training sentences, expect weaker results than the published compound PCFG, which trains on the full PTB. Report the right-branching baseline (~41% test F1) for comparison.
- Watch the `KL` column in the logs: if it stays near 0, the latent variable `z` is unused (posterior collapse).
