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
