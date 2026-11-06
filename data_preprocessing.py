import os
import json
import re
import numpy as np
import nltk
from nltk.tree import Tree

def strip_functional_tags(tree):
    # recursively remove functional tags and trace indices
    # e.g., NP-SBJ-1 becomes NP
    if isinstance(tree, Tree):
        label = tree.label()
        
        # skip labels starting with '-' (like -LRB-)
        if not label.startswith('-'):
            label = re.split(r'[-=]', label)[0]
        
        tree.set_label(label)
        
        # filter out trace nodes (commonly labeled -NONE-)
        new_children = []
        for child in tree:
            if isinstance(child, Tree) and child.label() == '-NONE-':
                continue
            new_children.append(strip_functional_tags(child))
        
        tree[:] = new_children
        
    return tree

def main():
    nltk.download('treebank', quiet=True)
    from nltk.corpus import treebank
    
    print("Loading Penn Treebank sample...")
    parsed_sentences = treebank.parsed_sents()
    total_sentences = len(parsed_sentences)
    print(f"Total sentences: {total_sentences}")
    
    # set the seed for reproducible partitioning
    np.random.seed(42)
    indices = np.arange(total_sentences)
    np.random.shuffle(indices)
    
    # get partition B as per project requirements
    partition_b_indices = indices[1305:2610]
    print(f"Partition B size: {len(partition_b_indices)}")
    
    # reset seed for train/val/test splitting within the partition
    np.random.seed(42)
    np.random.shuffle(partition_b_indices)
    
    num_sentences = len(partition_b_indices)
    train_size = int(0.8 * num_sentences)
    val_size = int(0.1 * num_sentences)
    
    train_indices = partition_b_indices[:train_size].tolist()
    val_indices = partition_b_indices[train_size:train_size + val_size].tolist()
    test_indices = partition_b_indices[train_size + val_size:].tolist()
    
    print(f"Train size: {len(train_indices)}")
    print(f"Validation size: {len(val_indices)}")
    print(f"Test size: {len(test_indices)}")
    
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    # set up output directory based on current script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    splits_path = os.path.join(data_dir, 'partition_b_splits.json')
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=4)
    print(f"Saved splits to {splits_path}")
    
    # process trees by stripping tags and removing traces
    processed_trees = {}
    for idx in partition_b_indices:
        tree = parsed_sentences[idx]
        tree_copy = tree.copy(deep=True)
        strip_functional_tags(tree_copy)
        
        if len(tree_copy) > 0:
            processed_trees[str(idx)] = tree_copy.pformat(margin=float('inf'))
        
    trees_path = os.path.join(data_dir, 'partition_b_trees.json')
    with open(trees_path, 'w') as f:
        json.dump(processed_trees, f, indent=4)
    print(f"Saved processed trees to {trees_path}")
    print("Preprocessing completed.")

if __name__ == "__main__":
    main()
