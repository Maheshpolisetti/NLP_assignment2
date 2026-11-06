import os
import json
import math
import numpy as np
from collections import defaultdict
from nltk.tree import Tree

class PCFGParser:
    def __init__(self, markov_order=0):
        self.markov_order = markov_order
        self.vocab = set()
        self.rules = defaultdict(int)
        self.nonterminals = set()
        self.preterminals = set()
        self.rule_probs = {}

    def preprocess_tree(self, tree):
        # convert tree to CNF 
        tree.collapse_unary(collapsePOS=True, collapseRoot=True)
        tree.chomsky_normal_form(horzMarkov=self.markov_order, vertMarkov=0)
        return tree

    def extract_rules(self, tree):
        # get production rules from a CNF tree
        for prod in tree.productions():
            self.rules[prod] += 1
            self.nonterminals.add(prod.lhs())
            if prod.is_lexical():
                self.vocab.add(prod.rhs()[0])
                self.preterminals.add(prod.lhs())

    def train(self, trees):
        for tree in trees:
            cnf_tree = self.preprocess_tree(tree.copy(deep=True))
            self.extract_rules(cnf_tree)
            
        # apply add-1 smoothing for rules seen < 5 times
        smoothed_rules = defaultdict(int)
        lhs_counts = defaultdict(int)
        
        for rule, count in self.rules.items():
            smoothed_count = count + 1 if count < 5 else count
            smoothed_rules[rule] = smoothed_count
            lhs_counts[rule.lhs()] += smoothed_count
            
        self.rule_probs = defaultdict(dict)
        for rule, count in smoothed_rules.items():
            prob = count / lhs_counts[rule.lhs()]
            if rule.is_lexical():
                self.rule_probs[rule.lhs()][rule.rhs()[0]] = prob
            else:
                self.rule_probs[rule.lhs()][rule.rhs()] = prob
                
        # Handle OOV words with a dummy <UNK> rule for all preterminals
        for pt in self.preterminals:
            # add 1 to lhs count for UNK
            new_lhs_count = lhs_counts[pt] + 1
            self.rule_probs[pt]['<UNK>'] = 1.0 / new_lhs_count
            
    def parse(self, sentence):
        # CKY chart parsing
        n = len(sentence)
        score = [[defaultdict(float) for _ in range(n + 1)] for _ in range(n + 1)]
        back = [[{} for _ in range(n + 1)] for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            word = sentence[i - 1]
            if word not in self.vocab:
                word = '<UNK>'
            
            for lhs, rhs_dict in self.rule_probs.items():
                if word in rhs_dict:
                    score[i - 1][i][lhs] = rhs_dict[word]
                    back[i - 1][i][lhs] = word
                    
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    for lhs, rhs_dict in self.rule_probs.items():
                        for rhs, prob in rhs_dict.items():
                            if isinstance(rhs, tuple) and len(rhs) == 2:
                                B, C = rhs
                                current_score = score[i][k][B] * score[k][j][C] * prob
                                if current_score > score[i][j][lhs]:
                                    score[i][j][lhs] = current_score
                                    back[i][j][lhs] = (k, B, C)
                                    
        # find the root
        best_score = 0
        best_root = None
        for lhs in score[0][n]:
            if score[0][n][lhs] > best_score:
                best_score = score[0][n][lhs]
                best_root = lhs
                
        if best_root is None:
            return None 
            
        return self._build_tree(back, 0, n, best_root)
        
    def _build_tree(self, back, i, j, lhs):
        node = back[i][j][lhs]
        if isinstance(node, str):
            return Tree(str(lhs), [node])
        else:
            k, B, C = node
            left_tree = self._build_tree(back, i, k, B)
            right_tree = self._build_tree(back, k, j, C)
            return Tree(str(lhs), [left_tree, right_tree])


def get_brackets(tree, index=0):
    brackets = set()
    if isinstance(tree, str):
        return brackets, index + 1
        
    start_index = index
    for child in tree:
        child_brackets, index = get_brackets(child, index)
        brackets.update(child_brackets)
        
    end_index = index
    
    # skip pre-terminal nodes and empty trees
    if len(tree) > 0 and isinstance(tree[0], Tree):
        label = tree.label()
        # Ignore artificial binarization nodes created by CNF
        if '|' not in label:
            # Handle collapsed unary chains (e.g., 'S+VP' -> 'S' and 'VP')
            # By splitting on '+' and adding all of them
            for sub_label in label.split('+'):
                brackets.add((sub_label, start_index, end_index))
                
    return brackets, end_index


def evaluate(gold_trees, pred_trees):
    labeled_match = 0
    unlabeled_match = 0
    gold_total = 0
    pred_total = 0
    
    for gold, pred in zip(gold_trees, pred_trees):
        if pred is None:
            gold_brackets, _ = get_brackets(gold)
            gold_total += len(gold_brackets)
            continue
            
        gold_brackets, _ = get_brackets(gold)
        pred_brackets, _ = get_brackets(pred)
        
        gold_total += len(gold_brackets)
        pred_total += len(pred_brackets)
        
        labeled_match += len(gold_brackets.intersection(pred_brackets))
        
        gold_unlabeled = {(start, end) for label, start, end in gold_brackets}
        pred_unlabeled = {(start, end) for label, start, end in pred_brackets}
        unlabeled_match += len(gold_unlabeled.intersection(pred_unlabeled))
        
    def calc_f1(match, gold_tot, pred_tot):
        prec = match / pred_tot if pred_tot > 0 else 0
        rec = match / gold_tot if gold_tot > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return prec, rec, f1
        
    l_prec, l_rec, l_f1 = calc_f1(labeled_match, gold_total, pred_total)
    u_prec, u_rec, u_f1 = calc_f1(unlabeled_match, gold_total, pred_total)
    
    return l_prec, l_rec, l_f1, u_prec, u_rec, u_f1


def main():
    # load files relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    
    print("Loading datasets...")
    with open(os.path.join(data_dir, 'partition_b_splits.json'), 'r') as f:
        splits = json.load(f)
        
    with open(os.path.join(data_dir, 'partition_b_trees.json'), 'r') as f:
        trees_dict = json.load(f)
        
    train_trees = [Tree.fromstring(trees_dict[str(idx)]) for idx in splits['train'] if str(idx) in trees_dict]
    test_trees = [Tree.fromstring(trees_dict[str(idx)]) for idx in splits['test'] if str(idx) in trees_dict]
    
    print(f"Loaded {len(train_trees)} training trees and {len(test_trees)} testing trees.")
    
    # 0th order and 1st order markovization
    for markov_order in [0, 1]:
        print(f"\nTraining CKY Parser (Markov Order = {markov_order})")
        parser = PCFGParser(markov_order=markov_order)
        parser.train(train_trees)
        
        print("Parsing test sentences...")
        pred_trees = []
        for i, gold_tree in enumerate(test_trees):
            sentence = gold_tree.leaves()
            pred_tree = parser.parse(sentence)
            pred_trees.append(pred_tree)
            
            if (i + 1) % 10 == 0:
                print(f"Parsed {i + 1}/{len(test_trees)} sentences")
                
        l_prec, l_rec, l_f1, u_prec, u_rec, u_f1 = evaluate(test_trees, pred_trees)
        
        print(f"\nResults for Markov Order {markov_order}:")
        print(f"Labeled Bracketing   - Precision: {l_prec:.4f}, Recall: {l_rec:.4f}, F1: {l_f1:.4f}")
        print(f"Unlabeled Bracketing - Precision: {u_prec:.4f}, Recall: {u_rec:.4f}, F1: {u_f1:.4f}")

if __name__ == "__main__":
    main()
