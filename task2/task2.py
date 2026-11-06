#!/usr/bin/env python3
"""
Task 2: Compound PCFG (Unsupervised Grammar Induction)
NLP Assignment 2 - Project 3, Partition B

A clean, minimal implementation of Compound PCFG (Kim et al., 2019)
reusing preprocessed trees from data_preprocessing.py directly.
"""

import os
import re
import sys
import glob
import json
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tree import Tree


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

PUNCT_TAGS = {',', '.', ':', '``', "''", '-LRB-', '-RRB-', '-NONE-'}


def extract_spans(tree):
    """
    Drop punctuation (standard unsupervised-parsing setup, Kim et al. 2019) and
    return (words, spans): non-trivial bracket spans (s, t), 0 <= s < t <= n - 1,
    excluding the whole-sentence span.
    """
    words = [w for w, tag in tree.pos() if tag not in PUNCT_TAGS]
    n = len(words)
    spans = set()

    def _traverse(node, start):
        if isinstance(node, str) or (len(node) == 1 and isinstance(node[0], str)):
            return start if node.label() in PUNCT_TAGS else start + 1
        curr = start
        for child in node:
            curr = _traverse(child, curr)
        end = curr
        s = start
        t = end - 1
        if s < t and not (s == 0 and t == n - 1):
            spans.add((s, t))
        return end

    _traverse(tree, 0)
    return words, spans


def load_data(data_dir: Path, max_len: int = 40, lowercase: bool = False):
    """
    Reuses existing partition_b_splits.json and partition_b_trees.json directly.
    """
    splits_file = data_dir / 'partition_b_splits.json'
    trees_file = data_dir / 'partition_b_trees.json'

    if not splits_file.exists() or not trees_file.exists():
        raise FileNotFoundError(f"Missing data in {data_dir}. Expected partition_b_splits.json and partition_b_trees.json")

    with open(splits_file, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    with open(trees_file, 'r', encoding='utf-8') as f:
        trees = json.load(f)

    def _process_split(indices, is_train=False):
        samples = []
        for idx in indices:
            s_idx = str(idx)
            if s_idx not in trees:
                continue
            words, gold_spans = extract_spans(Tree.fromstring(trees[s_idx]))
            if lowercase:
                words = [w.lower() for w in words]
            length = len(words)
            if length < 2 or (is_train and max_len and length > max_len):
                continue
            samples.append({
                'words': words,
                'gold_spans': gold_spans,
                'length': length
            })
        return samples

    train_data = _process_split(splits['train'], is_train=True)
    val_data = _process_split(splits['val'], is_train=False)
    test_data = _process_split(splits['test'], is_train=False)

    return train_data, val_data, test_data


def build_vocab(train_samples, min_freq: int = 2):
    """
    Build vocabulary strictly from training sentences.
    """
    counts = Counter()
    for s in train_samples:
        counts.update(s['words'])

    word2idx = {'<pad>': 0, '<unk>': 1}
    for w, c in counts.items():
        if c >= min_freq:
            word2idx[w] = len(word2idx)
    return word2idx


def make_batches(samples, word2idx, batch_size: int = 4):
    """
    Groups samples by sentence length for uniform batch processing.
    """
    length_bins = defaultdict(list)
    for s in samples:
        length_bins[s['length']].append(s)

    batches = []
    for length, items in length_bins.items():
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            tensor = torch.tensor(
                [[word2idx.get(w, 1) for w in item['words']] for item in chunk],
                dtype=torch.long
            )
            batches.append({'tensor': tensor, 'samples': chunk})
    return batches


# =============================================================================
# MODEL ARCHITECTURE: COMPOUND PCFG
# =============================================================================

class CompPCFG(nn.Module):
    """
    Compound PCFG Model (Kim et al., 2019).

    Architecture (per assignment spec):
      - Inference-network encoder: 1-layer BiLSTM, hidden 256
      - Rule probabilities: 2-layer MLP (hidden 256, ReLU) conditioned on z
      - NT=30, PT=60, z_dim=64, word_emb=256
    """
    def __init__(self, vocab_size: int,
                 w_dim: int = 256,
                 h_dim: int = 256,
                 z_dim: int = 64,
                 state_dim: int = 256,
                 nt_states: int = 30,
                 t_states: int = 60):
        super(CompPCFG, self).__init__()
        self.vocab_size = vocab_size
        self.z_dim = z_dim
        self.state_dim = state_dim
        self.nt_states = nt_states
        self.t_states = t_states
        self.all_states = nt_states + t_states

        # Learnable state embeddings
        self.root_emb = nn.Parameter(torch.randn(1, state_dim))
        self.nt_emb = nn.Parameter(torch.randn(nt_states, state_dim))
        self.t_emb = nn.Parameter(torch.randn(t_states, state_dim))

        # Inference network: BiLSTM encoder for variational posterior q(z|x)
        self.enc_emb = nn.Embedding(vocab_size, w_dim, padding_idx=0)
        self.enc_rnn = nn.LSTM(w_dim, h_dim, bidirectional=True, num_layers=1, batch_first=True)
        self.enc_params = nn.Linear(h_dim * 2, z_dim * 2)

        # Generative MLPs: 2-layer MLP (hidden 256, ReLU) conditioned on z
        in_dim = state_dim + z_dim
        self.root_mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, nt_states)
        )
        self.rule_mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.all_states ** 2)
        )
        self.vocab_mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, vocab_size)
        )

    def encode(self, x):
        """BiLSTM encoder -> mu, logvar for q(z|x)."""
        emb = self.enc_emb(x)
        h, _ = self.enc_rnn(emb)
        params = self.enc_params(h.max(dim=1)[0])
        return params[:, :self.z_dim], params[:, self.z_dim:]

    def _get_params(self, x, z):
        """Compute all PCFG parameters from input x and latent z."""
        batch_size, n = x.shape

        # Root log-probs: pi(z)
        root_input = torch.cat([self.root_emb.expand(batch_size, -1), z], dim=1)
        root_scores = F.log_softmax(self.root_mlp(root_input), dim=1)

        # Binary rule log-probs: R_A(z) for each NT parent A
        z_nt = z.unsqueeze(1).expand(batch_size, self.nt_states, self.z_dim)
        nt_input = torch.cat([self.nt_emb.unsqueeze(0).expand(batch_size, self.nt_states, -1), z_nt], dim=2)
        rule_scores = F.log_softmax(self.rule_mlp(nt_input), dim=2).view(
            batch_size, self.nt_states, self.all_states, self.all_states
        )

        # Emission log-probs: E_T(z) for each PT state T
        z_t = z.unsqueeze(1).expand(batch_size, self.t_states, self.z_dim)
        t_input = torch.cat([self.t_emb.unsqueeze(0).expand(batch_size, self.t_states, -1), z_t], dim=2)
        vocab_logprobs = F.log_softmax(self.vocab_mlp(t_input), dim=2).transpose(1, 2)
        # unary_scores[b, i, t] = log P(word_i | preterminal t)
        unary_scores = torch.gather(
            vocab_logprobs,
            dim=1,
            index=x.unsqueeze(2).expand(batch_size, n, self.t_states)
        )

        return root_scores, rule_scores, unary_scores

    def _sample_z(self, x, use_mean):
        """q(z|x) via BiLSTM encoder; returns z and per-sentence KL."""
        mu, logvar = self.encode(x)
        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        z = mu if use_mean else mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return z, kl

    def forward(self, x, use_mean: bool = False, kl_w: float = 1.0):
        """Returns negative ELBO (KL weighted by kl_w, mean over batch) and per-sentence KL."""
        z, kl = self._sample_z(x, use_mean)
        log_Z = self._cky(*self._get_params(x, z))
        return -(log_Z - kl_w * kl).mean(), kl

    @torch.no_grad()
    def parse(self, x):
        """Viterbi parse using the posterior mean of z; returns span sets."""
        z, _ = self._sample_z(x, use_mean=True)
        return self._cky(*self._get_params(x, z), viterbi=True)

    def _cky(self, root_scores, rule_scores, unary_scores, viterbi: bool = False):
        """
        Span-vectorised CKY in log space. All spans of the same width are
        processed in one tensor op, so Python loops are O(n^2) instead of O(n^3).

        chart[w] has shape (B, n-w, states): width-0 cells hold the T
        preterminal states, wider cells hold the NT states.
        inside  (viterbi=False): returns log P(x|z), shape (B,)
        viterbi (viterbi=True) : returns list of predicted span sets
        """
        B, n, _ = unary_scores.shape
        NT = self.nt_states
        chart, backptrs = [unary_scores], []

        for w in range(1, n):
            P = n - w
            vals, idxs = [], []
            for k in range(w):  # left child width k, right child width w-1-k
                L = chart[k][:, :P]
                R = chart[w - 1 - k][:, k + 1:k + 1 + P]
                ls = slice(NT, None) if k == 0 else slice(None, NT)
                rs = slice(NT, None) if k == w - 1 else slice(None, NT)
                rule = rule_scores[:, :, ls, rs]  # (B, NT, dl, dr)
                if viterbi:
                    comb = L[:, :, None, :, None] + R[:, :, None, None, :] + rule[:, None]
                    v, i = comb.flatten(3).max(-1)
                    idxs.append(i)
                else:
                    lm = L.max(-1, keepdim=True)[0]
                    rm = R.max(-1, keepdim=True)[0]
                    v = torch.einsum('bpl,bpr,balr->bpa', (L - lm).exp(), (R - rm).exp(), rule.exp())
                    v = v.clamp_min(1e-30).log() + lm + rm
                vals.append(v)
            stacked = torch.stack(vals, -1)  # (B, P, NT, w)
            if viterbi:
                best, k_best = stacked.max(-1)
                child = torch.stack(idxs, -1).gather(-1, k_best.unsqueeze(-1)).squeeze(-1)
                backptrs.append((k_best.cpu().numpy(), child.cpu().numpy()))
                chart.append(best)
            else:
                chart.append(torch.logsumexp(stacked, -1))

        top = chart[-1][:, 0] + root_scores  # (B, NT)
        if not viterbi:
            return torch.logsumexp(top, -1)

        roots = top.argmax(-1).tolist()
        T = self.t_states
        batch_spans = []
        for b in range(B):
            spans, stack = set(), [(0, n - 1, roots[b])]
            while stack:
                s, w, a = stack.pop()
                if w == 0:
                    continue
                if w < n - 1:
                    spans.add((s, s + w))
                k_arr, c_arr = backptrs[w - 1]
                k, c = int(k_arr[b, s, a]), int(c_arr[b, s, a])
                l, r = divmod(c, T if k == w - 1 else NT)
                stack.append((s, k, l))
                stack.append((s + k + 1, w - 1 - k, r))
            batch_spans.append(spans)
        return batch_spans


# =============================================================================
# TRAINING, EVALUATION & CHECKPOINTING
# =============================================================================

def train_epoch(model, batches, optimizer, device, kl_w=1.0):
    model.train()
    total_loss, total_kl, total_sents = 0.0, 0.0, 0
    random.shuffle(batches)

    num_batches = len(batches)
    epoch_start = time.time()

    for batch_idx, b in enumerate(batches):
        x = b['tensor'].to(device)
        b_size = x.size(0)

        optimizer.zero_grad()
        loss, kl = model(x, kl_w=kl_w)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        total_loss += loss.item() * b_size
        total_kl += kl.sum().item()
        total_sents += b_size

        # Progress reporting every 50 batches
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
            elapsed = time.time() - epoch_start
            avg_loss = total_loss / max(1, total_sents)
            print(f"  [{batch_idx+1}/{num_batches}] sents={total_sents} "
                  f"loss={avg_loss:.4f} elapsed={elapsed:.1f}s", flush=True)

    return total_loss / max(1, total_sents), total_kl / max(1, total_sents)


def span_f1(pred, gold):
    """Sentence-level unlabeled bracketing F1 between two span sets."""
    if not gold:
        return 1.0 if not pred else 0.0
    tp = len(pred & gold)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gold)
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0


def evaluate(model, batches, device):
    """Mean sentence-level F1 (%) of the model's Viterbi parses."""
    model.eval()
    f1s = []
    for b in batches:
        preds = model.parse(b['tensor'].to(device))
        f1s += [span_f1(p, s['gold_spans']) for p, s in zip(preds, b['samples'])]
    return 100.0 * float(np.mean(f1s))


def random_tree(i, j, n):
    """Spans of a uniformly-split random binary tree over words i..j."""
    if i >= j:
        return set()
    k = random.randint(i, j - 1)
    spans = random_tree(i, k, n) | random_tree(k + 1, j, n)
    if not (i == 0 and j == n - 1):
        spans.add((i, j))
    return spans


def baseline_f1(samples, kind):
    """Mean sentence-level F1 (%) of right-, left-branching or random trees."""
    f1s = []
    for s in samples:
        n = s['length']
        if kind == 'right':
            pred = {(i, n - 1) for i in range(1, n - 1)}
        elif kind == 'left':
            pred = {(0, j) for j in range(1, n - 1)}
        else:
            pred = random_tree(0, n - 1, n)
        f1s.append(span_f1(pred, s['gold_spans']))
    return 100.0 * float(np.mean(f1s))


def save_checkpoint(path: Path, epoch: int, model, optimizer, best_f1: float, word2idx: dict):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_f1': best_f1,
        'word2idx': word2idx
    }, path)


def load_latest_checkpoint(output_dir: Path, model, optimizer, device):
    """
    Auto-resumes from the latest checkpoint_epoch_*.pt or best.pt if available.
    """
    ckpts = glob.glob(str(output_dir / 'checkpoint_epoch_*.pt'))
    if not ckpts:
        best_p = output_dir / 'best.pt'
        if best_p.exists():
            ckpts = [str(best_p)]
    if not ckpts:
        return 1, 0.0

    def _extract_ep(p):
        m = re.search(r'checkpoint_epoch_(\d+)\.pt', p)
        return int(m.group(1)) if m else -1

    ckpts.sort(key=_extract_ep)
    latest_ckpt = ckpts[-1]
    print(f"Discovered existing checkpoint: {latest_ckpt}. Restoring state...")
    ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_f1 = ckpt.get('best_f1', 0.0)
    print(f"Resumed successfully! Next training epoch: {start_epoch}, previous best F1: {best_f1:.2f}%")
    return start_epoch, best_f1


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Task 2: Compound PCFG (Unsupervised)")
    parser.add_argument('--data_dir', type=str, default=None,
                        help="Path to directory containing partition_b_splits.json and partition_b_trees.json")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Path to save checkpoints (defaults to outputs/)")
    parser.add_argument('--epochs', type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument('--lowercase', action='store_true',
                        help="Lowercase all words (as in the official repo)")
    parser.add_argument('--small_init', action='store_true',
                        help="Init all parameters uniform(-0.01, 0.01) (as in the official repo)")
    parser.add_argument('--kl_warmup', type=int, default=0,
                        help="Ramp KL weight from 0 to 1 over this many epochs (0 = off)")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size, spec says 4-8 (default: 8)")
    parser.add_argument('--max_len', type=int, default=40,
                        help="Max sentence length for training (default: 40)")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate for Adam optimizer (default: 1e-3)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Path resolution relative to script location
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    data_dir = Path(args.data_dir) if args.data_dir else (project_root / 'data')
    run_name = (f"lc{int(args.lowercase)}_init{int(args.small_init)}"
                f"_kl{args.kl_warmup}_seed{args.seed}")
    output_dir = Path(args.output_dir) if args.output_dir else (script_dir / 'outputs' / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Loading preprocessed data from: {data_dir}")

    # 1. Load data directly from data_preprocessing.py outputs
    train_samples, val_samples, test_samples = load_data(data_dir, args.max_len, args.lowercase)
    print(f"Train sentences: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # 2. Build vocabulary from training split only
    word2idx = build_vocab(train_samples, min_freq=2)
    print(f"Vocabulary size: {len(word2idx)}")

    # 3. Create length-compatible batches
    train_batches = make_batches(train_samples, word2idx, batch_size=args.batch_size)
    val_batches = make_batches(val_samples, word2idx, batch_size=args.batch_size)
    test_batches = make_batches(test_samples, word2idx, batch_size=args.batch_size)
    print(f"Train batches: {len(train_batches)}, Val batches: {len(val_batches)}, Test batches: {len(test_batches)}")

    # 4. Initialize Compound PCFG model
    model = CompPCFG(
        vocab_size=len(word2idx),
        w_dim=256,
        h_dim=256,
        z_dim=64,
        state_dim=256,
        nt_states=30,
        t_states=60
    ).to(device)
    if args.small_init:
        for p in model.parameters():
            p.data.uniform_(-0.01, 0.01)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.75, 0.999))

    # 5. Checkpoint resume
    start_epoch, best_f1 = load_latest_checkpoint(output_dir, model, optimizer, device)

    # 6. Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Epoch {epoch:02d}/{args.epochs:02d} ---")
        epoch_start = time.time()

        kl_w = min(1.0, epoch / args.kl_warmup) if args.kl_warmup > 0 else 1.0
        train_loss, train_kl = train_epoch(model, train_batches, optimizer, device, kl_w)

        print("  Evaluating on val set...", flush=True)
        val_f1 = evaluate(model, val_batches, device)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {train_loss:.4f} | KL: {train_kl:.4f} "
              f"| KL_w: {kl_w:.2f} "
              f"| Val F1: {val_f1:.2f}% | Time: {epoch_time:.1f}s")

        # Save checkpoint every 2 epochs
        if epoch % 2 == 0 or epoch == args.epochs:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch:02d}.pt"
            save_checkpoint(ckpt_path, epoch, model, optimizer, best_f1, word2idx)
            print(f"Saved periodic checkpoint: {ckpt_path.name}")

        # Save best model by validation F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_path = output_dir / "best.pt"
            save_checkpoint(best_path, epoch, model, optimizer, best_f1, word2idx)
            print(f"*** New best model saved! (Val F1: {best_f1:.2f}%) ***")

    # 7. Test evaluation: final-epoch model, then best-val model, plus baselines
    results = {'run': run_name, 'test_f1_final': evaluate(model, test_batches, device)}
    ckpt = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    results['best_epoch'] = ckpt['epoch']
    results['test_f1_best'] = evaluate(model, test_batches, device)
    for kind in ('right', 'left', 'random'):
        results[f'baseline_{kind}'] = baseline_f1(test_samples, kind)

    print("=" * 60)
    for k, v in results.items():
        print(f"{k:16s}: {v:.2f}" if isinstance(v, float) else f"{k:16s}: {v}")
    print("=" * 60)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
