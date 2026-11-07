
import sys
import json
import time
import random
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tree import Tree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from task1 import evaluate as parseval_evaluate         
from task2 import extract_spans, span_f1, PUNCT_TAGS   


# =============================================================================
# DATA
# =============================================================================

def load_trees(data_dir: Path):
    with open(data_dir / 'partition_b_splits.json', 'r', encoding='utf-8') as f:
        splits = json.load(f)
    with open(data_dir / 'partition_b_trees.json', 'r', encoding='utf-8') as f:
        trees = json.load(f)

    def get(name):
        return [Tree.fromstring(trees[str(i)]) for i in splits[name] if str(i) in trees]

    return get('train'), get('val'), get('test')


def binarize(tree, markov_order):
    """Same CNF conversion as Task 1."""
    t = tree.copy(deep=True)
    t.collapse_unary(collapsePOS=True, collapseRoot=True)
    t.chomsky_normal_form(horzMarkov=markov_order, vertMarkov=0)
    return t


def _is_pt(n):
    return len(n) == 1 and isinstance(n[0], str)


def extract(tree):
    """CNF tree -> (root_label | None, binary rules, lexical rules) as label tuples."""
    rules, lex = [], []

    def rec(n):
        if _is_pt(n):
            lex.append((n.label(), n[0]))
            return
        if len(n) != 2 or not all(isinstance(c, Tree) for c in n):
            raise ValueError(f"Tree is not in CNF at node {n.label()}")
        l, r = n
        rules.append((n.label(), (l.label(), _is_pt(l)), (r.label(), _is_pt(r))))
        rec(l)
        rec(r)

    rec(tree)
    return (None if _is_pt(tree) else tree.label()), rules, lex


def encode(ex, nt2i, pt2i, w2i):
    """Label tuples -> integer arrays. Rules with symbols unseen in training are skipped."""
    root, rules, lex = ex
    NT = len(nt2i)
    skipped = 0
    R, L = [], []
    for a, (b, bp), (c, cp) in rules:
        try:
            ai = nt2i[a]
            bi = NT + pt2i[b] if bp else nt2i[b]
            ci = NT + pt2i[c] if cp else nt2i[c]
        except KeyError:
            skipped += 1
            continue
        R.append((ai, bi, ci))
    for t, w in lex:
        if t not in pt2i:
            skipped += 1
            continue
        L.append((pt2i[t], w2i.get(w, 0)))
    return {
        'root': nt2i.get(root, -1) if root is not None else -1,
        'rules': np.array(R, dtype=np.int64).reshape(-1, 3),
        'lex': np.array(L, dtype=np.int64).reshape(-1, 2),
        'skipped': skipped,
    }


def collate(items, device):
    roots = np.array([it['root'] for it in items if it['root'] >= 0], dtype=np.int64)
    rules = np.concatenate([it['rules'] for it in items])
    lex = np.concatenate([it['lex'] for it in items])
    to = lambda a: torch.from_numpy(a).to(device)
    return to(roots), to(rules), to(lex)


# =============================================================================
# MODEL: Task 2's neural PCFG parameterisation without z
# =============================================================================

class SupNeuralPCFG(nn.Module):
    def __init__(self, n_nt, n_pt, vocab_size, state_dim=256, hidden=256):
        super().__init__()
        self.n_nt, self.n_pt, self.S = n_nt, n_pt, n_nt + n_pt
        self.root_emb = nn.Parameter(torch.randn(1, state_dim))
        self.nt_emb = nn.Parameter(torch.randn(n_nt, state_dim))
        self.pt_emb = nn.Parameter(torch.randn(n_pt, state_dim))
        # 2-layer MLPs (hidden 256, ReLU) as in Task 2, but conditioned only on the state embedding
        self.root_mlp = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_nt))
        self.rule_mlp = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, self.S ** 2))
        self.vocab_mlp = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, vocab_size))

    def grammar(self):
        """Full set of log-probabilities: root (NT,), rules (NT, S, S), emissions (PT, V)."""
        root = F.log_softmax(self.root_mlp(self.root_emb), dim=-1).squeeze(0)
        rules = F.log_softmax(self.rule_mlp(self.nt_emb), dim=-1).view(self.n_nt, self.S, self.S)
        emis = F.log_softmax(self.vocab_mlp(self.pt_emb), dim=-1)
        return root, rules, emis

    def tree_loglik(self, roots, rules, lex):
        """Sum of log P(tree) over all trees in the batch (rule counts are observed)."""
        root_lp, rule_lp, emis_lp = self.grammar()
        return (root_lp[roots].sum()
                + rule_lp[rules[:, 0], rules[:, 1], rules[:, 2]].sum()
                + emis_lp[lex[:, 0], lex[:, 1]].sum())


# =============================================================================
# VITERBI CKY OVER THE LEARNED (PRUNED) LABELLED GRAMMAR
# =============================================================================

class SparseViterbiCKY:
    """
    Viterbi CKY. The neural grammar is dense (|NT| x S x S), so for decoding each
    parent keeps its top-k most probable binary rules (default k=300; the remaining
    mass is tiny after training). Rules are grouped by whether the left/right child
    is a pre-terminal (width-0 cell) or a non-terminal (wider cell), and every
    (span, group) is scored with one vectorised op.
    """

    def __init__(self, model, nt_names, pt_names, w2i, topk):
        with torch.no_grad():
            root_lp, rule_lp, emis_lp = model.grammar()
        self.dev = root_lp.device
        self.NT, self.PT, self.S = model.n_nt, model.n_pt, model.S
        self.nt_names, self.pt_names, self.w2i = nt_names, pt_names, w2i
        self.root_lp, self.emis = root_lp, emis_lp

        NT, S = self.NT, self.S
        k = min(topk, S * S)
        vals, idx = rule_lp.reshape(NT, S * S).topk(k, dim=1)
        A = torch.arange(NT, device=self.dev).unsqueeze(1).expand(NT, k).reshape(-1)
        B = torch.div(idx, S, rounding_mode='floor').reshape(-1)
        C = (idx % S).reshape(-1)
        lp = vals.reshape(-1)

        self.groups = []
        for lpt in (0, 1):
            for rpt in (0, 1):
                m = ((B >= NT).long() == lpt) & ((C >= NT).long() == rpt)
                self.groups.append({
                    'lpt': bool(lpt), 'rpt': bool(rpt),
                    'A': A[m], 'B': B[m] - lpt * NT, 'C': C[m] - rpt * NT, 'lp': lp[m],
                })
        for g in self.groups:
            g['B_np'], g['C_np'] = g['B'].cpu().numpy(), g['C'].cpu().numpy()

    @torch.no_grad()
    def parse(self, words):
        n, NT, dev = len(words), self.NT, self.dev
        NEG = float('-inf')
        wid = torch.tensor([self.w2i.get(w, 0) for w in words], device=dev)
        unary = self.emis[:, wid].t().contiguous()                     # (n, PT)
        if n == 1:
            return Tree(self.pt_names[int(unary[0].argmax())], [words[0]])

        chart = torch.full((n, n, NT), NEG, device=dev)                # chart[width, start, NT]
        bk = torch.zeros((n, n, NT), dtype=torch.long, device=dev)     # split k (left width)
        bg = torch.zeros_like(bk)                                      # rule group id
        br = torch.zeros_like(bk)                                      # rule index inside group

        for w in range(1, n):
            for s in range(n - w):
                best = torch.full((NT,), NEG, device=dev)
                cbk, cbg, cbr = (torch.zeros(NT, dtype=torch.long, device=dev) for _ in range(3))
                for gid, g in enumerate(self.groups):
                    if g['A'].numel() == 0:
                        continue
                    if g['lpt'] and g['rpt']:
                        ks = [0] if w == 1 else []
                    elif g['lpt']:
                        ks = [0] if w >= 2 else []
                    elif g['rpt']:
                        ks = [w - 1] if w >= 2 else []
                    else:
                        ks = list(range(1, w - 1))
                    if not ks:
                        continue
                    kt = torch.tensor(ks, device=dev)
                    Lc = unary[s].unsqueeze(0) if g['lpt'] else chart[kt, s]
                    Rc = unary[s + kt + 1] if g['rpt'] else chart[w - 1 - kt, s + kt + 1]
                    vals = (Lc[:, g['B']] + Rc[:, g['C']] + g['lp']).reshape(-1)   # (K*R,)
                    K, R = len(ks), g['A'].numel()
                    Arep = g['A'].repeat(K)

                    gbest = torch.full((NT,), NEG, device=dev).scatter_reduce(
                        0, Arep, vals, 'amax', include_self=True)
                    upd = gbest > best
                    if not upd.any():
                        continue
                    pos = torch.arange(K * R, device=dev)
                    cand = torch.where(vals == gbest[Arep], pos, torch.full_like(pos, K * R))
                    arg = torch.full((NT,), K * R, device=dev).scatter_reduce(
                        0, Arep, cand, 'amin', include_self=True)
                    arg = torch.where(upd, arg, torch.zeros_like(arg))
                    best = torch.where(upd, gbest, best)
                    cbk = torch.where(upd, kt[arg // R], cbk)
                    cbg = torch.where(upd, torch.full_like(cbg, gid), cbg)
                    cbr = torch.where(upd, arg % R, cbr)
                chart[w, s], bk[w, s], bg[w, s], br[w, s] = best, cbk, cbg, cbr

        top = chart[n - 1, 0] + self.root_lp
        if not torch.isfinite(top.max()):
            return None                                                 # pruned grammar cannot parse it
        root = int(top.argmax())
        bk, bg, br = bk.cpu().numpy(), bg.cpu().numpy(), br.cpu().numpy()

        def leaf(i, p):
            return Tree(self.pt_names[p], [words[i]])

        def build(w, s, a):
            k, gid, r = int(bk[w, s, a]), int(bg[w, s, a]), int(br[w, s, a])
            g = self.groups[gid]
            b, c = int(g['B_np'][r]), int(g['C_np'][r])
            left = leaf(s, b) if g['lpt'] else build(k, s, b)
            right = leaf(s + k + 1, c) if g['rpt'] else build(w - 1 - k, s + k + 1, c)
            return Tree(self.nt_names[a], [left, right])

        return build(n - 1, 0, root)


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def pred_spans_no_punct(pred_tree, gold_tree):
    """
    Spans of a predicted tree in Task 2's convention: punctuation removed (using the
    gold POS tags), inclusive (s, t) indices, trivial / whole-sentence spans dropped.
    """
    if pred_tree is None:
        return set()
    keep = [tag not in PUNCT_TAGS for _, tag in gold_tree.pos()]
    cum = [0]
    for kp in keep:
        cum.append(cum[-1] + int(kp))
    n = cum[-1]
    spans = set()

    def rec(node, i):
        if isinstance(node, str) or _is_pt(node):
            return i + 1
        j = i
        for c in node:
            j = rec(c, j)
        a, b = cum[i], cum[j] - 1
        if a < b and not (a == 0 and b == n - 1):
            spans.add((a, b))
        return j

    rec(pred_tree, 0)
    return spans


@torch.no_grad()
def mean_nll(model, items, device):
    """Mean per-tree negative log-likelihood and per-event NLL on held-out gold trees."""
    model.eval()
    roots, rules, lex = collate(items, device)
    ll = model.tree_loglik(roots, rules, lex).item()
    events = len(roots) + len(rules) + len(lex)
    return -ll / len(items), -ll / events


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Task 3: Supervised Neural PCFG")
    p.add_argument('--data_dir', type=str, default=None)
    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=32, help="trees per step")
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=8, help="early stop on val NLL (0 = off)")
    p.add_argument('--markov_order', type=int, default=0)
    p.add_argument('--min_freq', type=int, default=2, help="rarer words -> <unk> (as in Task 2)")
    p.add_argument('--state_dim', type=int, default=256)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--topk', type=int, default=300, help="rules kept per parent for CKY")
    p.add_argument('--max_rule_entries', type=float, default=2e8,
                   help="abort if |NT|*S*S exceeds this (memory guard)")
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = HERE.parent / 'data'
        if not (data_dir / 'partition_b_splits.json').exists():
            data_dir = HERE / 'data'
    out_dir = Path(args.output_dir) if args.output_dir else HERE / 'outputs' / f"task3_m{args.markov_order}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}\nLoading data from {data_dir}")
    train_t, val_t, test_t = load_trees(data_dir)
    print(f"Train {len(train_t)} | Val {len(val_t)} | Test {len(test_t)} trees")

    # ---- gold trees -> CNF -> rule tuples (same CNF as Task 1) -----------------
    ex_train = [extract(binarize(t, args.markov_order)) for t in train_t]
    ex_val = [extract(binarize(t, args.markov_order)) for t in val_t]

    nt_set, pt_set, wc = set(), set(), Counter()
    for root, rules, lex in ex_train:
        for a, (b, bp), (c, cp) in rules:
            nt_set.add(a)
            if not bp: nt_set.add(b)
            if not cp: nt_set.add(c)
        for t, w in lex:
            pt_set.add(t)
            wc[w] += 1
    nt_names, pt_names = sorted(nt_set), sorted(pt_set)
    nt2i = {s: i for i, s in enumerate(nt_names)}
    pt2i = {s: i for i, s in enumerate(pt_names)}
    w2i = {'<unk>': 0}
    for w, c in wc.items():
        if c >= args.min_freq:
            w2i[w] = len(w2i)
    NT, PT, V = len(nt_names), len(pt_names), len(w2i)
    S = NT + PT
    print(f"|NT|={NT}  |PT|={PT}  |V|={V}  rule table entries={NT * S * S:,}")
    if NT * S * S > args.max_rule_entries:
        sys.exit("Rule table too large for memory; use --markov_order 0 or raise --max_rule_entries.")

    train_items = [encode(e, nt2i, pt2i, w2i) for e in ex_train]
    val_items = [encode(e, nt2i, pt2i, w2i) for e in ex_val]
    print(f"Val events skipped (symbol unseen in train): {sum(i['skipped'] for i in val_items)}")

    # ---- model + supervised MLE ---------------------------------------------
    model = SupNeuralPCFG(NT, PT, V, args.state_dim, args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val, best_epoch, bad = float('inf'), 0, 0
    best_path = out_dir / 'best.pt'
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        order = list(range(len(train_items)))
        random.shuffle(order)
        tot, cnt = 0.0, 0
        for i in range(0, len(order), args.batch_size):
            items = [train_items[j] for j in order[i:i + args.batch_size]]
            roots, rules, lex = collate(items, device)
            loss = -model.tree_loglik(roots, rules, lex) / len(items)    # per-tree NLL
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            tot += loss.item() * len(items)
            cnt += len(items)
        val_tree_nll, val_event_nll = mean_nll(model, val_items, device)
        print(f"Epoch {epoch:02d}/{args.epochs} | train NLL/tree {tot / cnt:8.3f} | "
              f"val NLL/tree {val_tree_nll:8.3f} (per event {val_event_nll:.4f}) | {time.time() - t0:.1f}s")
        if val_tree_nll < best_val:
            best_val, best_epoch, bad = val_tree_nll, epoch, 0
            torch.save({'model': model.state_dict(), 'nt': nt_names, 'pt': pt_names, 'w2i': w2i}, best_path)
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                print(f"Early stopping (best epoch {best_epoch}).")
                break

    model.load_state_dict(torch.load(best_path, map_location=device)['model'])
    model.eval()
    print(f"\nRestored best model: epoch {best_epoch}, val NLL/tree {best_val:.3f}")

    # ---- decode the test split with Viterbi CKY ------------------------------
    cky = SparseViterbiCKY(model, nt_names, pt_names, w2i, args.topk)
    pred_trees, fails = [], 0
    t0 = time.time()
    for i, gold in enumerate(test_t):
        pred = cky.parse(gold.leaves())
        fails += pred is None
        pred_trees.append(pred)
        if (i + 1) % 10 == 0 or i + 1 == len(test_t):
            print(f"Parsed {i + 1}/{len(test_t)} sentences ({time.time() - t0:.0f}s)", flush=True)

    # (a) Task 1 metric: PARSEVAL, corpus-level, punctuation kept
    l_p, l_r, l_f, u_p, u_r, u_f = parseval_evaluate(test_t, pred_trees)
    # (b) Task 2 metric: unlabelled, sentence-level, punctuation removed
    f1s = []
    for gold, pred in zip(test_t, pred_trees):
        words, gold_spans = extract_spans(gold)
        if len(words) < 2:
            continue
        f1s.append(span_f1(pred_spans_no_punct(pred, gold), gold_spans))
    t2_f1 = 100.0 * float(np.mean(f1s))

    results = {
        'markov_order': args.markov_order, 'NT': NT, 'PT': PT, 'vocab': V, 'params': n_params,
        'best_epoch': best_epoch, 'best_val_nll_per_tree': best_val,
        'topk': args.topk, 'parse_failures': fails,
        'task1_metric': {'labeled': {'P': l_p, 'R': l_r, 'F1': l_f},
                         'unlabeled': {'P': u_p, 'R': u_r, 'F1': u_f}},
        'task2_metric_unlabeled_sentence_F1_percent': t2_f1,
    }
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 66)
    print("Task 3 - Supervised Neural PCFG (test split)")
    print(f"PARSEVAL (Task 1 scorer)   labeled   P {l_p:.4f}  R {l_r:.4f}  F1 {l_f:.4f}")
    print(f"                           unlabeled P {u_p:.4f}  R {u_r:.4f}  F1 {u_f:.4f}")
    print(f"Task 2 scorer (sent-level, no punct, unlabeled) F1: {t2_f1:.2f}%")
    print(f"Parse failures (pruned grammar): {fails}/{len(test_t)}")
    print("=" * 66)


if __name__ == '__main__':
    main()
