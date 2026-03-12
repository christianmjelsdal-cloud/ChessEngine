#!/usr/bin/env python3
"""
PyTorch NNUE Trainer (CPU-Optimized, Streaming, Enhanced v3)
============================================================
Supports arbitrarily large datasets via streaming chunked loading.
No RAM limit - processes data in chunks without loading everything at once.

ENHANCEMENTS (v3 - adds rebalancing):
  All v2 features plus:
  11. Eval soft-capping - downweights extreme eval positions in loss
  12. Draw upweighting - increases loss contribution of drawn positions
  13. Combine with generate_draws.py for Syzygy-sourced endgame data

ENHANCEMENTS (v4 - multi-dataset support):
  14. Multiple training data sources with configurable sampling ratios
  15. Separate base data from self-play data for iterative training loops
  16. Per-source validation split and statistics

Usage:
    # Basic training (backward compatible):
    py -3.10 train_nnue.py --epochs 500 --batch-size 8192 --fresh --early-stop 80

    # Enhanced with rebalancing:
    py -3.10 train_nnue.py --epochs 500 --batch-size 8192 --early-stop 99999 \
        --lr 0.0003 --load-weights assets/nnue_weights.bin --enhanced

    # Multi-dataset (70% base + 30% self-play):
    py -3.10 train_nnue.py --enhanced --epochs 20 \
        --data assets/base_data.bin \
        --extra-data assets/selfplay_v1.bin 0.3

    # Multiple extra sources:
    py -3.10 train_nnue.py --enhanced --epochs 20 \
        --data assets/base_data.bin \
        --extra-data assets/selfplay_v1.bin 0.25 \
        --extra-data assets/endgame_draws.bin 0.10

    # Manual rebalancing knobs:
    py -3.10 train_nnue.py --eval-soft-cap 8.0 --draw-weight 3.0 ...

Requirements:
    pip install torch numpy matplotlib
"""

import argparse
import struct
import time
import sys
import os
import random
import csv
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# =============================================================================
# Constants - must match C++ NNUE.h exactly
# =============================================================================
NUM_FEATURES = 768
L1_SIZE = 512
L2_SIZE = 128
L3_SIZE = 128

# Datasets larger than this use streaming mode (saves RAM)
MAX_PRELOAD_POSITIONS = 8_000_000

# Chunk size for streaming mode
STREAM_CHUNK_SIZE = 500_000

# =============================================================================
# Feature mirror - must match C++ mirrorFeature() exactly
# =============================================================================
def build_mirror_table():
    table = [0] * NUM_FEATURES
    for feat in range(NUM_FEATURES):
        piece_index = feat // 64
        square_index = feat % 64
        rank = square_index // 8
        col = square_index % 8
        mirrored_piece = (piece_index + 6) if piece_index < 6 else (piece_index - 6)
        mirrored_rank = 7 - rank
        table[feat] = mirrored_piece * 64 + mirrored_rank * 8 + col
    return table

MIRROR_TABLE = build_mirror_table()

# =============================================================================
# Game Phase Classification
# =============================================================================
def compute_material_phase(features):
    phase = 0
    for feat in features:
        piece_offset = (feat % 384) // 64
        if piece_offset == 1:    phase += 1
        elif piece_offset == 2:  phase += 1
        elif piece_offset == 3:  phase += 2
        elif piece_offset == 4:  phase += 4
    return phase

def classify_phase(features):
    """0=Opening (>=20), 1=Middlegame (8-19), 2=Endgame (<8)"""
    phase = compute_material_phase(features)
    if phase >= 20: return 0
    if phase >= 8:  return 1
    return 2

# =============================================================================
# Sample Rebalancing Weights (NEW in v3)
# =============================================================================
def compute_sample_weights(search_eval, game_result, eval_soft_cap, draw_weight):
    """
    Per-sample loss weights to fix dataset imbalances:
      - eval_soft_cap: positions with |eval| > cap get weight = cap / |eval|
        (reduces dominance of trivially won/lost positions)
      - draw_weight: multiplier for drawn positions (result ≈ 0.5)
        (compensates for draw underrepresentation)
    Returns None if no rebalancing is needed (both disabled).
    """
    if eval_soft_cap <= 0 and draw_weight <= 1.0:
        return None

    weights = torch.ones_like(search_eval)

    # Soft-cap extreme evals: linear decay beyond threshold
    if eval_soft_cap > 0:
        abs_eval = torch.abs(search_eval)
        eval_w = torch.clamp(eval_soft_cap / torch.clamp(abs_eval, min=0.01), max=1.0)
        weights = weights * eval_w

    # Upweight draws
    if draw_weight > 1.0:
        is_draw = (game_result > 0.4) & (game_result < 0.6)
        weights = torch.where(is_draw, weights * draw_weight, weights)

    return weights

# =============================================================================
# Binary format reader - scan offsets and load positions
# =============================================================================
def scan_positions(filename):
    """Fast scan: builds list of byte offsets for random access without loading data."""
    print(f"Scanning {filename} for position offsets...")
    with open(filename, 'rb') as f:
        raw_header = f.read(4)
        num_positions = struct.unpack_from('<I', raw_header, 0)[0]
        print(f"  Total positions: {num_positions:,}")

        offsets = []
        offset = 4
        f.seek(offset)

        for i in range(num_positions):
            offsets.append(offset)
            header = f.read(2)
            if len(header) < 2:
                print(f"  Warning: truncated at position {i}")
                num_positions = i
                break
            num_features = struct.unpack_from('<H', header, 0)[0]
            skip = 2 * num_features + 1 + 8
            f.seek(skip, 1)
            offset += 2 + skip

            if (i + 1) % 1_000_000 == 0:
                print(f"  Scanned {i+1:,}/{num_positions:,}...")

    print(f"  Scan complete: {len(offsets):,} positions indexed.")
    return num_positions, offsets

def load_positions_at_offsets(filename, offsets, filter_eval_max=0.0):
    """
    Load positions by byte offsets using memory-mapped file.
    Offsets should be pre-sorted for sequential disk reads.
    If filter_eval_max > 0, positions with |eval| > threshold are excluded.
    """
    n = len(offsets)
    mirror = MIRROR_TABLE

    all_white  = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    all_black  = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    all_stm    = np.zeros(n, dtype=np.float32)
    all_result = np.zeros(n, dtype=np.float32)
    all_eval   = np.zeros(n, dtype=np.float32)
    all_phases = np.zeros(n, dtype=np.int32)

    import mmap
    with open(filename, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for i, off in enumerate(offsets):
            num_features = struct.unpack_from('<H', mm, off)[0]
            pos = off + 2
            features = struct.unpack_from(f'<{num_features}H', mm, pos)
            pos += 2 * num_features
            stm = mm[pos]
            pos += 1
            game_result, search_eval = struct.unpack_from('<ff', mm, pos)

            for feat in features:
                all_white[i, feat] = 1.0
                all_black[i, mirror[feat]] = 1.0

            all_stm[i]    = float(stm)
            all_result[i] = game_result
            all_eval[i]   = search_eval
            all_phases[i] = classify_phase(features)

        mm.close()

    # Filter extreme evaluations if requested
    if filter_eval_max > 0:
        mask = np.abs(all_eval) <= filter_eval_max
        kept = int(mask.sum())
        if kept < n:
            all_white  = all_white[mask]
            all_black  = all_black[mask]
            all_stm    = all_stm[mask]
            all_result = all_result[mask]
            all_eval   = all_eval[mask]
            all_phases = all_phases[mask]
            n = kept

    return {
        'white':  torch.from_numpy(all_white),
        'black':  torch.from_numpy(all_black),
        'stm':    torch.from_numpy(all_stm),
        'result': torch.from_numpy(all_result),
        'eval':   torch.from_numpy(all_eval),
        'phases': all_phases,
        'count':  n,
    }

# =============================================================================
# Fast preload (small datasets)
# =============================================================================
def load_training_data(filename, max_positions=0, filter_eval_max=0.0):
    """Load entire dataset into RAM. Only used for small datasets."""
    print(f"Loading positions from {filename}...")

    with open(filename, 'rb') as f:
        raw = f.read()

    num_positions = struct.unpack_from('<I', raw, 0)[0]
    print(f"  Total positions: {num_positions:,}")
    if max_positions > 0 and max_positions < num_positions:
        num_positions = max_positions
        print(f"  Capped to:       {num_positions:,} (--max-positions)")

    all_white  = np.zeros((num_positions, NUM_FEATURES), dtype=np.float32)
    all_black  = np.zeros((num_positions, NUM_FEATURES), dtype=np.float32)
    all_stm    = np.zeros(num_positions, dtype=np.float32)
    all_result = np.zeros(num_positions, dtype=np.float32)
    all_eval   = np.zeros(num_positions, dtype=np.float32)
    all_phases = np.zeros(num_positions, dtype=np.int32)

    mirror = MIRROR_TABLE
    offset = 4
    kept = 0

    try:
        for i in range(num_positions):
            num_features = struct.unpack_from('<H', raw, offset)[0]
            offset += 2
            features = struct.unpack_from(f'<{num_features}H', raw, offset)
            offset += 2 * num_features
            stm = raw[offset]
            offset += 1
            game_result, search_eval = struct.unpack_from('<ff', raw, offset)
            offset += 8

            # Filter extreme evaluations
            if filter_eval_max > 0 and abs(search_eval) > filter_eval_max:
                continue

            for feat in features:
                all_white[kept, feat] = 1.0
                all_black[kept, mirror[feat]] = 1.0

            all_stm[kept]    = float(stm)
            all_result[kept] = game_result
            all_eval[kept]   = search_eval
            all_phases[kept] = classify_phase(features)
            kept += 1

            if (i + 1) % 200000 == 0:
                print(f"  Loaded {i+1:,}/{num_positions:,}...")
    except KeyboardInterrupt:
        print("\nInterrupted during data loading - exiting cleanly.")
        sys.exit(0)

    if kept < num_positions:
        print(f"  Filtered: {num_positions - kept:,} extreme-eval positions removed, {kept:,} kept")
        all_white  = all_white[:kept]
        all_black  = all_black[:kept]
        all_stm    = all_stm[:kept]
        all_result = all_result[:kept]
        all_eval   = all_eval[:kept]
        all_phases = all_phases[:kept]
        num_positions = kept

    print(f"  Converting to tensors...")

    data = {
        'white':  torch.from_numpy(all_white),
        'black':  torch.from_numpy(all_black),
        'stm':    torch.from_numpy(all_stm),
        'result': torch.from_numpy(all_result),
        'eval':   torch.from_numpy(all_eval),
        'phases': all_phases,
        'count':  num_positions,
    }

    phase_counts = [0, 0, 0]
    for p in all_phases[:num_positions]:
        phase_counts[p] += 1
    print(f"Phase distribution: Opening={phase_counts[0]:,}, Middlegame={phase_counts[1]:,}, Endgame={phase_counts[2]:,}")

    return data

# =============================================================================
# Weight I/O - matches C++ binary format exactly
# FIXED: Uses architecture constants instead of hardcoded values
# =============================================================================
def save_weights_cpp(net, filename):
    with open(filename, 'wb') as f:
        f.write(net.l1_weight.detach().cpu().numpy().astype(np.float32).tobytes())
        f.write(net.l1_bias.detach().cpu().numpy().astype(np.float32).tobytes())
        f.write(net.l2.weight.detach().cpu().numpy().T.astype(np.float32).tobytes())
        f.write(net.l2.bias.detach().cpu().numpy().astype(np.float32).tobytes())
        f.write(net.l3.weight.detach().cpu().numpy().T.astype(np.float32).tobytes())
        f.write(net.l3.bias.detach().cpu().numpy().astype(np.float32).tobytes())
        f.write(net.output.weight.detach().cpu().numpy().flatten().astype(np.float32).tobytes())
        f.write(net.output.bias.detach().cpu().numpy().astype(np.float32).tobytes())
    print(f"Weights saved to {filename}")

def load_weights_cpp(net, filename):
    """Load weights from C++ binary format. Uses architecture constants for correct sizing."""
    with open(filename, 'rb') as f:
        # L1: [NUM_FEATURES, L1_SIZE]
        l1w = np.frombuffer(f.read(NUM_FEATURES * L1_SIZE * 4), dtype=np.float32).reshape(NUM_FEATURES, L1_SIZE)
        net.l1_weight.data = torch.from_numpy(l1w.copy())
        l1b = np.frombuffer(f.read(L1_SIZE * 4), dtype=np.float32).copy()
        net.l1_bias.data = torch.from_numpy(l1b)
        # L2: input is L1_SIZE*2 (white+black accumulators), output is L2_SIZE
        l2w = np.frombuffer(f.read(L1_SIZE * 2 * L2_SIZE * 4), dtype=np.float32).reshape(L1_SIZE * 2, L2_SIZE)
        net.l2.weight.data = torch.from_numpy(l2w.T.copy())
        l2b = np.frombuffer(f.read(L2_SIZE * 4), dtype=np.float32).copy()
        net.l2.bias.data = torch.from_numpy(l2b)
        # L3: [L2_SIZE, L3_SIZE]
        l3w = np.frombuffer(f.read(L2_SIZE * L3_SIZE * 4), dtype=np.float32).reshape(L2_SIZE, L3_SIZE)
        net.l3.weight.data = torch.from_numpy(l3w.T.copy())
        l3b = np.frombuffer(f.read(L3_SIZE * 4), dtype=np.float32).copy()
        net.l3.bias.data = torch.from_numpy(l3b)
        # Output: [1, L3_SIZE]
        ow = np.frombuffer(f.read(L3_SIZE * 4), dtype=np.float32).reshape(1, L3_SIZE)
        net.output.weight.data = torch.from_numpy(ow.copy())
        ob = np.frombuffer(f.read(4), dtype=np.float32).copy()
        net.output.bias.data = torch.from_numpy(ob)
    print(f"Weights loaded from {filename}")

# =============================================================================
# NNUE Network
# =============================================================================
class SCReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0) ** 2

class NNUENetwork(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.l1_weight = nn.Parameter(torch.zeros(NUM_FEATURES, L1_SIZE))
        self.l1_bias   = nn.Parameter(torch.zeros(L1_SIZE))
        self.l2        = nn.Linear(L1_SIZE * 2, L2_SIZE)
        self.l3        = nn.Linear(L2_SIZE, L3_SIZE)
        self.output    = nn.Linear(L3_SIZE, 1)
        self.screlu    = SCReLU()
        self.drop2     = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.drop3     = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        avg_active = 30.0
        nn.init.normal_(self.l1_weight, 0, 1.0 / avg_active)
        nn.init.zeros_(self.l1_bias)
        nn.init.kaiming_normal_(self.l2.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.l2.bias)
        nn.init.kaiming_normal_(self.l3.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.l3.bias)
        nn.init.kaiming_normal_(self.output.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.output.bias)

    def forward(self, white_features, black_features, stm):
        white_acc = torch.mm(white_features, self.l1_weight) + self.l1_bias
        black_acc = torch.mm(black_features, self.l1_weight) + self.l1_bias
        white_acc = self.screlu(white_acc)
        black_acc = self.screlu(black_acc)

        stm_mask = stm.unsqueeze(1)
        stm_acc  = white_acc * (1 - stm_mask) + black_acc * stm_mask
        opp_acc  = black_acc * (1 - stm_mask) + white_acc * stm_mask
        l1_out   = torch.cat([stm_acc, opp_acc], dim=1)

        l2_out = self.drop2(self.screlu(self.l2(l1_out)))
        l3_out = self.drop3(self.screlu(self.l3(l2_out)))
        return self.output(l3_out)

# =============================================================================
# Stochastic Weight Averaging (manual, compatible with any PyTorch version)
# =============================================================================
class ManualSWA:
    """Accumulates weight snapshots and averages them for better generalization."""
    def __init__(self):
        self.sum_state = None
        self.count = 0

    def update(self, model):
        """Add current model weights to the running sum."""
        state = model.state_dict()
        if self.sum_state is None:
            self.sum_state = {k: v.clone().float() for k, v in state.items()}
        else:
            for k in self.sum_state:
                self.sum_state[k] += state[k].float()
        self.count += 1

    def apply(self, model):
        """Apply averaged weights to the model."""
        if self.sum_state is not None and self.count > 0:
            avg = {k: (v / self.count) for k, v in self.sum_state.items()}
            model.load_state_dict(avg)
            print(f"  SWA: Applied averaged weights ({self.count} snapshots)")
            return True
        return False

# =============================================================================
# Loss Function (with label smoothing support)
# =============================================================================
def compute_loss(net, white, black, stm, game_result, search_eval,
                 lam, eval_scale, label_smoothing=0.0):
    raw_output     = net(white, black, stm)
    predicted      = raw_output.squeeze(1) * 400.0
    predicted_white = torch.where(stm < 0.5, predicted, -predicted)
    sig_pred       = torch.sigmoid(predicted_white / eval_scale)
    sig_target     = torch.sigmoid(search_eval / eval_scale)

    # Label smoothing: soften game results toward 0.5
    if label_smoothing > 0:
        game_result = game_result * (1.0 - label_smoothing) + 0.5 * label_smoothing

    eval_loss   = (sig_pred - sig_target) ** 2
    result_loss = (sig_pred - game_result) ** 2
    return lam * eval_loss + (1.0 - lam) * result_loss

# =============================================================================
# Training Log & Progress Graph
# =============================================================================
LOG_FILE  = 'training_log.csv'
PLOT_FILE = 'training_progress.png'

def append_log(epoch, train_loss, val_loss, lr, epoch_time, run_id, phase_losses=None):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'run_id', 'epoch', 'loss', 'val_loss',\
                             'lr', 'epoch_time_s',\
                             'opening_loss', 'middlegame_loss', 'endgame_loss'])
        pl = phase_losses or {}
        writer.writerow([datetime.now().isoformat(), run_id, epoch,\
                         f'{train_loss:.8f}', f'{val_loss:.8f}',\
                         f'{lr:.8f}', f'{epoch_time:.2f}',\
                         f'{pl.get("opening", 0):.8f}',\
                         f'{pl.get("middlegame", 0):.8f}',\
                         f'{pl.get("endgame", 0):.8f}'])

def read_log():
    if not os.path.exists(LOG_FILE):
        return []
    rows = []
    with open(LOG_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def generate_plot(log_data=None, show=False):
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed - skipping plot.")
        return
    if log_data is None:
        log_data = read_log()
    if not log_data:
        print("No training log data to plot.")
        return

    epochs    = list(range(1, len(log_data) + 1))
    losses    = [float(row['loss']) for row in log_data]
    lrs       = [float(row['lr']) for row in log_data]
    val_losses     = [float(row.get('val_loss', 0)) for row in log_data]
    has_val        = any(v > 0 for v in val_losses)
    epoch_times    = [float(row.get('epoch_time_s', 0)) for row in log_data]
    has_times      = any(t > 0 for t in epoch_times)
    opening_losses    = [float(row.get('opening_loss', 0)) for row in log_data]
    middlegame_losses = [float(row.get('middlegame_loss', 0)) for row in log_data]
    endgame_losses    = [float(row.get('endgame_loss', 0)) for row in log_data]
    has_phases = any(v > 0 for v in opening_losses)

    run_ids     = [row.get('run_id', '?') for row in log_data]
    unique_runs = []
    for r in run_ids:
        if r not in unique_runs:
            unique_runs.append(r)

    n_plots = 2
    if has_phases: n_plots += 1
    if has_times:  n_plots += 1

    ratios = [3]
    if has_phases: ratios.append(2)
    if has_times:  ratios.append(1)
    ratios.append(1)

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 + 3 * n_plots), sharex=True,
                              gridspec_kw={'height_ratios': ratios})
    if n_plots == 1:
        axes = [axes]
    fig.suptitle('NNUE Training Progress', fontsize=14, fontweight='bold')

    ax_idx = 0

    ax1 = axes[ax_idx]; ax_idx += 1
    ax1.plot(epochs, losses, '-', color='#2196F3', linewidth=1.2, label='Train Loss')
    if has_val:
        ax1.plot(epochs, val_losses, '-', color='#E91E63', linewidth=1.2, alpha=0.9, label='Val Loss')
        for i in range(len(epochs)):
            if val_losses[i] > 0 and val_losses[i] > losses[i] * 1.05:
                ax1.axvspan(epochs[i] - 0.5, epochs[i] + 0.5, alpha=0.08, color='red', linewidth=0)

    if len(unique_runs) > 1:
        for run_id in unique_runs[1:]:
            first_epoch = next(e for e, r in zip(epochs, run_ids) if r == run_id)
            ax1.axvline(x=first_epoch, color='gray', linestyle=':', alpha=0.4, linewidth=1)
        ax1.axvline(x=-1, color='gray', linestyle=':', alpha=0.4, linewidth=1,
                     label=f'Run boundary ({len(unique_runs)} runs)')

    if len(losses) >= 5:
        window   = max(3, min(15, len(losses) // 7))
        half     = window // 2
        smoothed = []
        for i in range(len(losses)):
            lo = max(0, i - half)
            hi = min(len(losses), i + half + 1)
            smoothed.append(sum(losses[lo:hi]) / (hi - lo))
        ax1.plot(epochs, smoothed, '--', color='#FF5722', linewidth=2, alpha=0.7,
                 label=f'Trend (avg x{window})')

    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='upper right')

    best_idx = losses.index(min(losses))
    ax1.annotate(f'Best: {losses[best_idx]:.6f}\n(epoch {epochs[best_idx]})',
                 xy=(epochs[best_idx], losses[best_idx]),
                 xytext=(30, 30), textcoords='offset points',
                 fontsize=9, color='green',
                 arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    stats_lines = [f'Total epochs: {len(epochs)}', f'Best train loss: {min(losses):.6f}']
    if has_val:
        valid_vals = [v for v in val_losses if v > 0]
        if valid_vals:
            stats_lines.append(f'Best val loss: {min(valid_vals):.6f}')
            latest_ratio = val_losses[-1] / losses[-1] if losses[-1] > 0 and val_losses[-1] > 0 else 0
            if latest_ratio > 1.05:
                stats_lines.append(f'! Overfit ratio: {latest_ratio:.2f}x')
            else:
                stats_lines.append(f'No overfitting ({latest_ratio:.2f}x)')
    stats_lines.append(f'Runs: {len(unique_runs)}')
    ax1.text(0.02, 0.02, '\n'.join(stats_lines), transform=ax1.transAxes, fontsize=9,
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    if has_phases:
        ax_ph = axes[ax_idx]; ax_idx += 1
        ax_ph.plot(epochs, opening_losses,    '-', color='#4CAF50', linewidth=1.2, alpha=0.9, label='Opening')
        ax_ph.plot(epochs, middlegame_losses, '-', color='#FF9800', linewidth=1.2, alpha=0.9, label='Middlegame')
        ax_ph.plot(epochs, endgame_losses,    '-', color='#F44336', linewidth=1.2, alpha=0.9, label='Endgame')
        ax_ph.set_ylabel('Phase Loss', fontsize=11)
        ax_ph.set_yscale('log')
        ax_ph.grid(True, alpha=0.3)
        ax_ph.legend(fontsize=9, loc='upper right')

    if has_times:
        ax_t = axes[ax_idx]; ax_idx += 1
        ax_t.plot(epochs, epoch_times, '-', color='#607D8B', linewidth=1, alpha=0.8)
        ax_t.fill_between(epochs, 0, epoch_times, alpha=0.15, color='#607D8B')
        ax_t.set_ylabel('Epoch (s)', fontsize=11)
        ax_t.grid(True, alpha=0.3)
        if epoch_times:
            avg_time = sum(epoch_times) / len(epoch_times)
            ax_t.axhline(y=avg_time, color='#607D8B', linestyle='--', alpha=0.5)
            ax_t.text(0.98, 0.85, f'avg {avg_time:.1f}s', transform=ax_t.transAxes,
                     fontsize=9, ha='right', color='#607D8B')

    ax_lr = axes[ax_idx]
    ax_lr.plot(epochs, lrs, '-', color='#9C27B0', linewidth=1)
    ax_lr.set_xlabel('Epoch', fontsize=11)
    ax_lr.set_ylabel('LR', fontsize=11)
    ax_lr.set_yscale('log')
    ax_lr.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches='tight')
    print(f"Progress graph saved to {PLOT_FILE}")
    plt.close()

    if show:
        import subprocess
        if sys.platform == 'win32':
            subprocess.Popen(['start', '', PLOT_FILE], shell=True)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', PLOT_FILE])
        else:
            subprocess.Popen(['xdg-open', PLOT_FILE])

# =============================================================================
# Utility
# =============================================================================
# =============================================================================
# Multi-Dataset Preparation
# =============================================================================
def prepare_datasets(primary_path, extra_data_args, max_positions=0):
    """
    Build a list of dataset descriptors from --data and --extra-data args.
    Each descriptor contains: path, label, ratio, num_positions.
    Ratios are normalised so they sum to 1.0.

    Returns:
        datasets: list of dicts with keys: path, label, ratio, num_positions
        total_positions: total available positions across all datasets
    """
    datasets = []

    # Primary dataset
    with open(primary_path, 'rb') as f:
        primary_count = struct.unpack('<I', f.read(4))[0]
    datasets.append({
        'path': primary_path,
        'label': os.path.basename(primary_path),
        'ratio': None,  # filled in below
        'num_positions': primary_count,
    })

    # Extra datasets
    extra_total_ratio = 0.0
    if extra_data_args:
        for filepath, ratio_str in extra_data_args:
            ratio = float(ratio_str)
            if ratio <= 0 or ratio >= 1.0:
                print(f"WARNING: --extra-data ratio {ratio} for {filepath} must be in (0, 1). Skipping.")
                continue
            if not os.path.exists(filepath):
                print(f"WARNING: --extra-data file '{filepath}' not found. Skipping.")
                continue
            with open(filepath, 'rb') as f:
                count = struct.unpack('<I', f.read(4))[0]
            extra_total_ratio += ratio
            datasets.append({
                'path': filepath,
                'label': os.path.basename(filepath),
                'ratio': ratio,
                'num_positions': count,
            })

    if extra_total_ratio >= 1.0:
        print(f"WARNING: Extra dataset ratios sum to {extra_total_ratio:.2f} (>=1.0). Normalising.")
        for ds in datasets[1:]:
            ds['ratio'] = ds['ratio'] / (extra_total_ratio + 0.01)
        extra_total_ratio = sum(ds['ratio'] for ds in datasets[1:])

    # Primary gets the remainder
    datasets[0]['ratio'] = 1.0 - extra_total_ratio

    # Apply max_positions cap proportionally
    total_available = sum(ds['num_positions'] for ds in datasets)
    if max_positions > 0 and max_positions < total_available:
        scale = max_positions / total_available
        for ds in datasets:
            ds['num_positions'] = int(ds['num_positions'] * scale)
        total_available = sum(ds['num_positions'] for ds in datasets)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Dataset Configuration ({len(datasets)} source{'s' if len(datasets) > 1 else ''}):")
    for i, ds in enumerate(datasets):
        tag = "PRIMARY" if i == 0 else f"EXTRA #{i}"
        print(f"  [{tag}] {ds['label']}")
        print(f"    Positions: {ds['num_positions']:,}  |  Ratio: {ds['ratio']:.1%}")
    print(f"  Total positions: {total_available:,}")
    print(f"{'='*60}\n")

    return datasets, total_available


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m{s:02d}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"

# =============================================================================
# Batch Processing Helper (v3: supports sample weights)
# =============================================================================
def process_batch(net, optimizer, white, black, stm, result, eval_t,
                  lam, eval_scale, label_smoothing, accum_steps, grad_clip,
                  eval_soft_cap=0.0, draw_weight=1.0):
    """Process one mini-batch with gradient accumulation and sample rebalancing."""
    loss_tensor = compute_loss(net, white, black, stm, result, eval_t,
                               lam, eval_scale, label_smoothing)

    # Apply per-sample rebalancing weights
    sample_weights = compute_sample_weights(eval_t, result, eval_soft_cap, draw_weight)
    if sample_weights is not None:
        loss_tensor = loss_tensor * sample_weights

    loss = loss_tensor.mean()
    scaled_loss = loss / accum_steps
    scaled_loss.backward()
    return loss.item()

# =============================================================================
# Training Loop
# =============================================================================
def train(args):
    device = torch.device('cpu')
    print(f"Using device: CPU (optimized)")

    if hasattr(torch, 'set_num_threads'):
        num_cores = os.cpu_count() or 4
        torch.set_num_threads(num_cores)
        print(f"  CPU threads: {num_cores}")

    # --- Determine dataset size (multi-dataset aware) ---
    datasets, num_positions = prepare_datasets(
        args.data,
        getattr(args, 'extra_data', None),
        max_positions=args.max_positions
    )
    multi_dataset = len(datasets) > 1

    # Force streaming if dense arrays would exceed ~4 GB RAM
    estimated_ram_gb = (num_positions * NUM_FEATURES * 4 * 2) / (1024**3)  # white+black arrays
    streaming_mode = num_positions > MAX_PRELOAD_POSITIONS or estimated_ram_gb > 4.0
    if streaming_mode:
        print(f"  Mode: STREAMING (dataset too large for RAM preload)")
        print(f"  Chunk size: {STREAM_CHUNK_SIZE:,} positions per chunk")
    else:
        print(f"  Mode: PRELOAD (full dataset fits in RAM)")

    lam            = args.lam
    eval_scale     = args.eval_scale
    batch_size     = args.batch_size
    accum_steps    = args.grad_accum
    effective_batch = batch_size * accum_steps
    label_smoothing = args.label_smoothing

    # --- Create network ---
    net = NNUENetwork(dropout=args.dropout)

    weights_path    = args.load_weights or args.output
    checkpoint_path = os.path.splitext(args.output)[0] + '_checkpoint.pt'

    if not args.fresh and os.path.exists(weights_path):
        try:
            load_weights_cpp(net, weights_path)
            print(f"Loaded existing weights from {weights_path}")
        except Exception as e:
            print(f"Could not load weights (architecture changed?): {e}")
            print(f"Starting with fresh random weights.")
    elif args.fresh:
        print("Starting with fresh random weights (--fresh).")

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=args.weight_decay)

    # --- LR Scheduler ---
    if args.cosine_restarts:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.cosine_t0,
            T_mult=args.cosine_t_mult,
            eta_min=args.lr_min
        )
        schedule_name = f"Cosine Warm Restarts (T0={args.cosine_t0}, Tmult={args.cosine_t_mult})"
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr_min
        )
        schedule_name = f"Cosine Decay (T_max={args.epochs})"

    # --- Load optimizer checkpoint ---
    if not args.fresh and not args.load_weights and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            print(f"Resumed optimizer state (LR: {scheduler.get_last_lr()[0]:.6f})")
        except Exception as e:
            print(f"Could not load checkpoint (starting fresh optimizer): {e}")
    elif args.load_weights:
        print(f"  Fresh optimizer (--load-weights skips old optimizer state)")

    # --- Print configuration ---
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Positions:         {num_positions:,}")
    print(f"  Batch size:        {batch_size} (effective: {effective_batch} with grad accum x{accum_steps})")
    print(f"  Epochs:            {args.epochs}")
    print(f"  Learning rate:     {args.lr}")
    print(f"  LR schedule:       {schedule_name}")
    print(f"  LR min:            {args.lr_min}")
    print(f"  LR warmup:         {args.warmup_steps} optimizer steps" if args.warmup_steps > 0 else "  LR warmup:         OFF")
    print(f"  Lambda (eval/res): {lam}")
    print(f"  Eval scale:        {eval_scale}")
    print(f"  Phase balanced:    {args.phase_balanced}")
    print(f"  Early stop:        {args.early_stop} epochs")
    print(f"  Weight decay:      {args.weight_decay}")
    print(f"  Dropout:           {args.dropout}")
    print(f"  Grad accumulation: {accum_steps}x")
    print(f"  Label smoothing:   {label_smoothing}" if label_smoothing > 0 else "  Label smoothing:   OFF")
    swa_str = f"ON (start epoch {args.swa_start})" if args.swa else "OFF"
    print(f"  SWA:               {swa_str}")
    filter_str = f"{args.filter_eval_max:.1f}" if args.filter_eval_max > 0 else "OFF"
    print(f"  Filter eval max:   {filter_str}")
    soft_cap_str = f"{args.eval_soft_cap:.1f}" if args.eval_soft_cap > 0 else "OFF"
    print(f"  Eval soft-cap:     {soft_cap_str}")
    draw_w_str = f"{args.draw_weight:.1f}x" if args.draw_weight > 1.0 else "OFF"
    print(f"  Draw weight:       {draw_w_str}")
    print(f"  Async prefetch:    {'ON' if streaming_mode else 'N/A (preload mode)'}")
    print(f"  Activation:        SCReLU (squared clipped ReLU)")
    print(f"  Architecture:      768->{L1_SIZE}->{L2_SIZE}->{L3_SIZE}->1")
    print(f"  Grad clip norm:    {args.grad_clip}")
    print(f"  Streaming:         {streaming_mode}")
    if multi_dataset:
        print(f"  Datasets:          {len(datasets)} sources (ratio-based sampling)")
        for ds in datasets:
            print(f"    - {ds['label']:30s}  ratio={ds['ratio']:.1%}  pos={ds['num_positions']:,}")
    print(f"{'='*60}\n")

    # --- Setup validation set (multi-dataset aware) ---
    rng = random.Random(42)

    if streaming_mode:
        # Scan offsets for each dataset, split train/val per source
        all_train_sources = []  # list of (path, offsets_list) per dataset
        all_val_offsets   = []  # combined val offsets with file paths

        for ds in datasets:
            ds_count, ds_offsets = scan_positions(ds['path'])
            # Cap to configured num_positions for this source
            ds_offsets = ds_offsets[:ds['num_positions']]

            # 10% val, capped proportionally
            ds_indices = list(range(len(ds_offsets)))
            rng.shuffle(ds_indices)
            ds_val_size = max(1, len(ds_indices) // 10)
            ds_val_size = min(ds_val_size, int(500_000 * ds['ratio']))

            ds_val_idx   = ds_indices[:ds_val_size]
            ds_train_idx = ds_indices[ds_val_size:]

            ds_val_offsets   = [ds_offsets[i] for i in ds_val_idx]
            ds_train_offsets = [ds_offsets[i] for i in ds_train_idx]

            all_train_sources.append({
                'path': ds['path'],
                'label': ds['label'],
                'ratio': ds['ratio'],
                'offsets': ds_train_offsets,
            })
            all_val_offsets.extend([(ds['path'], off) for off in ds_val_offsets])

            print(f"  [{ds['label']}] train: {len(ds_train_offsets):,}  val: {ds_val_size:,}")

        # Store for backward compat variables
        train_offsets = None  # Not used directly anymore
        total_train = sum(len(s['offsets']) for s in all_train_sources)
        val_size = len(all_val_offsets)

        # Load combined validation set
        print(f"\n  Loading validation set ({val_size:,} positions from {len(datasets)} source(s))...")
        # Group val offsets by file for efficient loading
        val_by_file = {}
        for path, off in all_val_offsets:
            val_by_file.setdefault(path, []).append(off)
        val_parts = []
        for path, offsets in val_by_file.items():
            part = load_positions_at_offsets(path, sorted(offsets), filter_eval_max=0.0)
            val_parts.append(part)

        # Merge val parts
        if len(val_parts) == 1:
            val_data = val_parts[0]
        else:
            val_data = {
                'white':  torch.cat([p['white']  for p in val_parts]),
                'black':  torch.cat([p['black']  for p in val_parts]),
                'stm':    torch.cat([p['stm']    for p in val_parts]),
                'result': torch.cat([p['result'] for p in val_parts]),
                'eval':   torch.cat([p['eval']   for p in val_parts]),
                'phases': np.concatenate([p['phases'] for p in val_parts]),
                'count':  sum(p['count'] for p in val_parts),
            }
        del val_parts

        print(f"  Total train: {total_train:,}  |  Total val: {val_size:,}")

    else:
        # PRELOAD MODE: load each dataset and combine
        all_data_parts = []
        for ds in datasets:
            part = load_training_data(ds['path'], max_positions=ds['num_positions'],
                                      filter_eval_max=args.filter_eval_max)
            all_data_parts.append(part)
            print(f"  [{ds['label']}] loaded: {part['count']:,} positions")

        # Merge all parts
        if len(all_data_parts) == 1:
            data = all_data_parts[0]
        else:
            data = {
                'white':  torch.cat([p['white']  for p in all_data_parts]),
                'black':  torch.cat([p['black']  for p in all_data_parts]),
                'stm':    torch.cat([p['stm']    for p in all_data_parts]),
                'result': torch.cat([p['result'] for p in all_data_parts]),
                'eval':   torch.cat([p['eval']   for p in all_data_parts]),
                'phases': np.concatenate([p['phases'] for p in all_data_parts]),
                'count':  sum(p['count'] for p in all_data_parts),
            }
        del all_data_parts

        num_positions = data['count']
        all_indices = list(range(num_positions))
        rng.shuffle(all_indices)
        val_size     = max(1, num_positions // 10)
        val_size     = min(val_size, 500_000)
        val_indices  = all_indices[:val_size]
        train_indices = all_indices[val_size:]
        print(f"  Train positions: {len(train_indices):,}")
        print(f"  Val positions:   {val_size:,}")

        val_data = {
            'white':  data['white'][val_indices],
            'black':  data['black'][val_indices],
            'stm':    data['stm'][val_indices],
            'result': data['result'][val_indices],
            'eval':   data['eval'][val_indices],
            'phases': data['phases'][list(val_indices)],
        }

    # Build validation phase indices
    val_phase_indices = {0: [], 1: [], 2: []}
    for i, p in enumerate(val_data['phases']):
        val_phase_indices[p].append(i)

    val_white  = val_data['white']
    val_black  = val_data['black']
    val_stm    = val_data['stm']
    val_result = val_data['result']
    val_eval   = val_data['eval']

    best_loss         = float('inf')
    epochs_no_improve = 0
    start_time        = time.time()
    run_id            = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_log_data      = read_log()
    has_phase_data    = any(len(v) > 0 for v in val_phase_indices.values())

    # SWA setup
    swa = ManualSWA() if args.swa else None

    # Global optimizer step counter (for warmup)
    global_step  = 0
    warmup_steps = args.warmup_steps

    # Pre-build phase-balanced structure for preload mode
    if not streaming_mode and args.phase_balanced:
        train_phase_lists = {0: [], 1: [], 2: []}
        for local_i, orig_i in enumerate(train_indices):
            p = data['phases'][orig_i]
            train_phase_lists[p].append(orig_i)
        counts_p = {k: len(v) for k, v in train_phase_lists.items()}
        if all(c > 0 for c in counts_p.values()):
            max_c = max(counts_p.values())
            cached_balanced = []
            for phase in [0, 1, 2]:
                src    = train_phase_lists[phase]
                target = min(max_c, len(src) * 5)
                cached_balanced.extend([src[i % len(src)] for i in range(target)])
        else:
            cached_balanced = train_indices
        print(f"  Phase-balanced samples: {len(cached_balanced):,} (from {len(train_indices):,} train)")

    print(f"  Mode: eager (standard PyTorch)")

    # --- Smart ETA state ---
    ema_epoch_time  = None
    epoch_time_hist = []
    EMA_ALPHA       = 0.3
    TREND_WINDOW    = 5

    def smart_eta(hist, ema, epochs_left):
        """Returns (predicted_next_epoch_s, total_eta_s) using EMA + linear trend."""
        import numpy as np
        n = len(hist)
        if n == 0 or ema is None:
            return 0.0, 0.0
        trend_per_epoch = 0.0
        if n >= TREND_WINDOW:
            window = hist[-TREND_WINDOW:]
            xs = list(range(TREND_WINDOW))
            slope = np.polyfit(xs, window, 1)[0]
            trend_per_epoch = min(slope, 0.0)
        next_epoch = max(ema + trend_per_epoch, ema * 0.5)
        total = 0.0
        predicted = next_epoch
        for _ in range(epochs_left):
            total += predicted
            predicted = max(predicted + trend_per_epoch, predicted * 0.5)
        return next_epoch, total

    # --- Async prefetch helper for streaming mode ---
    def load_chunk_async(filename, offsets_slice, filt_max):
        sorted_slice = sorted(offsets_slice)
        return load_positions_at_offsets(filename, sorted_slice, filter_eval_max=filt_max)

    try:
        for epoch in range(args.epochs):
            epoch_start = time.time()
            net.train()

            total_loss   = 0.0
            num_batches  = 0
            accum_count  = 0
            optimizer.zero_grad()

            if streaming_mode:
                # === STREAMING MODE with async prefetching (multi-dataset) ===
                # Sample offsets from each source according to ratios
                combined_tagged_offsets = []  # list of (filepath, offset)
                for src in all_train_sources:
                    src_offsets = src['offsets'].copy()
                    random.shuffle(src_offsets)
                    # Sample ratio * total_train positions from this source
                    n_sample = int(src['ratio'] * total_train)
                    if n_sample >= len(src_offsets):
                        # Oversample with replacement if needed
                        sampled = random.choices(src_offsets, k=n_sample)
                    else:
                        sampled = src_offsets[:n_sample]
                    for off in sampled:
                        combined_tagged_offsets.append((src['path'], off))

                random.shuffle(combined_tagged_offsets)

                chunk_size = STREAM_CHUNK_SIZE
                num_chunks = (len(combined_tagged_offsets) + chunk_size - 1) // chunk_size

                # Pre-split all chunks, grouping by file within each chunk for efficient I/O
                chunk_slices = []
                for ci in range(num_chunks):
                    start_idx = ci * chunk_size
                    end_idx   = min(start_idx + chunk_size, len(combined_tagged_offsets))
                    chunk_slices.append(combined_tagged_offsets[start_idx:end_idx])

                def load_multi_chunk_async(tagged_offsets, filt_max):
                    """Load a chunk that may span multiple files."""
                    by_file = {}
                    for path, off in tagged_offsets:
                        by_file.setdefault(path, []).append(off)
                    parts = []
                    for path, offsets in by_file.items():
                        part = load_positions_at_offsets(path, sorted(offsets), filter_eval_max=filt_max)
                        parts.append(part)
                    if len(parts) == 1:
                        return parts[0]
                    return {
                        'white':  torch.cat([p['white']  for p in parts]),
                        'black':  torch.cat([p['black']  for p in parts]),
                        'stm':    torch.cat([p['stm']    for p in parts]),
                        'result': torch.cat([p['result'] for p in parts]),
                        'eval':   torch.cat([p['eval']   for p in parts]),
                        'phases': np.concatenate([p['phases'] for p in parts]),
                        'count':  sum(p['count'] for p in parts),
                    }

                # Start async prefetch of first chunk
                executor = ThreadPoolExecutor(max_workers=1)
                next_future = executor.submit(load_multi_chunk_async,
                                              chunk_slices[0], args.filter_eval_max)

                chunk_times = []
                for chunk_i in range(num_chunks):
                    chunk_start_time = time.time()

                    # Wait for current chunk to finish loading
                    chunk_data = next_future.result()

                    # Immediately start loading next chunk (overlaps with training)
                    if chunk_i + 1 < num_chunks:
                        next_future = executor.submit(load_multi_chunk_async,
                                                      chunk_slices[chunk_i + 1],
                                                      args.filter_eval_max)

                    white_all  = chunk_data['white']
                    black_all  = chunk_data['black']
                    stm_all    = chunk_data['stm']
                    result_all = chunk_data['result']
                    eval_all   = chunk_data['eval']
                    chunk_n    = chunk_data['count']

                    chunk_indices = list(range(chunk_n))
                    random.shuffle(chunk_indices)
                    idx_tensor = torch.tensor(chunk_indices, dtype=torch.long)

                    for batch_start in range(0, chunk_n, batch_size):
                        batch_idx = idx_tensor[batch_start:batch_start + batch_size]

                        loss_val = process_batch(
                            net, optimizer,
                            white_all[batch_idx], black_all[batch_idx],
                            stm_all[batch_idx], result_all[batch_idx],
                            eval_all[batch_idx],
                            lam, eval_scale, label_smoothing,
                            accum_steps, args.grad_clip,
                            eval_soft_cap=args.eval_soft_cap,
                            draw_weight=args.draw_weight,
                        )
                        total_loss  += loss_val
                        num_batches += 1
                        accum_count += 1

                        if accum_count >= accum_steps:
                            if args.grad_clip > 0:
                                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                            optimizer.step()
                            optimizer.zero_grad()
                            accum_count = 0
                            global_step += 1

                            # LR warmup (linear ramp)
                            if warmup_steps > 0 and global_step <= warmup_steps:
                                warmup_lr = args.lr * global_step / warmup_steps
                                for pg in optimizer.param_groups:
                                    pg['lr'] = warmup_lr

                    # Free chunk memory
                    del chunk_data, white_all, black_all, stm_all, result_all, eval_all

                    chunk_elapsed = time.time() - chunk_start_time
                    chunk_times.append(chunk_elapsed)
                    if len(chunk_times) == 1:
                        ema_chunk = chunk_elapsed
                    else:
                        ema_chunk = chunk_times[0]
                        for ct in chunk_times[1:]:
                            ema_chunk = 0.3 * ct + 0.7 * ema_chunk
                    remaining_chunks = num_chunks - (chunk_i + 1)
                    chunk_eta = ema_chunk * remaining_chunks

                    if num_chunks > 1:
                        eta_str = f"{int(chunk_eta//60)}m{int(chunk_eta%60):02d}s" if chunk_eta >= 60 else f"{chunk_eta:.0f}s"
                        print(f"  Epoch {epoch+1} chunk {chunk_i+1}/{num_chunks} | "
                              f"{chunk_elapsed:.1f}s (ema {ema_chunk:.1f}s) | "
                              f"Chunk ETA: {eta_str}    ", end='\r')

                executor.shutdown(wait=False)

            else:
                # === PRELOAD MODE ===
                if args.phase_balanced:
                    indices = cached_balanced.copy()
                    random.shuffle(indices)
                else:
                    indices = train_indices.copy()
                    random.shuffle(indices)

                idx_tensor  = torch.tensor(indices, dtype=torch.long)
                num_samples = len(indices)

                white_all  = data['white']
                black_all  = data['black']
                stm_all    = data['stm']
                result_all = data['result']
                eval_all   = data['eval']

                for batch_start in range(0, num_samples, batch_size):
                    batch_idx = idx_tensor[batch_start:batch_start + batch_size]

                    loss_val = process_batch(
                        net, optimizer,
                        white_all[batch_idx], black_all[batch_idx],
                        stm_all[batch_idx], result_all[batch_idx],
                        eval_all[batch_idx],
                        lam, eval_scale, label_smoothing,
                        accum_steps, args.grad_clip,
                        eval_soft_cap=args.eval_soft_cap,
                        draw_weight=args.draw_weight,
                    )
                    total_loss  += loss_val
                    num_batches += 1
                    accum_count += 1

                    if accum_count >= accum_steps:
                        if args.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                        optimizer.step()
                        optimizer.zero_grad()
                        accum_count = 0
                        global_step += 1

                        # LR warmup
                        if warmup_steps > 0 and global_step <= warmup_steps:
                            warmup_lr = args.lr * global_step / warmup_steps
                            for pg in optimizer.param_groups:
                                pg['lr'] = warmup_lr

            # Flush remaining accumulated gradients
            if accum_count > 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0

            # Step the LR scheduler (only after warmup is complete)
            if warmup_steps == 0 or global_step >= warmup_steps:
                scheduler.step()

            avg_loss   = total_loss / max(num_batches, 1)
            epoch_time = time.time() - epoch_start
            elapsed    = time.time() - start_time

            # --- Validation (no label smoothing, no rebalancing for true loss) ---
            net.eval()
            with torch.no_grad():
                val_loss_tensor = compute_loss(
                    net, val_white, val_black, val_stm,
                    val_result, val_eval, lam, eval_scale,
                    label_smoothing=0.0
                )
                val_loss = val_loss_tensor.mean().item()

                phase_losses = {}
                phase_names  = {0: 'opening', 1: 'middlegame', 2: 'endgame'}
                for phase_id, phase_name in phase_names.items():
                    p_idx = val_phase_indices[phase_id]
                    if p_idx:
                        phase_losses[phase_name] = val_loss_tensor[p_idx].mean().item()
                    else:
                        phase_losses[phase_name] = 0.0
            net.train()

            # --- SWA update ---
            if swa is not None and (epoch + 1) >= args.swa_start:
                swa.update(net)

            epochs_done = epoch + 1
            epochs_left = args.epochs - epochs_done
            current_lr  = optimizer.param_groups[0]['lr']

            # --- Smart ETA ---
            epoch_time_hist.append(epoch_time)
            if ema_epoch_time is None:
                ema_epoch_time = epoch_time
            else:
                ema_epoch_time = EMA_ALPHA * epoch_time + (1 - EMA_ALPHA) * ema_epoch_time

            next_epoch_s, total_eta_s = smart_eta(epoch_time_hist, ema_epoch_time, epochs_left)
            next_eta_str  = format_time(next_epoch_s) if next_epoch_s > 0 else "?"
            total_eta_str = format_time(total_eta_s)  if total_eta_s  > 0 else "?"

            swa_marker = " [SWA]" if swa is not None and epochs_done >= args.swa_start else ""
            print(f"Epoch {epochs_done:4d}/{args.epochs} | Train: {avg_loss:.6f} | "
                  f"Val: {val_loss:.6f} | LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.1f}s | Next: ~{next_eta_str} | Total ETA: {total_eta_str}{swa_marker}")
            if has_phase_data:
                print(f"      Phase loss - O: {phase_losses.get('opening',0):.6f}  "
                      f"M: {phase_losses.get('middlegame',0):.6f}  "
                      f"E: {phase_losses.get('endgame',0):.6f}")

            append_log(epochs_done, avg_loss, val_loss, current_lr, epoch_time,
                       run_id, phase_losses)
            all_log_data.append({
                'timestamp': datetime.now().isoformat(),
                'run_id': run_id, 'epoch': str(epochs_done),
                'loss': f'{avg_loss:.8f}', 'val_loss': f'{val_loss:.8f}',
                'lr': f'{current_lr:.8f}', 'epoch_time_s': f'{epoch_time:.2f}',
                'opening_loss':    f'{phase_losses.get("opening",    0):.8f}',
                'middlegame_loss': f'{phase_losses.get("middlegame", 0):.8f}',
                'endgame_loss':    f'{phase_losses.get("endgame",    0):.8f}',
            })

            if epochs_done % args.plot_every == 0:
                generate_plot(all_log_data)

            if val_loss < best_loss - 0.0001:
                best_loss         = val_loss
                epochs_no_improve = 0
                save_weights_cpp(net, args.output)
                torch.save({'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict()}, checkpoint_path)
            else:
                epochs_no_improve += 1
                if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
                    print(f"\nEarly stopping: no improvement for {args.early_stop} epochs.")
                    break

            if epochs_done % args.save_every == 0:
                save_weights_cpp(net, args.output)
                torch.save({'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict()}, checkpoint_path)

    except KeyboardInterrupt:
        print(f"\n\nStopped by user after epoch {epoch + 1}.")
        save_weights_cpp(net, args.output)
        torch.save({'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()}, checkpoint_path)
        generate_plot(all_log_data)
        print(f"Total time: {format_time(time.time() - start_time)}")
        if swa is not None and swa.count > 0:
            print("Applying SWA averaged weights...")
            swa.apply(net)
            swa_path = os.path.splitext(args.output)[0] + '_swa.bin'
            save_weights_cpp(net, swa_path)
            print(f"  SWA weights also saved to {swa_path}")
        return

    # --- End of training ---
    if swa is not None and swa.count > 0:
        swa_backup = os.path.splitext(args.output)[0] + '_pre_swa.bin'
        save_weights_cpp(net, swa_backup)
        print(f"  Pre-SWA weights backed up to {swa_backup}")
        print(f"Applying SWA averaged weights ({swa.count} snapshots)...")
        swa.apply(net)
        save_weights_cpp(net, args.output)
    else:
        save_weights_cpp(net, args.output)

    generate_plot(all_log_data)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Total time:  {format_time(total_time)}")
    print(f"  Best loss:   {best_loss:.6f}")
    print(f"  Weights:     {args.output}")
    if swa is not None and swa.count > 0:
        print(f"  SWA applied: Yes ({swa.count} snapshots averaged)")
    print(f"{'='*60}")

# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch NNUE Trainer (CPU-Optimized, Enhanced v3)')

    # Data
    parser.add_argument('--data',         type=str,   default='assets/training_data.bin',
                        help='Primary training data file (default: assets/training_data.bin)')
    parser.add_argument('--extra-data',   nargs=2,    action='append', metavar=('FILE', 'RATIO'),
                        help='Additional dataset with sampling ratio, e.g. --extra-data selfplay.bin 0.3 '
                             '(can be repeated). Ratios are relative weights; primary gets the remainder.')
    parser.add_argument('--load-weights', type=str,   default=None,
                        help='Load weights from file (starts fresh optimizer to avoid shape mismatch)')
    parser.add_argument('--fresh',        action='store_true',
                        help='Start with random weights (ignore existing)')
    parser.add_argument('--output',       type=str,   default='assets/nnue_weights.bin')

    # Core training
    parser.add_argument('--epochs',     type=int,   default=500)
    parser.add_argument('--batch-size', type=int,   default=8192)
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--lam',        type=float, default=0.5,
                        help='Blend: lam*eval_loss + (1-lam)*result_loss')
    parser.add_argument('--eval-scale', type=float, default=400.0)

    # LR schedule
    parser.add_argument('--cosine-t0',     type=int,   default=50)
    parser.add_argument('--cosine-t-mult', type=int,   default=2)
    parser.add_argument('--lr-min',        type=float, default=1e-6)
    parser.add_argument('--cosine-restarts', action='store_true', default=True,
                        help='Use cosine annealing with warm restarts (default)')
    parser.add_argument('--no-cosine-restarts', dest='cosine_restarts', action='store_false',
                        help='Use single cosine decay (better for fine-tuning)')

    # Regularization
    parser.add_argument('--grad-clip',    type=float, default=1.0)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--dropout',      type=float, default=0.1)

    # Phase balancing
    parser.add_argument('--phase-balanced',    action='store_true', default=True)
    parser.add_argument('--no-phase-balanced', dest='phase_balanced', action='store_false')

    # Stopping & saving
    parser.add_argument('--early-stop',  type=int, default=15)
    parser.add_argument('--save-every',  type=int, default=10)

    # Plotting
    parser.add_argument('--plot',       action='store_true')
    parser.add_argument('--plot-every', type=int, default=10)

    # Streaming
    parser.add_argument('--chunk-size', type=int, default=STREAM_CHUNK_SIZE,
                        help=f'Positions per chunk in streaming mode (default: {STREAM_CHUNK_SIZE:,})')
    parser.add_argument('--max-positions', type=int, default=0,
                        help='Limit training to first N positions (0=all)')

    # === ENHANCED FEATURES ===
    parser.add_argument('--enhanced', action='store_true', default=False,
                        help='Enable all enhancements with good defaults '
                             '(warmup, grad accum, SWA, label smoothing, rebalancing)')

    parser.add_argument('--warmup-steps', type=int, default=0,
                        help='Linear LR warmup over N optimizer steps (default: 0=off, enhanced: 1000)')
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='Gradient accumulation steps (default: 1=off, enhanced: 4)')
    parser.add_argument('--swa', action='store_true', default=False,
                        help='Enable Stochastic Weight Averaging')
    parser.add_argument('--swa-start', type=int, default=3,
                        help='Start SWA after this many epochs (default: 3)')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing factor (default: 0=off, enhanced: 0.02)')
    parser.add_argument('--filter-eval-max', type=float, default=0.0,
                        help='Hard-filter positions with |eval| > this (default: 0=off, enhanced: 10.0)')

    # === v3: REBALANCING ===
    parser.add_argument('--eval-soft-cap', type=float, default=0.0,
                        help='Soft-cap: downweight positions with |eval| > this in loss '
                             '(0=off, enhanced: 8.0). Weight = min(1, cap/|eval|)')
    parser.add_argument('--draw-weight', type=float, default=1.0,
                        help='Loss multiplier for drawn positions (1.0=off, enhanced: 3.0). '
                             'Compensates for draw underrepresentation in dataset')

    args = parser.parse_args()

    # Apply --enhanced defaults (only override if not explicitly set)
    if args.enhanced:
        if args.warmup_steps == 0:
            args.warmup_steps = 1000
        if args.grad_accum == 1:
            args.grad_accum = 4
        if not args.swa:
            args.swa = True
        if args.label_smoothing == 0.0:
            args.label_smoothing = 0.02
        if args.filter_eval_max == 0.0:
            args.filter_eval_max = 10.0
        if args.eval_soft_cap == 0.0:
            args.eval_soft_cap = 8.0
        if args.draw_weight <= 1.0:
            args.draw_weight = 3.0
        # Single cosine decay is better for fine-tuning
        args.cosine_restarts = False
        print("=== Enhanced mode v3: warmup=1000, grad_accum=4, SWA=ON, "
              "label_smooth=0.02, filter_eval=10.0, "
              f"eval_soft_cap=8.0, draw_weight=3.0x, cosine_decay ===\n")

    STREAM_CHUNK_SIZE = args.chunk_size

    if args.plot:
        if not HAS_MATPLOTLIB:
            print("ERROR: matplotlib required. Install with: pip install matplotlib")
            sys.exit(1)
        generate_plot(show=True)
    else:
        train(args)
