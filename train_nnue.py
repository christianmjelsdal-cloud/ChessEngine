#!/usr/bin/env python3
"""
PyTorch NNUE Trainer (CPU-Optimized, Streaming)
=================================================
Supports arbitrarily large datasets via streaming chunked loading.
No RAM limit — processes data in chunks without loading everything at once.

CHANGES from previous version:
  1. StreamingDataset: loads data in chunks, no full preallocation
  2. Automatic mode selection: small datasets (<= MAX_PRELOAD_POSITIONS) use
     fast preload mode; large datasets use streaming mode
  3. All other features unchanged: SCReLU, cosine annealing, phase balancing,
     weight decay, dropout, gradient clipping, validation loss, etc.

Usage:
    py -3.10 train_nnue.py --epochs 500 --batch-size 8192 --fresh --early-stop 80 --phase-balanced

Requirements:
    pip install torch numpy
"""

import argparse
import struct
import time
import sys
import os
import random
import csv
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend (no window needed)
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# =============================================================================
# Constants — must match C++ NNUE.h exactly
# =============================================================================
NUM_FEATURES = 768
L1_SIZE = 512   # CHANGED: 256 -> 512 for 42M+ position dataset
L2_SIZE = 128   # CHANGED: 64 -> 128 for more capacity
L3_SIZE = 128   # CHANGED: 64 -> 128 for more capacity

# Datasets larger than this use streaming mode (saves RAM)
MAX_PRELOAD_POSITIONS = 8_000_000  # 8M positions ~ 24 GB RAM (safe threshold)

# Chunk size for streaming mode — how many positions to load at once per batch step
STREAM_CHUNK_SIZE = 500_000  # ~1.5 GB per chunk

# =============================================================================
# Feature mirror — must match C++ mirrorFeature() exactly
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
# Binary format reader — returns (num_positions, byte_offsets_list)
# =============================================================================
def scan_positions(filename):
    """
    Fast scan: reads the file to build a list of byte offsets for each position.
    Allows random access without loading all feature data into RAM.
    Returns (num_positions, offsets) where offsets[i] = byte offset of position i.
    """
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
            # Read num_features (2 bytes)
            header = f.read(2)
            if len(header) < 2:
                print(f"  Warning: truncated at position {i}")
                num_positions = i
                break
            num_features = struct.unpack_from('<H', header, 0)[0]
            # Skip: features (2*num_features) + stm (1) + result+eval (8)
            skip = 2 * num_features + 1 + 8
            f.seek(skip, 1)
            offset += 2 + skip

            if (i + 1) % 1_000_000 == 0:
                print(f"  Scanned {i+1:,}/{num_positions:,}...")

    print(f"  Scan complete: {len(offsets):,} positions indexed.")
    return num_positions, offsets


def load_positions_at_offsets(filename, offsets):
    """
    Load a specific set of positions by their byte offsets.
    Returns dict with white, black, stm, result, eval, phases tensors.
    """
    n = len(offsets)
    mirror = MIRROR_TABLE

    all_white  = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    all_black  = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    all_stm    = np.zeros(n, dtype=np.float32)
    all_result = np.zeros(n, dtype=np.float32)
    all_eval   = np.zeros(n, dtype=np.float32)
    all_phases = np.zeros(n, dtype=np.int32)

    with open(filename, 'rb') as f:
        for i, off in enumerate(offsets):
            f.seek(off)
            num_features = struct.unpack('<H', f.read(2))[0]
            features = struct.unpack(f'<{num_features}H', f.read(2 * num_features))
            stm = struct.unpack('B', f.read(1))[0]
            game_result, search_eval = struct.unpack('<ff', f.read(8))

            for feat in features:
                all_white[i, feat] = 1.0
                all_black[i, mirror[feat]] = 1.0

            all_stm[i]    = float(stm)
            all_result[i] = game_result
            all_eval[i]   = search_eval
            all_phases[i] = classify_phase(features)

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
# Fast preload (small datasets) — original approach
# =============================================================================
def load_training_data(filename):
    """Load entire dataset into RAM. Only used for small datasets."""
    print(f"Loading positions from {filename}...")

    with open(filename, 'rb') as f:
        raw = f.read()

    num_positions = struct.unpack_from('<I', raw, 0)[0]
    print(f"  Total positions: {num_positions:,}")

    all_white  = np.zeros((num_positions, NUM_FEATURES), dtype=np.float32)
    all_black  = np.zeros((num_positions, NUM_FEATURES), dtype=np.float32)
    all_stm    = np.zeros(num_positions, dtype=np.float32)
    all_result = np.zeros(num_positions, dtype=np.float32)
    all_eval   = np.zeros(num_positions, dtype=np.float32)
    all_phases = np.zeros(num_positions, dtype=np.int32)

    mirror = MIRROR_TABLE
    offset = 4

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

            for feat in features:
                all_white[i, feat] = 1.0
                all_black[i, mirror[feat]] = 1.0

            all_stm[i]    = float(stm)
            all_result[i] = game_result
            all_eval[i]   = search_eval
            all_phases[i] = classify_phase(features)

            if (i + 1) % 200000 == 0:
                print(f"  Loaded {i+1:,}/{num_positions:,}...")
    except KeyboardInterrupt:
        print("\nInterrupted during data loading — exiting cleanly.")
        sys.exit(0)

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
    for p in all_phases:
        phase_counts[p] += 1
    print(f"Phase distribution: Opening={phase_counts[0]:,}, Middlegame={phase_counts[1]:,}, Endgame={phase_counts[2]:,}")

    return data


# =============================================================================
# Weight I/O — matches C++ binary format exactly
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
    with open(filename, 'rb') as f:
        l1w = np.frombuffer(f.read(768 * 256 * 4), dtype=np.float32).reshape(768, 256)
        net.l1_weight.data = torch.from_numpy(l1w.copy())
        l1b = np.frombuffer(f.read(256 * 4), dtype=np.float32).copy()
        net.l1_bias.data = torch.from_numpy(l1b)
        l2w = np.frombuffer(f.read(512 * L2_SIZE * 4), dtype=np.float32).reshape(512, L2_SIZE)
        net.l2.weight.data = torch.from_numpy(l2w.T.copy())
        l2b = np.frombuffer(f.read(L2_SIZE * 4), dtype=np.float32).copy()
        net.l2.bias.data = torch.from_numpy(l2b)
        l3w = np.frombuffer(f.read(L2_SIZE * L3_SIZE * 4), dtype=np.float32).reshape(L2_SIZE, L3_SIZE)
        net.l3.weight.data = torch.from_numpy(l3w.T.copy())
        l3b = np.frombuffer(f.read(L3_SIZE * 4), dtype=np.float32).copy()
        net.l3.bias.data = torch.from_numpy(l3b)
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
        scale = 0.1
        std = scale * (2.0 / (NUM_FEATURES + L1_SIZE)) ** 0.5
        nn.init.normal_(self.l1_weight, 0, std)
        nn.init.zeros_(self.l1_bias)
        std = scale * (2.0 / (L1_SIZE * 2 + L2_SIZE)) ** 0.5
        nn.init.normal_(self.l2.weight, 0, std)
        nn.init.zeros_(self.l2.bias)
        std = scale * (2.0 / (L2_SIZE + L3_SIZE)) ** 0.5
        nn.init.normal_(self.l3.weight, 0, std)
        nn.init.zeros_(self.l3.bias)
        std = scale * (2.0 / (L3_SIZE + 1)) ** 0.5
        nn.init.normal_(self.output.weight, 0, std)
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
# Training Log & Progress Graph
# =============================================================================
LOG_FILE  = 'training_log.csv'
PLOT_FILE = 'training_progress.png'

def append_log(epoch, train_loss, val_loss, lr, epoch_time, run_id, phase_losses=None):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'run_id', 'epoch', 'loss', 'val_loss',
                             'lr', 'epoch_time_s',
                             'opening_loss', 'middlegame_loss', 'endgame_loss'])
        pl = phase_losses or {}
        writer.writerow([datetime.now().isoformat(), run_id, epoch,
                         f'{train_loss:.8f}', f'{val_loss:.8f}',
                         f'{lr:.8f}', f'{epoch_time:.2f}',
                         f'{pl.get("opening", 0):.8f}',
                         f'{pl.get("middlegame", 0):.8f}',
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
        print("matplotlib not installed — skipping plot.")
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
                 label=f'Trend (avg ×{window})')

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
                stats_lines.append(f'⚠ Overfit ratio: {latest_ratio:.2f}×')
            else:
                stats_lines.append(f'✓ No overfitting ({latest_ratio:.2f}×)')
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
# Training Loop
# =============================================================================
def compute_loss(net, white, black, stm, game_result, search_eval, lam, eval_scale):
    raw_output     = net(white, black, stm)
    predicted      = raw_output.squeeze(1) * 400.0
    predicted_white = torch.where(stm < 0.5, predicted, -predicted)
    sig_pred       = torch.sigmoid(predicted_white / eval_scale)
    sig_target     = torch.sigmoid(search_eval / eval_scale)
    eval_loss      = (sig_pred - sig_target) ** 2
    result_loss    = (sig_pred - game_result) ** 2
    return lam * eval_loss + (1.0 - lam) * result_loss


def train(args):
    device = torch.device('cpu')
    print(f"Using device: CPU (optimized)")

    if hasattr(torch, 'set_num_threads'):
        num_cores = os.cpu_count() or 4
        torch.set_num_threads(num_cores)
        print(f"  CPU threads: {num_cores}")

    # ─── Determine dataset size ───
    with open(args.data, 'rb') as f:
        num_positions = struct.unpack('<I', f.read(4))[0]
    print(f"  Total positions: {num_positions:,}")

    streaming_mode = num_positions > MAX_PRELOAD_POSITIONS
    if streaming_mode:
        print(f"  Mode: STREAMING (dataset too large for RAM preload)")
        print(f"  Chunk size: {STREAM_CHUNK_SIZE:,} positions per chunk")
    else:
        print(f"  Mode: PRELOAD (full dataset fits in RAM)")

    lam        = args.lam
    eval_scale = args.eval_scale
    batch_size = args.batch_size

    # ─── Create network ───
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

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.cosine_t0,
        T_mult=args.cosine_t_mult,
        eta_min=args.lr_min
    )

    if not args.fresh and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            print(f"Resumed optimizer state (LR: {scheduler.get_last_lr()[0]:.6f})")
        except Exception as e:
            print(f"Could not load checkpoint (starting fresh optimizer): {e}")

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Positions:        {num_positions:,}")
    print(f"  Batch size:       {batch_size}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Learning rate:    {args.lr}")
    print(f"  LR schedule:      Cosine Annealing (T0={args.cosine_t0}, Tmult={args.cosine_t_mult})")
    print(f"  LR min:           {args.lr_min}")
    print(f"  Lambda:           {lam}")
    print(f"  Eval scale:       {eval_scale}")
    print(f"  Phase balanced:   {args.phase_balanced}")
    print(f"  Early stop:       {args.early_stop} epochs")
    print(f"  Weight decay:     {args.weight_decay}")
    print(f"  Dropout:          {args.dropout}")
    print(f"  Activation:       SCReLU (squared clipped ReLU)")
    print(f"  Architecture:     768→{L1_SIZE}→{L2_SIZE}→{L3_SIZE}→1")
    print(f"  Grad clip norm:   {args.grad_clip}")
    print(f"  Streaming:        {streaming_mode}")
    print(f"{'='*60}\n")

    # ─── Setup validation set (fixed ~10% sample, loaded once) ───
    rng = random.Random(42)
    all_indices = list(range(num_positions))
    rng.shuffle(all_indices)
    val_size   = max(1, num_positions // 10)
    val_size   = min(val_size, 500_000)   # Cap validation at 500K (enough, saves RAM)
    val_indices  = all_indices[:val_size]
    train_indices = all_indices[val_size:]

    print(f"  Train positions: {len(train_indices):,}")
    print(f"  Val positions:   {val_size:,}")

    if streaming_mode:
        # For streaming: scan offsets once, keep in memory (just ints — cheap)
        _, all_offsets = scan_positions(args.data)
        val_offsets = [all_offsets[i] for i in val_indices]
        train_offsets = [all_offsets[i] for i in train_indices]

        print(f"  Loading validation set ({val_size:,} positions)...")
        val_data = load_positions_at_offsets(args.data, val_offsets)
    else:
        # Preload mode — load everything
        data = load_training_data(args.data)
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

    best_loss        = float('inf')
    epochs_no_improve = 0
    start_time       = time.time()
    run_id           = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_log_data     = read_log()
    has_phase_data   = any(len(v) > 0 for v in val_phase_indices.values())

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

    print("  Mode: eager (standard PyTorch)")

    try:
        for epoch in range(args.epochs):
            epoch_start = time.time()
            net.train()

            total_loss  = 0.0
            num_batches = 0

            if streaming_mode:
                # ── Streaming: shuffle offsets, load in chunks ──
                chunk_offsets = train_offsets.copy()
                random.shuffle(chunk_offsets)

                chunk_size = STREAM_CHUNK_SIZE
                num_chunks = (len(chunk_offsets) + chunk_size - 1) // chunk_size

                for chunk_i in range(num_chunks):
                    chunk_slice = chunk_offsets[chunk_i * chunk_size : (chunk_i + 1) * chunk_size]
                    chunk_data  = load_positions_at_offsets(args.data, chunk_slice)

                    white_all  = chunk_data['white']
                    black_all  = chunk_data['black']
                    stm_all    = chunk_data['stm']
                    result_all = chunk_data['result']
                    eval_all   = chunk_data['eval']

                    chunk_indices = list(range(len(chunk_slice)))
                    random.shuffle(chunk_indices)
                    idx_tensor = torch.tensor(chunk_indices, dtype=torch.long)

                    for batch_start in range(0, len(chunk_indices), batch_size):
                        batch_idx   = idx_tensor[batch_start:batch_start + batch_size]
                        loss_tensor = compute_loss(
                            net,
                            white_all[batch_idx], black_all[batch_idx],
                            stm_all[batch_idx],   result_all[batch_idx],
                            eval_all[batch_idx],  lam, eval_scale
                        )
                        loss = loss_tensor.mean()
                        optimizer.zero_grad()
                        loss.backward()
                        if args.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                        optimizer.step()
                        total_loss  += loss.item()
                        num_batches += 1

                    # Free chunk memory
                    del chunk_data, white_all, black_all, stm_all, result_all, eval_all

                    if num_chunks > 1:
                        print(f"  Epoch {epoch+1} chunk {chunk_i+1}/{num_chunks} done", end='\r')

            else:
                # ── Preload: use cached tensors ──
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
                    batch_idx   = idx_tensor[batch_start:batch_start + batch_size]
                    loss_tensor = compute_loss(
                        net,
                        white_all[batch_idx], black_all[batch_idx],
                        stm_all[batch_idx],   result_all[batch_idx],
                        eval_all[batch_idx],  lam, eval_scale
                    )
                    loss = loss_tensor.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                    optimizer.step()
                    total_loss  += loss.item()
                    num_batches += 1

            scheduler.step()

            avg_loss   = total_loss / max(num_batches, 1)
            epoch_time = time.time() - epoch_start
            elapsed    = time.time() - start_time

            # ─── Validation ───
            net.eval()
            with torch.no_grad():
                val_loss_tensor = compute_loss(
                    net, val_white, val_black, val_stm,
                    val_result, val_eval, lam, eval_scale
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

            epochs_done = epoch + 1
            epochs_left = args.epochs - epochs_done
            eta_seconds = (elapsed / epochs_done) * epochs_left
            eta_str     = format_time(eta_seconds)
            current_lr  = scheduler.get_last_lr()[0]

            print(f"Epoch {epochs_done:4d}/{args.epochs} | Train: {avg_loss:.6f} | "
                  f"Val: {val_loss:.6f} | LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.1f}s | ETA: {eta_str}")
            if has_phase_data:
                print(f"      Phase loss — O: {phase_losses.get('opening',0):.6f}  "
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
                best_loss        = val_loss
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
        return

    save_weights_cpp(net, args.output)
    generate_plot(all_log_data)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Total time:  {format_time(total_time)}")
    print(f"  Best loss:   {best_loss:.6f}")
    print(f"  Weights:     {args.output}")
    print(f"{'='*60}")


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
# Main
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch NNUE Trainer (CPU-Optimized, SCReLU + Cosine Annealing)')

    parser.add_argument('--data',         type=str,   default='assets/training_data.bin')
    parser.add_argument('--load-weights', type=str,   default=None)
    parser.add_argument('--fresh',        action='store_true')
    parser.add_argument('--output',       type=str,   default='assets/nnue_weights.bin')

    parser.add_argument('--epochs',     type=int,   default=500)
    parser.add_argument('--batch-size', type=int,   default=8192)
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--lam',        type=float, default=0.5)
    parser.add_argument('--eval-scale', type=float, default=400.0)

    parser.add_argument('--cosine-t0',     type=int,   default=50)
    parser.add_argument('--cosine-t-mult', type=int,   default=2)
    parser.add_argument('--lr-min',        type=float, default=1e-6)
    parser.add_argument('--lr-decay',      type=float, default=0.995)  # deprecated

    parser.add_argument('--grad-clip',    type=float, default=1.0)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--dropout',      type=float, default=0.1)

    parser.add_argument('--phase-balanced',    action='store_true', default=True)
    parser.add_argument('--no-phase-balanced', dest='phase_balanced', action='store_false')

    parser.add_argument('--early-stop',  type=int, default=15)
    parser.add_argument('--save-every',  type=int, default=10)

    parser.add_argument('--plot',       action='store_true')
    parser.add_argument('--plot-every', type=int, default=10)

    # Streaming tuning
    parser.add_argument('--chunk-size', type=int, default=STREAM_CHUNK_SIZE,
                        help=f'Positions per chunk in streaming mode (default: {STREAM_CHUNK_SIZE:,})')

    args = parser.parse_args()
    STREAM_CHUNK_SIZE = args.chunk_size

    if args.plot:
        if not HAS_MATPLOTLIB:
            print("ERROR: matplotlib required. Install with: pip install matplotlib")
            sys.exit(1)
        generate_plot(show=True)
    else:
        train(args)
