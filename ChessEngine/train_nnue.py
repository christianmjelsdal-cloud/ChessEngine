#!/usr/bin/env python3
"""
PyTorch NNUE Trainer (CPU-Optimized)
=====================================
Pre-computes all tensors at startup for maximum throughput.
No DataLoader overhead — pure tensor operations.

Usage:
    python train_nnue.py [options]
    py -3.10 train_nnue.py --epochs 500 --batch-size 8192

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
L1_SIZE = 256
L2_SIZE = 32
L3_SIZE = 32

# =============================================================================
# Feature mirror — must match C++ mirrorFeature() exactly
# =============================================================================
def build_mirror_table():
    """Pre-compute mirror table for all 768 features."""
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
# Game Phase Classification — matches C++ Trainer::computeMaterialPhase()
# =============================================================================
def compute_material_phase(features):
    phase = 0
    for feat in features:
        piece_offset = (feat % 384) // 64
        if piece_offset == 1:    phase += 1  # Knight
        elif piece_offset == 2:  phase += 1  # Bishop
        elif piece_offset == 3:  phase += 2  # Rook
        elif piece_offset == 4:  phase += 4  # Queen
    return phase

def classify_phase(features):
    """0=Opening (>=20), 1=Middlegame (8-19), 2=Endgame (<8)"""
    phase = compute_material_phase(features)
    if phase >= 20: return 0
    if phase >= 8:  return 1
    return 2

# =============================================================================
# Binary I/O — matches C++ exactly
# =============================================================================
def load_training_data(filename):
    """Load training_data.bin and pre-compute ALL tensors upfront."""
    print(f"Loading positions from {filename}...")
    
    with open(filename, 'rb') as f:
        num_positions = struct.unpack('<I', f.read(4))[0]
        print(f"  Total positions: {num_positions}")
        
        # Pre-allocate numpy arrays (much faster than building tensors one by one)
        all_white = np.zeros((num_positions, NUM_FEATURES), dtype=np.float32)
        all_black = np.zeros((num_positions, NUM_FEATURES), dtype=np.float32)
        all_stm = np.zeros(num_positions, dtype=np.float32)
        all_result = np.zeros(num_positions, dtype=np.float32)
        all_eval = np.zeros(num_positions, dtype=np.float32)
        all_phases = np.zeros(num_positions, dtype=np.int32)
        
        try:
            for i in range(num_positions):
                num_features = struct.unpack('<H', f.read(2))[0]
                features = struct.unpack(f'<{num_features}H', f.read(2 * num_features))
                stm = struct.unpack('<B', f.read(1))[0]
                game_result = struct.unpack('<f', f.read(4))[0]
                search_eval = struct.unpack('<f', f.read(4))[0]
                
                # Set sparse features directly into dense arrays
                for feat in features:
                    all_white[i, feat] = 1.0
                    all_black[i, MIRROR_TABLE[feat]] = 1.0
                
                all_stm[i] = float(stm)
                all_result[i] = game_result
                all_eval[i] = search_eval
                all_phases[i] = classify_phase(features)
                
                if (i + 1) % 100000 == 0:
                    print(f"  Loaded {i+1}/{num_positions}...")
        except KeyboardInterrupt:
            print("\nInterrupted during data loading — exiting cleanly.")
            sys.exit(0)
    
    print(f"  Converting to tensors...")
    
    # Convert to PyTorch tensors (single allocation, no per-batch overhead)
    data = {
        'white': torch.from_numpy(all_white),
        'black': torch.from_numpy(all_black),
        'stm': torch.from_numpy(all_stm),
        'result': torch.from_numpy(all_result),
        'eval': torch.from_numpy(all_eval),
        'phases': all_phases,
        'count': num_positions,
    }
    
    # Phase stats
    phase_counts = [0, 0, 0]
    for p in all_phases:
        phase_counts[p] += 1
    print(f"Phase distribution: Opening={phase_counts[0]}, Middlegame={phase_counts[1]}, Endgame={phase_counts[2]}")
    
    return data

# =============================================================================
# Phase-balanced index generation
# =============================================================================
def build_phase_balanced_indices(phases, count):
    """Build oversampled indices for phase-balanced training."""
    phase_indices = {0: [], 1: [], 2: []}
    for i in range(count):
        phase_indices[phases[i]].append(i)
    
    counts = {k: len(v) for k, v in phase_indices.items()}
    if all(c > 0 for c in counts.values()):
        max_count = max(counts.values())
        indices = []
        for phase in [0, 1, 2]:
            src = phase_indices[phase]
            target = min(max_count, len(src) * 5)  # Cap at 5x
            sampled = [src[i % len(src)] for i in range(target)]
            indices.extend(sampled)
    else:
        indices = list(range(count))
    
    random.shuffle(indices)
    return indices

# =============================================================================
# Weight I/O — matches C++ binary format exactly
# =============================================================================
def save_weights_cpp(net, filename):
    with open(filename, 'wb') as f:
        # L1 weights: [768][256]
        f.write(net.l1_weight.detach().cpu().numpy().astype(np.float32).tobytes())
        # L1 biases: [256]
        f.write(net.l1_bias.detach().cpu().numpy().astype(np.float32).tobytes())
        # L2 weights: C++ [512][32], PyTorch (32, 512) — transpose
        f.write(net.l2.weight.detach().cpu().numpy().T.astype(np.float32).tobytes())
        # L2 biases: [32]
        f.write(net.l2.bias.detach().cpu().numpy().astype(np.float32).tobytes())
        # L3 weights: C++ [32][32], PyTorch (32, 32) — transpose
        f.write(net.l3.weight.detach().cpu().numpy().T.astype(np.float32).tobytes())
        # L3 biases: [32]
        f.write(net.l3.bias.detach().cpu().numpy().astype(np.float32).tobytes())
        # Output weights: [32]
        f.write(net.output.weight.detach().cpu().numpy().flatten().astype(np.float32).tobytes())
        # Output bias: [1]
        f.write(net.output.bias.detach().cpu().numpy().astype(np.float32).tobytes())
    print(f"Weights saved to {filename}")

def load_weights_cpp(net, filename):
    with open(filename, 'rb') as f:
        l1w = np.frombuffer(f.read(768 * 256 * 4), dtype=np.float32).reshape(768, 256)
        net.l1_weight.data = torch.from_numpy(l1w.copy())
        
        l1b = np.frombuffer(f.read(256 * 4), dtype=np.float32).copy()
        net.l1_bias.data = torch.from_numpy(l1b)
        
        l2w = np.frombuffer(f.read(512 * 32 * 4), dtype=np.float32).reshape(512, 32)
        net.l2.weight.data = torch.from_numpy(l2w.T.copy())
        
        l2b = np.frombuffer(f.read(32 * 4), dtype=np.float32).copy()
        net.l2.bias.data = torch.from_numpy(l2b)
        
        l3w = np.frombuffer(f.read(32 * 32 * 4), dtype=np.float32).reshape(32, 32)
        net.l3.weight.data = torch.from_numpy(l3w.T.copy())
        
        l3b = np.frombuffer(f.read(32 * 4), dtype=np.float32).copy()
        net.l3.bias.data = torch.from_numpy(l3b)
        
        ow = np.frombuffer(f.read(32 * 4), dtype=np.float32).reshape(1, 32)
        net.output.weight.data = torch.from_numpy(ow.copy())
        
        ob = np.frombuffer(f.read(4), dtype=np.float32).copy()
        net.output.bias.data = torch.from_numpy(ob)
    print(f"Weights loaded from {filename}")

# =============================================================================
# NNUE Network — exact replica of C++ architecture
# =============================================================================
class ClippedReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)

class NNUENetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_weight = nn.Parameter(torch.zeros(NUM_FEATURES, L1_SIZE))
        self.l1_bias = nn.Parameter(torch.zeros(L1_SIZE))
        self.l2 = nn.Linear(L1_SIZE * 2, L2_SIZE)
        self.l3 = nn.Linear(L2_SIZE, L3_SIZE)
        self.output = nn.Linear(L3_SIZE, 1)
        self.crelu = ClippedReLU()
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
        white_acc = self.crelu(white_acc)
        black_acc = self.crelu(black_acc)
        
        stm_mask = stm.unsqueeze(1)
        stm_acc = white_acc * (1 - stm_mask) + black_acc * stm_mask
        opp_acc = black_acc * (1 - stm_mask) + white_acc * stm_mask
        l1_out = torch.cat([stm_acc, opp_acc], dim=1)
        
        l2_out = self.crelu(self.l2(l1_out))
        l3_out = self.crelu(self.l3(l2_out))
        return self.output(l3_out)

# =============================================================================
# Training Log & Progress Graph
# =============================================================================
LOG_FILE = 'training_log.csv'
PLOT_FILE = 'training_progress.png'

def append_log(epoch, loss, lr, epoch_time, run_id):
    """Append one row to the persistent training log CSV."""
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'run_id', 'epoch', 'loss', 'lr', 'epoch_time_s'])
        writer.writerow([datetime.now().isoformat(), run_id, epoch, f'{loss:.8f}', f'{lr:.8f}', f'{epoch_time:.2f}'])

def read_log():
    """Read the full training log. Returns list of dicts."""
    if not os.path.exists(LOG_FILE):
        return []
    rows = []
    with open(LOG_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def generate_plot(log_data=None, show=False):
    """Generate training progress graph from the log file."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed — skipping plot. Install with: pip install matplotlib")
        return
    
    if log_data is None:
        log_data = read_log()
    
    if not log_data:
        print("No training log data to plot.")
        return
    
    epochs = list(range(1, len(log_data) + 1))
    losses = [float(row['loss']) for row in log_data]
    lrs = [float(row['lr']) for row in log_data]
    
    # Detect run boundaries for coloring
    run_ids = [row.get('run_id', '?') for row in log_data]
    unique_runs = []
    for r in run_ids:
        if r not in unique_runs:
            unique_runs.append(r)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('NNUE Training Progress (All-Time)', fontsize=14, fontweight='bold')
    
    # --- Loss plot --- (one continuous line, run boundaries marked subtly)
    ax1.plot(epochs, losses, '-', color='#2196F3', linewidth=1.2, label='Loss')
    
    # Mark run boundaries with vertical lines
    if len(unique_runs) > 1:
        for run_id in unique_runs[1:]:  # skip first run
            first_epoch = next(e for e, r in zip(epochs, run_ids) if r == run_id)
            ax1.axvline(x=first_epoch, color='gray', linestyle=':', alpha=0.4, linewidth=1)
        # Add one label for the legend
        ax1.axvline(x=-1, color='gray', linestyle=':', alpha=0.4, linewidth=1,
                     label=f'Run boundary ({len(unique_runs)} runs)')
    
    # Smoothed trend line (rolling average, window = ~15% of data, min 3, max 15)
    if len(losses) >= 5:
        window = max(3, min(15, len(losses) // 7))
        half = window // 2
        smoothed = []
        for i in range(len(losses)):
            lo = max(0, i - half)
            hi = min(len(losses), i + half + 1)
            smoothed.append(sum(losses[lo:hi]) / (hi - lo))
        ax1.plot(epochs, smoothed, '--', color='#FF5722', linewidth=2, alpha=0.8,
                 label=f'Trend (rolling avg ×{window})')
    
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='upper right')
    
    # Annotate best loss
    best_idx = losses.index(min(losses))
    ax1.annotate(f'Best: {losses[best_idx]:.6f}\n(epoch {epochs[best_idx]})',
                 xy=(epochs[best_idx], losses[best_idx]),
                 xytext=(30, 30), textcoords='offset points',
                 fontsize=9, color='green',
                 arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    
    # --- Learning rate plot ---
    ax2.plot(epochs, lrs, '-', color='#9C27B0', linewidth=1)
    ax2.set_xlabel('Epoch (all-time)', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Stats box
    stats_text = (f'Total epochs: {len(epochs)}\n'
                  f'Best loss: {min(losses):.6f}\n'
                  f'Latest loss: {losses[-1]:.6f}\n'
                  f'Training runs: {len(unique_runs)}')
    ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches='tight')
    print(f"Progress graph saved to {PLOT_FILE}")
    
    plt.close()
    if show:
        # Open the saved image with the default viewer instead of plt.show()
        import subprocess, sys
        if sys.platform == 'win32':
            subprocess.Popen(['start', '', PLOT_FILE], shell=True)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', PLOT_FILE])
        else:
            subprocess.Popen(['xdg-open', PLOT_FILE])

# =============================================================================
# Training Loop — optimized: no DataLoader, pure tensor slicing
# =============================================================================
def train(args):
    device = torch.device('cpu')
    print(f"Using device: CPU (optimized)")
    
    # Set thread count for maximum CPU utilization
    if hasattr(torch, 'set_num_threads'):
        num_cores = os.cpu_count() or 4
        torch.set_num_threads(num_cores)
        print(f"  CPU threads: {num_cores}")
    
    # Load and pre-compute ALL tensors
    data = load_training_data(args.data)
    if data['count'] == 0:
        print("ERROR: No training data found!")
        return
    
    # All tensors are already on CPU and ready to slice
    white_all = data['white']
    black_all = data['black']
    stm_all = data['stm']
    result_all = data['result']
    eval_all = data['eval']
    
    # Create network
    net = NNUENetwork()
    
    # Auto-load existing weights: explicit --load-weights, or resume from output path
    weights_path = args.load_weights or args.output
    checkpoint_path = os.path.splitext(args.output)[0] + '_checkpoint.pt'
    resumed = False
    
    if os.path.exists(weights_path):
        load_weights_cpp(net, weights_path)
        print(f"Loaded existing weights from {weights_path}")
    
    # Optimizer: Adam converges faster than SGD for this network
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    
    # Restore optimizer/scheduler state if checkpoint exists (seamless continuation)
    if os.path.exists(checkpoint_path) and not args.fresh:
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            resumed = True
            print(f"Resumed optimizer state (LR: {scheduler.get_last_lr()[0]:.6f})")
        except Exception as e:
            print(f"Could not load checkpoint (starting fresh optimizer): {e}")
    
    lam = args.lam
    eval_scale = args.eval_scale
    batch_size = args.batch_size
    
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Positions:      {data['count']}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  LR decay:       {args.lr_decay}")
    print(f"  Lambda:         {lam}")
    print(f"  Eval scale:     {eval_scale}")
    print(f"  Phase balanced: {args.phase_balanced}")
    print(f"  Early stop:     {args.early_stop} epochs")
    print(f"{'='*60}\n")
    
    best_loss = float('inf')
    epochs_no_improve = 0
    start_time = time.time()
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_log_data = read_log()  # Load previous runs for combined plotting
    
    try:
        for epoch in range(args.epochs):
            epoch_start = time.time()
            net.train()
            
            # Build epoch indices
            if args.phase_balanced:
                indices = build_phase_balanced_indices(data['phases'], data['count'])
            else:
                indices = list(range(data['count']))
                random.shuffle(indices)
            
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            num_samples = len(indices)
            
            total_loss = 0.0
            num_batches = 0
            
            # Pure tensor-sliced batching — no DataLoader overhead
            for batch_start in range(0, num_samples, batch_size):
                batch_idx = idx_tensor[batch_start:batch_start + batch_size]
                
                white_feat  = white_all[batch_idx]
                black_feat  = black_all[batch_idx]
                stm         = stm_all[batch_idx]
                game_result = result_all[batch_idx]
                search_eval = eval_all[batch_idx]
                
                # Forward pass
                raw_output = net(white_feat, black_feat, stm)
                predicted  = raw_output.squeeze(1) * 400.0
                
                # Flip to white's POV when black to move
                predicted_white = torch.where(stm < 0.5, predicted, -predicted)
                
                # Sigmoid of predicted and target
                sig_pred   = torch.sigmoid(predicted_white / eval_scale)
                sig_target = torch.sigmoid(search_eval / eval_scale)
                
                # Lambda-blended loss
                eval_loss   = (sig_pred - sig_target) ** 2
                result_loss = (sig_pred - game_result) ** 2
                loss = (lam * eval_loss + (1.0 - lam) * result_loss).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            scheduler.step()
            
            avg_loss   = total_loss / max(num_batches, 1)
            epoch_time = time.time() - epoch_start
            elapsed    = time.time() - start_time
            
            epochs_done = epoch + 1
            epochs_left = args.epochs - epochs_done
            eta_seconds = (elapsed / epochs_done) * epochs_left
            eta_str     = format_time(eta_seconds)
            
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epochs_done:4d}/{args.epochs} | Loss: {avg_loss:.6f} | "
                  f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s | ETA: {eta_str}")
            
            # Log to CSV
            append_log(epochs_done, avg_loss, current_lr, epoch_time, run_id)
            all_log_data.append({
                'timestamp': datetime.now().isoformat(),
                'run_id': run_id,
                'epoch': str(epochs_done),
                'loss': f'{avg_loss:.8f}',
                'lr': f'{current_lr:.8f}',
                'epoch_time_s': f'{epoch_time:.2f}'
            })
            
            # Update graph periodically
            if epochs_done % args.plot_every == 0:
                generate_plot(all_log_data)
            
            # Save best weights
            if avg_loss < best_loss - 0.0001:
                best_loss = avg_loss
                epochs_no_improve = 0
                save_weights_cpp(net, args.output)
                # Save optimizer checkpoint for seamless resume
                torch.save({'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict()}, checkpoint_path)
            else:
                epochs_no_improve += 1
                if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
                    print(f"\nEarly stopping: no improvement for {args.early_stop} epochs.")
                    break
            
            # Periodic save
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
        total_time = time.time() - start_time
        print(f"Total time: {format_time(total_time)}")
        return
    
    # Final save and plot
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
    parser = argparse.ArgumentParser(description='PyTorch NNUE Trainer (CPU-Optimized)')
    
    parser.add_argument('--data', type=str, default='assets/training_data.bin',
                        help='Path to training_data.bin')
    parser.add_argument('--load-weights', type=str, default=None,
                        help='Path to existing weights to continue training from')
    parser.add_argument('--fresh', action='store_true',
                        help='Ignore checkpoint and start optimizer from scratch (still loads weights)')
    parser.add_argument('--output', type=str, default='assets/nnue_weights.bin',
                        help='Output path for trained weights')
    
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8192,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.995,
                        help='LR decay per epoch')
    parser.add_argument('--lam', type=float, default=0.5,
                        help='Lambda: weight between eval loss and result loss')
    parser.add_argument('--eval-scale', type=float, default=400.0,
                        help='Sigmoid scaling factor for eval')
    
    parser.add_argument('--phase-balanced', action='store_true', default=True,
                        help='Use phase-balanced training (default: on)')
    parser.add_argument('--no-phase-balanced', dest='phase_balanced', action='store_false',
                        help='Disable phase-balanced training')
    
    parser.add_argument('--early-stop', type=int, default=15,
                        help='Stop after N epochs without improvement (0=disabled)')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save weights every N epochs')
    
    parser.add_argument('--plot', action='store_true',
                        help='Just show the training progress graph (no training)')
    parser.add_argument('--plot-every', type=int, default=10,
                        help='Update progress graph every N epochs (default: 10)')
    
    args = parser.parse_args()
    
    if args.plot:
        if not HAS_MATPLOTLIB:
            print("ERROR: matplotlib required for plotting. Install with: pip install matplotlib")
            sys.exit(1)
        generate_plot(show=True)
    else:
        train(args)
