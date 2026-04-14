"""
analyze_phases.py
-----------------
Scans your training data binary and reports the opening / middlegame / endgame
position ratio, plus basic dataset statistics.

Usage (from your ChessEngine folder):
    py -3.10 analyze_phases.py
    py -3.10 analyze_phases.py --data assets\\training_data.bin
    py -3.10 analyze_phases.py --data assets\\training_data.bin --sample 500000
"""

import struct
import mmap
import argparse
import random
import time
import os

# ── Phase classification (same thresholds as train_nnue.py) ──────────────────

def compute_material_phase(features):
    # NOTE: feat % 384 assumes 768-feature encoding (384 features per side).
    # If the feature encoding scheme changes, this formula must be updated.
    phase = 0
    for feat in features:
        piece_offset = (feat % 384) // 64
        if   piece_offset == 1: phase += 1   # knight
        elif piece_offset == 2: phase += 1   # bishop
        elif piece_offset == 3: phase += 2   # rook
        elif piece_offset == 4: phase += 4   # queen
    return phase

def classify_phase(features):
    """0 = Opening (phase>=20), 1 = Middlegame (8-19), 2 = Endgame (<8)"""
    p = compute_material_phase(features)
    if p >= 20: return 0
    if p >= 8:  return 1
    return 2

# ── Binary format scanner ────────────────────────────────────────────────────

def scan_offsets(filename):
    """Returns list of byte offsets, one per position (variable-length format)."""
    size = os.path.getsize(filename)
    offsets = []
    with open(filename, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:  # F6.1b: ensure mmap cleanup on exception
            pos = 0
            # FIX H-2: Detect versioned format by checking for NNUE magic, not heuristic.
            header_raw = mm[0:4]
            if header_raw == b'NNUE':
                # Versioned format: 4-byte magic + 1-byte version + 4-byte count = 9
                pos = 9
            else:
                # Legacy format: first 4 bytes are uint32 position count
                candidate_count = struct.unpack_from('<I', header_raw, 0)[0]
                MIN_RECORD = 2 + 1*2 + 1 + 8
                MAX_RECORD = 2 + 768*2 + 1 + 8
                if candidate_count > 0 and candidate_count * MIN_RECORD <= size <= candidate_count * MAX_RECORD * 2:
                    pos = 4
            print(f"Scanning {filename} ({size/1e9:.2f} GB)...")
            t0 = time.time()
            last_report = t0
            while pos < len(mm):  # F6.1a: let struct.error catch truncated records
                offsets.append(pos)
                try:
                    num_feat = struct.unpack_from('<H', mm, pos)[0]
                    if num_feat == 0 or num_feat > 2048:
                        break
                    record_size = 2 + num_feat * 2 + 1 + 8  # +1 for stm byte
                    pos += record_size
                except struct.error:
                    break
                now = time.time()
                if now - last_report >= 5.0:
                    pct = pos / size * 100
                    elapsed = now - t0
                    eta = elapsed / (pos / size) - elapsed if pos > 0 else 0
                    print(f"  {len(offsets):,} positions found ({pct:.1f}%) — ETA {eta:.0f}s", end='\r')
                    last_report = now
        finally:
            mm.close()
    print(f"\n  Done — {len(offsets):,} total positions in {time.time()-t0:.1f}s")
    return offsets, filename

def read_features_at(mm, offset):
    num_feat = struct.unpack_from('<H', mm, offset)[0]
    features = struct.unpack_from(f'<{num_feat}H', mm, offset + 2)
    return features

def read_eval_at(mm, offset):
    num_feat = struct.unpack_from('<H', mm, offset)[0]
    game_result, search_eval = struct.unpack_from('<ff', mm, offset + 2 + num_feat * 2 + 1)  # +1 for stm byte
    return game_result, search_eval

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Analyze phase distribution of training data')
    parser.add_argument('--data',   default='assets/training_data.bin', help='Path to .bin training data')
    parser.add_argument('--sample', type=int, default=0,
                        help='Number of positions to sample (0 = all, slow for 65M+)')
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: File not found: {args.data}")
        print("Tip: run from your ChessEngine folder, or pass --data <path>")
        return

    offsets, filename = scan_offsets(args.data)
    total = len(offsets)

    if args.sample > 0 and args.sample < total:
        sample_offsets = random.sample(offsets, args.sample)
        sample_label = f"{args.sample:,} sampled positions"
    else:
        sample_offsets = offsets
        sample_label = f"all {total:,} positions"

    print(f"\nClassifying phases for {sample_label}...")
    counts = [0, 0, 0]
    eval_buckets = {'extreme (>1000)': 0, 'strong (500-1000)': 0, 'normal (<500)': 0}
    result_counts = {1.0: 0, 0.5: 0, 0.0: 0}
    # Draw positions by phase: draws_by_phase[phase] = count of draw positions
    draws_by_phase = [0, 0, 0]

    t0 = time.time()
    with open(filename, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:  # F6.1b: ensure mmap cleanup on exception
            for i, off in enumerate(sample_offsets):
                features = read_features_at(mm, off)
                phase = classify_phase(features)
                counts[phase] += 1

                game_result, search_eval = read_eval_at(mm, off)
                ae = abs(search_eval)
                if ae > 1000:   eval_buckets['extreme (>1000)'] += 1
                elif ae > 500:  eval_buckets['strong (500-1000)'] += 1
                else:           eval_buckets['normal (<500)']   += 1

                rounded = round(game_result * 2) / 2   # snap to 0.0 / 0.5 / 1.0
                result_counts[rounded] = result_counts.get(rounded, 0) + 1

                # Track draws per phase
                if abs(game_result - 0.5) < 0.01:
                    draws_by_phase[phase] += 1

                if i % 100000 == 0 and i > 0:
                    elapsed = time.time() - t0
                    eta = elapsed / i * (len(sample_offsets) - i)
                    print(f"  {i:,}/{len(sample_offsets):,} ({i/len(sample_offsets)*100:.1f}%) — ETA {eta:.0f}s", end='\r')
        finally:
            mm.close()

    n = sum(counts)
    print(f"\n\n{'='*55}")
    print(f"  PHASE DISTRIBUTION  ({sample_label})")
    print(f"{'='*55}")
    print(f"  Opening     (>=20 material): {counts[0]:>10,}  ({counts[0]/n*100:5.1f}%)")
    print(f"  Middlegame  ( 8-19 material): {counts[1]:>10,}  ({counts[1]/n*100:5.1f}%)")
    print(f"  Endgame     (  <8 material): {counts[2]:>10,}  ({counts[2]/n*100:5.1f}%)")
    print(f"  {'─'*49}")
    print(f"  Total                        {n:>10,}")

    print(f"\n  EVAL DISTRIBUTION")
    print(f"  {'─'*49}")
    for label, cnt in eval_buckets.items():
        print(f"  {label:<22}: {cnt:>10,}  ({cnt/n*100:5.1f}%)")

    print(f"\n  RESULT DISTRIBUTION")
    print(f"  {'─'*49}")
    labels = {1.0: 'White wins (1.0)', 0.5: 'Draw       (0.5)', 0.0: 'Black wins (0.0)'}
    for val, label in labels.items():
        cnt = result_counts.get(val, 0)
        print(f"  {label:<22}: {cnt:>10,}  ({cnt/n*100:5.1f}%)")

    # Draw statistics section
    total_draws = result_counts.get(0.5, 0)
    print(f"\n  DRAW STATISTICS")
    print(f"  {'─'*49}")
    print(f"  Total draw positions:     {total_draws:>10,}  ({total_draws/n*100:5.1f}% of dataset)")
    print(f"  Draws by phase:")
    phase_names = ['Opening', 'Middlegame', 'Endgame']
    for i, name in enumerate(phase_names):
        draw_pct_of_phase = draws_by_phase[i] / max(counts[i], 1) * 100
        draw_pct_of_draws = draws_by_phase[i] / max(total_draws, 1) * 100
        print(f"    {name:<12}: {draws_by_phase[i]:>8,}  "
              f"({draw_pct_of_phase:5.1f}% of phase, "
              f"{draw_pct_of_draws:5.1f}% of draws)")

    print(f"\n  IMBALANCE CHECK")
    print(f"  {'─'*49}")
    ideal = n / 3
    for i, name in enumerate(phase_names):
        ratio = counts[i] / ideal
        flag = '  ✅' if 0.7 <= ratio <= 1.4 else '  ⚠️  IMBALANCED'
        print(f"  {name:<12}: {ratio:.2f}x ideal{flag}")
    print(f"{'='*55}\n")

if __name__ == '__main__':
    main()
