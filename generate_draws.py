#!/usr/bin/env python3
"""
Generate Drawn Endgame Positions via Local Syzygy Tablebases
=============================================================
Probes local Syzygy .rtbw files directly - no API, no rate limit.
Generates thousands of drawn positions per second.

Binary format per position (matches train_nnue.py):
  [2B: num_features] [num_features * 2B: feature_indices] [1B: stm] [4B: result] [4B: eval]

Setup:
  1. Download Syzygy WDL files (see --help for download links)
  2. Point --syzygy-path at the directory containing the .rtbw files
  3. Run!

Usage:
    py -3.10 generate_draws.py --syzygy-path C:/syzygy --target 500000 --merge assets/training_data.bin

Requirements:
    pip install chess
"""

import argparse
import struct
import random
import time
import sys
import os
import shutil

try:
    import chess
    import chess.syzygy
except ImportError:
    print("ERROR: python-chess required. Install with: pip install chess")
    sys.exit(1)

# =============================================================================
# Draw source types (must match DrawSource enum in train_nnue.py)
# =============================================================================
# 0 = NOT_DRAW, 1 = UNKNOWN, 2 = STALEMATE, 3 = INSUFFICIENT,
# 4 = FIFTY_MOVE, 5 = REPETITION, 6 = AGREEMENT, 7 = TABLEBASE
DRAW_SOURCE_TABLEBASE = 7  # All positions from this script are Syzygy tablebase draws

# =============================================================================
# Phase classification (same thresholds as analyze_phases.py / train_nnue.py)
# =============================================================================
def compute_material_phase(features):
    """Compute material-based game phase from feature indices."""
    phase = 0
    for feat in features:
        piece_offset = (feat % 384) // 64
        if   piece_offset == 1: phase += 1   # knight
        elif piece_offset == 2: phase += 1   # bishop
        elif piece_offset == 3: phase += 2   # rook
        elif piece_offset == 4: phase += 4   # queen
    return phase

def classify_phase(features):
    """0=Opening (>=20), 1=Middlegame (8-19), 2=Endgame (<8)"""
    p = compute_material_phase(features)
    if p >= 20: return 0
    if p >= 8:  return 1
    return 2

PHASE_NAMES = ['Opening', 'Middlegame', 'Endgame']

# =============================================================================
# Feature encoding - must match C++ / train_nnue.py exactly
# =============================================================================
PIECE_TO_INDEX = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}

def board_to_features(board):
    features = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            idx = PIECE_TO_INDEX[(piece.piece_type, piece.color)]
            features.append(idx * 64 + square)
    return sorted(features)

# =============================================================================
# Endgame piece configurations
# =============================================================================
ENDGAME_CONFIGS = {
    # 4-piece (very high draw rate, tiny files ~100MB total)
    'KRvKR':  [(chess.ROOK,   chess.WHITE), (chess.ROOK,   chess.BLACK)],
    'KBvKB':  [(chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.BLACK)],
    'KNvKN':  [(chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    'KPvKP':  [(chess.PAWN,   chess.WHITE), (chess.PAWN,   chess.BLACK)],
    'KRvKB':  [(chess.ROOK,   chess.WHITE), (chess.BISHOP, chess.BLACK)],
    'KRvKN':  [(chess.ROOK,   chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    'KBvKN':  [(chess.BISHOP, chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    'KQvKQ':  [(chess.QUEEN,  chess.WHITE), (chess.QUEEN,  chess.BLACK)],

    # 5-piece (good variety, ~9GB for all 5-piece files)
    'KRPvKR': [(chess.ROOK,   chess.WHITE), (chess.PAWN,   chess.WHITE), (chess.ROOK,   chess.BLACK)],
    'KBPvKB': [(chess.BISHOP, chess.WHITE), (chess.PAWN,   chess.WHITE), (chess.BISHOP, chess.BLACK)],
    'KRvKBN': [(chess.ROOK,   chess.WHITE), (chess.BISHOP, chess.BLACK), (chess.KNIGHT, chess.BLACK)],
    'KBNvKR': [(chess.BISHOP, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.ROOK,   chess.BLACK)],
    'KRPvKB': [(chess.ROOK,   chess.WHITE), (chess.PAWN,   chess.WHITE), (chess.BISHOP, chess.BLACK)],
    'KRPvKN': [(chess.ROOK,   chess.WHITE), (chess.PAWN,   chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    'KNPvKN': [(chess.KNIGHT, chess.WHITE), (chess.PAWN,   chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    'KBPvKN': [(chess.BISHOP, chess.WHITE), (chess.PAWN,   chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    'KPPvKP': [(chess.PAWN,   chess.WHITE), (chess.PAWN,   chess.WHITE), (chess.PAWN,   chess.BLACK)],

    # 6-piece (requires 6-piece .rtbw files, ~150GB - optional)
    'KRPvKRP':[(chess.ROOK,   chess.WHITE), (chess.PAWN,   chess.WHITE),
               (chess.ROOK,   chess.BLACK), (chess.PAWN,   chess.BLACK)],
    'KBPvKBP':[(chess.BISHOP, chess.WHITE), (chess.PAWN,   chess.WHITE),
               (chess.BISHOP, chess.BLACK), (chess.PAWN,   chess.BLACK)],
    'KPPvKPP':[(chess.PAWN,   chess.WHITE), (chess.PAWN,   chess.WHITE),
               (chess.PAWN,   chess.BLACK), (chess.PAWN,   chess.BLACK)],
}

# Configs that only use 4-piece files (safe default if user only downloaded 4-piece)
CONFIGS_4PIECE = ['KRvKR', 'KBvKB', 'KNvKN', 'KPvKP', 'KRvKB', 'KRvKN', 'KBvKN', 'KQvKQ']
CONFIGS_5PIECE = ['KRPvKR', 'KBPvKB', 'KRvKBN', 'KBNvKR', 'KRPvKB', 'KRPvKN',
                  'KNPvKN', 'KBPvKN', 'KPPvKP']

def generate_random_position(pieces_config, rng):
    board = chess.Board.empty()
    all_pieces = [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK)] + pieces_config
    squares = rng.sample(chess.SQUARES, len(all_pieces))

    for (ptype, color), sq in zip(all_pieces, squares):
        if ptype == chess.PAWN:
            rank = chess.square_rank(sq)
            if rank == 0 or rank == 7:
                return None
        board.set_piece_at(sq, chess.Piece(ptype, color))

    board.turn = rng.choice([chess.WHITE, chess.BLACK])
    board.set_castling_fen('-')

    if board.was_into_check() or not board.is_valid():
        return None

    return board

# =============================================================================
# Binary writer
# =============================================================================
def write_positions_bin(positions, filename):
    """Write positions in standard binary format.
    
    Note: All positions from this script are tablebase draws (DrawSource.TABLEBASE = 7).
    The draw source is NOT embedded in the binary — the trainer identifies these
    positions by their source file. Phase labels are computed at load time.
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    phase_counts = [0, 0, 0]
    from training_format import write_header as tf_write_header
    with open(filename, 'wb') as f:
        tf_write_header(f, len(positions))
        for features, stm, result, eval_val in positions:
            f.write(struct.pack('<H', len(features)))
            for feat in features:
                f.write(struct.pack('<H', feat))
            f.write(struct.pack('B', stm))
            f.write(struct.pack('<ff', result, eval_val))
            phase_counts[classify_phase(features)] += 1
    print(f"Wrote {len(positions):,} positions to {filename}")
    print(f"  Draw source: TABLEBASE (all {len(positions):,} positions)")
    print(f"  Phase distribution:")
    for i, name in enumerate(PHASE_NAMES):
        pct = phase_counts[i] / max(len(positions), 1) * 100
        print(f"    {name:<12}: {phase_counts[i]:>8,}  ({pct:5.1f}%)")

def merge_bin_files(source, dest):
    from training_format import read_header as tf_read_header, write_header as tf_write_header, HEADER_SIZE
    _, src_count, src_offset = tf_read_header(source)
    _, dst_count, dst_offset = tf_read_header(dest)

    new_count = dst_count + src_count
    temp = dest + '.tmp'
    with open(dest, 'rb') as fin, open(source, 'rb') as src_f, open(temp, 'wb') as fout:
        # Write new versioned header
        tf_write_header(fout, new_count)
        # Copy dest data (skip its header)
        fin.seek(dst_offset)
        while True:
            chunk = fin.read(1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)
        # Copy source data (skip its header)
        src_f.seek(src_offset)
        while True:
            chunk = src_f.read(1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)
    shutil.move(temp, dest)
    print(f"Merged: {dest} now has {new_count:,} positions ({dst_count:,} original + {src_count:,} new tablebase draws)")

# =============================================================================
# Main
# =============================================================================
def generate(args):
    print(f"\nOpening Syzygy tablebases at: {args.syzygy_path}")
    try:
        tb = chess.syzygy.open_tablebase(args.syzygy_path)
    except Exception as e:
        print(f"ERROR opening tablebases: {e}")
        print("Make sure the path contains .rtbw files.")
        sys.exit(1)

    # Select configs
    if args.configs:
        configs = {}
        for name in args.configs:
            if name not in ENDGAME_CONFIGS:
                print(f"Unknown config: {name}. Available: {', '.join(ENDGAME_CONFIGS.keys())}")
                sys.exit(1)
            configs[name] = ENDGAME_CONFIGS[name]
    elif args.only_4piece:
        configs = {k: ENDGAME_CONFIGS[k] for k in CONFIGS_4PIECE}
    else:
        configs = ENDGAME_CONFIGS

    config_names = list(configs.keys())
    rng = random.Random(args.seed)

    print(f"Using {len(configs)} configs: {', '.join(config_names)}")
    print(f"Target: {args.target:,} drawn positions")
    print(f"Batch size: {args.batch_size:,} (write every N draws)")
    print()

    positions = []
    attempts = 0
    probe_calls = 0
    probe_errors = 0
    start_time = time.time()
    stats = {name: {'attempts': 0, 'draws': 0} for name in config_names}
    seen_fens = set()

    try:
        while len(positions) < args.target:
            config_name = rng.choice(config_names)
            config = configs[config_name]

            board = generate_random_position(config, rng)
            attempts += 1
            stats[config_name]['attempts'] += 1

            if board is None:
                continue

            fen = board.fen()
            if fen in seen_fens:
                continue
            # FIX 7.16: Cap seen_fens to prevent unbounded memory growth
            if len(seen_fens) > 1_000_000:
                seen_fens.clear()
            seen_fens.add(fen)

            # Probe local tablebase (fast!)
            try:
                wdl = tb.probe_wdl(board)
                probe_calls += 1
            except chess.syzygy.MissingTableError:
                # This config's tablebase file not downloaded, skip silently
                probe_errors += 1
                continue
            except Exception:
                probe_errors += 1
                continue

            # WDL: 0 = draw (also treat cursed win/blessed loss as draws: -1, 1)
            if wdl in (0, 1, -1):
                features = board_to_features(board)
                stm = 0 if board.turn == chess.WHITE else 1
                positions.append((features, stm, 0.5, 0.0))
                stats[config_name]['draws'] += 1

                if len(positions) % 500 == 0:
                    elapsed = time.time() - start_time
                    rate = len(positions) / elapsed if elapsed > 0 else 0
                    eta = (args.target - len(positions)) / rate if rate > 0 else 0
                    eta_str = f"{int(eta//3600)}h{int((eta%3600)//60)}m" if eta >= 3600 else \
                              f"{int(eta//60)}m{int(eta%60):02d}s" if eta >= 60 else f"{eta:.0f}s"
                    print(f"  {len(positions):,}/{args.target:,} draws "
                          f"({rate:.0f}/s, ETA: {eta_str}, {probe_errors} skipped)", end='\r')

    except KeyboardInterrupt:
        print(f"\n\nStopped. Saving {len(positions):,} positions...")

    tb.close()
    elapsed = time.time() - start_time

    print(f"\n\n{'='*60}")
    print(f"Done!")
    print(f"  Time:          {elapsed:.1f}s")
    print(f"  Attempts:      {attempts:,}")
    print(f"  Probe calls:   {probe_calls:,}")
    print(f"  Probe errors:  {probe_errors:,} (missing tablebase files)")
    print(f"  Draws found:   {len(positions):,}")
    print(f"  Speed:         {len(positions)/elapsed:.0f} draws/sec")
    if probe_calls > 0:
        print(f"  Draw rate:     {100*len(positions)/probe_calls:.1f}%")
    print(f"\n  Per-config:")
    for name in config_names:
        s = stats[name]
        dr = 100 * s['draws'] / max(s['attempts'], 1)
        print(f"    {name:12s}: {s['draws']:6,} draws / {s['attempts']:6,} ({dr:.0f}%)")
    print(f"{'='*60}\n")

    if not positions:
        print("No positions generated. Check your --syzygy-path and ensure .rtbw files exist.")
        return

    write_positions_bin(positions, args.output)

    if args.merge:
        if not os.path.exists(args.merge):
            print(f"Merge target {args.merge} not found - saving standalone only.")
        else:
            merge_bin_files(args.output, args.merge)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate drawn endgame positions from local Syzygy tablebases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DOWNLOAD SYZYGY FILES:
  4-piece WDL (~100MB, recommended minimum):
    https://tablebase.lichess.ovh/tables/standard/3-4-5/
    Download all *.rtbw files for 3/4-piece endings

  5-piece WDL (~9GB, for more variety):
    Same URL, download 5-piece *.rtbw files

  Or use the Fishtest rsync mirror:
    rsync -av --include='*.rtbw' --exclude='*' rsync://tablebase.sesse.net/syzygy/3-4-5/ ./syzygy/

QUICK START:
  # Only 4-piece configs (fastest, needs ~100MB of files):
  py -3.10 generate_draws.py --syzygy-path C:/syzygy --only-4piece --target 500000 --merge assets/training_data.bin

  # All configs (needs 5-piece files too):
  py -3.10 generate_draws.py --syzygy-path C:/syzygy --target 500000 --merge assets/training_data.bin
""")

    parser.add_argument('--syzygy-path', type=str, required=True,
                        help='Path to directory containing Syzygy .rtbw files')
    parser.add_argument('--target', type=int, default=500_000,
                        help='Number of drawn positions to generate (default: 500000)')
    parser.add_argument('--output', type=str, default='assets/drawn_endgames.bin',
                        help='Output binary file (default: assets/drawn_endgames.bin)')
    parser.add_argument('--merge', type=str, default=None,
                        help='After generation, merge into this existing .bin file')
    parser.add_argument('--configs', nargs='+', default=None,
                        help=f'Specific configs. Available: {", ".join(ENDGAME_CONFIGS.keys())}')
    parser.add_argument('--only-4piece', action='store_true',
                        help='Only use 4-piece configs (needs smaller ~100MB download)')
    parser.add_argument('--batch-size', type=int, default=10_000,
                        help='Progress report interval (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()
    generate(args)
