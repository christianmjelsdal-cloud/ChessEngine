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
    py -3.10 train_nnue.py --eval-soft-cap 8.0 --draw-weight 3.0 --mate-boost 3.0 ...

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
import tempfile
import atexit
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np


def _validate_training_data(game_result, stm, label=""):
    """INFO [7.24]: Sanity-check loaded training data for corruption."""
    import numpy as np
    gr = game_result if isinstance(game_result, np.ndarray) else np.asarray(game_result)
    valid_results = np.isin(gr.ravel()[:10000], [0.0, 0.5, 1.0])
    if not valid_results.all():
        n_bad = (~valid_results).sum()
        print(f"WARNING [{label}]: {n_bad}/10000 sampled game_result values not in {{0, 0.5, 1}}")
    if stm is not None:
        s = stm if isinstance(stm, np.ndarray) else np.asarray(stm)
        valid_stm = np.isin(s.ravel()[:10000], [0, 1])
        if not valid_stm.all():
            n_bad = (~valid_stm).sum()
            print(f"WARNING [{label}]: {n_bad}/10000 sampled stm values not in {{0, 1}}")

# ---------------------------------------------------------------------------
# Memory-mapped temp file management — avoids OOM for large array concatenation
# ---------------------------------------------------------------------------
_memmap_tmpdir = None
_memmap_tmpfiles = []

def _get_memmap_tmpdir():
    """Lazily create a temp directory for memmap files."""
    global _memmap_tmpdir
    if _memmap_tmpdir is None:
        _memmap_tmpdir = tempfile.mkdtemp(prefix='nnue_mm_')
    return _memmap_tmpdir

def _cleanup_memmap_files():
    """Remove all memmap temp files on exit."""
    import shutil
    for f in _memmap_tmpfiles:
        try:
            os.unlink(f)
        except OSError:
            pass
    if _memmap_tmpdir and os.path.isdir(_memmap_tmpdir):
        try:
            shutil.rmtree(_memmap_tmpdir, ignore_errors=True)
        except OSError:
            pass

atexit.register(_cleanup_memmap_files)

# Threshold: arrays larger than 256 MB use disk-backed memmap instead of RAM
_MEMMAP_THRESHOLD_BYTES = 256 * 1024 * 1024

def _make_memmap(shape, dtype=np.int32, tag='arr'):
    """Create a memory-mapped temp file and return the np.memmap array."""
    dt = np.dtype(dtype)
    tmpfile = os.path.join(_get_memmap_tmpdir(),
                           f'{tag}_{len(_memmap_tmpfiles)}.dat')
    _memmap_tmpfiles.append(tmpfile)
    return np.memmap(tmpfile, dtype=dt, mode='w+', shape=shape)

def _safe_concat_np(arrays, tag='concat'):
    # INFO [7.21]: Validate that all arrays have consistent dtype and shape
    if len(arrays) > 1:
        expected_dtype = arrays[0].dtype
        expected_shape = arrays[0].shape[1:]
        for i, a in enumerate(arrays[1:], 1):
            if a.dtype != expected_dtype:
                raise ValueError(f"_safe_concat_np: array {{i}} dtype {{a.dtype}} != expected {{expected_dtype}}")
            if a.shape[1:] != expected_shape:
                raise ValueError(f"_safe_concat_np: array {{i}} shape[1:] {{a.shape[1:]}} != expected {{expected_shape}}")
    """Concatenate numpy arrays; uses memmap if the result would be large.

    For small results (< 256 MB), behaves identically to np.concatenate.
    For large results, writes into a disk-backed memmap to avoid OOM.
    """
    if len(arrays) == 1:
        return arrays[0]
    total_rows = sum(a.shape[0] for a in arrays)
    rest_shape = arrays[0].shape[1:]
    dt = arrays[0].dtype
    nbytes = int(np.prod((total_rows,) + rest_shape, dtype=np.int64)) * dt.itemsize
    if nbytes <= _MEMMAP_THRESHOLD_BYTES:
        return np.concatenate(arrays)
    # Disk-backed path
    result = _make_memmap((total_rows,) + rest_shape, dtype=dt, tag=tag)
    offset = 0
    for arr in arrays:
        n = arr.shape[0]
        result[offset:offset + n] = arr
        offset += n
    result.flush()
    return result

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
# HalfKAv2 Feature Encoding Constants
# =============================================================================
# HalfKAv2: features indexed by (king_square, piece_type, piece_square) per perspective
# 64 king buckets × 10 non-king piece types × 64 squares = 40,960 features
HALFKAV2_FEATURES = 64 * 10 * 64  # = 40,960
PIECES_PER_PERSPECTIVE = 10  # 5 own non-king + 5 opponent non-king
SQUARES = 64

# =============================================================================
# Constants - must match C++ NNUE.h exactly
# =============================================================================
NUM_FEATURES = HALFKAV2_FEATURES  # 40,960 (HalfKAv2 encoding)
NUM_FEATURES_768 = 768  # legacy encoding used in binary files
L1_SIZE = 512   # must match C++ NNUE.h
L2_SIZE = 128   # must match C++ NNUE.h
L3_SIZE = 64    # must match C++ NNUE.h
WDL_SIZE = 3    # win/draw/loss phase heads

# Datasets larger than this use streaming mode (saves RAM)
MAX_PRELOAD_POSITIONS = 8_000_000

# Chunk size for streaming mode
STREAM_CHUNK_SIZE = 200_000
MAX_ACTIVE_FEATURES = 64  # max active HalfKAv2 features per side per position (actual ~30)

def decode_768_feature(feat):
    """Decode a 768-encoding feature index into (piece_type, piece_color, square).
    piece_type: 0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King
    piece_color: 0=White, 1=Black
    square: 0-63 (rank*8 + file)
    """
    piece_index = feat // 64
    square = feat % 64
    piece_color = 0 if piece_index < 6 else 1
    piece_type = piece_index % 6
    return piece_type, piece_color, square

def halfkav2_feature(king_sq, piece_type, piece_color, piece_sq, perspective):
    """Compute HalfKAv2 feature index for a single piece.
    
    perspective: 0=White, 1=Black
    For the given perspective:
      - Own pieces use indices 0-4 (pawn, knight, bishop, rook, queen)
      - Opponent pieces use indices 5-9
      - If perspective is Black, squares are vertically mirrored (rank flipped)
    
    Returns feature index in [0, 40959] or -1 if this is a king.
    """
    if piece_type == 5:  # King - not encoded as a feature
        return -1
    
    # Determine piece index relative to perspective
    is_own_piece = (piece_color == perspective)
    piece_idx = piece_type  # 0-4 for pawn..queen
    if not is_own_piece:
        piece_idx += 5  # 5-9 for opponent pieces
    
    # Mirror squares for black perspective
    if perspective == 1:
        king_sq = king_sq ^ 56  # flip rank: rank*8+file -> (7-rank)*8+file
        piece_sq = piece_sq ^ 56
    
    return king_sq * 640 + piece_idx * 64 + piece_sq

def convert_768_to_halfkav2(features_768):
    """Convert a list of 768-encoding feature indices to HalfKAv2 features.
    
    Returns (white_features, black_features) as lists of HalfKAv2 feature indices.
    """
    # First pass: find king squares
    white_king_sq = -1
    black_king_sq = -1
    pieces = []  # list of (piece_type, piece_color, square)
    
    for feat in features_768:
        pt, pc, sq = decode_768_feature(feat)
        if pt == 5:  # King
            if pc == 0:
                white_king_sq = sq
            else:
                black_king_sq = sq
        else:
            pieces.append((pt, pc, sq))
    
    # Second pass: compute HalfKAv2 features for both perspectives
    white_feats = []
    black_feats = []
    
    for pt, pc, sq in pieces:
        wf = halfkav2_feature(white_king_sq, pt, pc, sq, perspective=0)
        bf = halfkav2_feature(black_king_sq, pt, pc, sq, perspective=1)
        if wf >= 0:
            white_feats.append(wf)
        if bf >= 0:
            black_feats.append(bf)
    
    return white_feats, black_feats


def densify_batch(idx_np):
    """Convert sparse (n, MAX_ACTIVE_FEATURES) int32 numpy array to dense (n, NUM_FEATURES) float32 tensor.

    Padding value is -1. Only valid (non-negative) feature indices are written to avoid
    last-write-wins corruption at index 0 from padding entries.
    """
    idx = torch.from_numpy(idx_np.astype(np.int64))     # (n, MAX_ACTIVE_FEATURES)
    mask = idx >= 0                                       # (n, MAX_ACTIVE_FEATURES) bool
    # Build row indices matching the shape, then select only valid entries
    row_idx = torch.arange(idx.shape[0], dtype=torch.long).unsqueeze(1).expand_as(idx)
    valid_rows = row_idx[mask]                            # (num_valid,)
    valid_cols = idx[mask]                                # (num_valid,)  — all >= 0
    dense = torch.zeros(idx_np.shape[0], NUM_FEATURES, dtype=torch.float32)
    dense[valid_rows, valid_cols] = 1.0
    return dense


# =============================================================================
# Feature mirror - must match C++ mirrorFeature() exactly
# Mirror table for legacy 768-encoding (used to decode binary files)
# =============================================================================
def build_mirror_table():
    table = [0] * NUM_FEATURES_768
    for feat in range(NUM_FEATURES_768):
        piece_index = feat // 64
        square_index = feat % 64
        rank = square_index // 8
        col = square_index % 8
        mirrored_piece = (piece_index + 6) if piece_index < 6 else (piece_index - 6)
        mirrored_rank = 7 - rank
        table[feat] = mirrored_piece * 64 + mirrored_rank * 8 + col
    return table

MIRROR_TABLE = build_mirror_table()
MIRROR_TABLE_NP = np.array(MIRROR_TABLE, dtype=np.int64)

# F6.4: Removed unused compute_material_phase() and classify_phase() scalar functions
# (each consumer script has its own copy; vectorized version used in training)

# =============================================================================
# Draw Position Tagging
# =============================================================================
class DrawSource:
    """Draw source type identifiers."""
    NOT_DRAW = 0
    UNKNOWN_DRAW = 1      # draw from game result, source unknown
    STALEMATE = 2
    INSUFFICIENT = 3
    FIFTY_MOVE = 4
    REPETITION = 5
    AGREEMENT = 6
    TABLEBASE = 7          # from Syzygy tablebase (generate_draws.py)

def tag_draw_positions(game_results, draw_source_array=None):
    """Tag drawn positions from game results.
    
    Args:
        game_results: numpy array of game results (0.0, 0.5, 1.0)
        draw_source_array: optional numpy array of draw source types
    
    Returns:
        is_draw: boolean numpy array
        draw_sources: numpy array of DrawSource values
    """
    is_draw = np.abs(game_results - 0.5) < 0.01
    if draw_source_array is not None:
        draw_sources = draw_source_array
    else:
        draw_sources = np.where(is_draw, DrawSource.UNKNOWN_DRAW, DrawSource.NOT_DRAW)
    return is_draw, draw_sources

# =============================================================================
# Sample Rebalancing Weights (NEW in v3)
# =============================================================================
def compute_sample_weights(search_eval, game_result, eval_soft_cap, draw_weight,
                           mate_boost=0.0):
    """
    Per-sample loss weights to fix dataset imbalances:
      - eval_soft_cap: positions with |eval| > cap get weight = cap / |eval|
        (reduces dominance of trivially won/lost positions)
      - draw_weight: multiplier for drawn positions (result ~ 0.5)
        (compensates for draw underrepresentation)
      - mate_boost: multiplier for high-eval (|eval|>2000cp) decisive positions.
        Counteracts soft-cap to teach mate patterns. 0 = disabled.
    Returns None if no rebalancing is needed (all disabled).
    """
    if eval_soft_cap <= 0 and draw_weight == 1.0 and mate_boost <= 0:  # FIX: != 1.0 means active
        return None

    weights = torch.ones_like(search_eval)

    # Soft-cap extreme evals: linear decay beyond threshold
    if eval_soft_cap > 0:
        abs_eval = torch.abs(search_eval)
        eval_w = torch.clamp(eval_soft_cap / torch.clamp(abs_eval, min=0.01), max=1.0)
        weights = weights * eval_w

    # Mate-boost: upweight high-eval positions in decisive games
    # This counteracts soft-cap for positions near checkmate
    if mate_boost > 0:
        abs_eval = torch.abs(search_eval)
        is_decisive = (game_result < 0.1) | (game_result > 0.9)
        is_high_eval = abs_eval > 2000.0
        mate_mask = is_decisive & is_high_eval
        weights = torch.where(mate_mask, weights * mate_boost, weights)

    # Upweight draws
    if draw_weight != 1.0:  # FIX: works for both suppression (< 1.0) and boost (> 1.0)
        is_draw = (game_result > 0.4) & (game_result < 0.6)
        weights = torch.where(is_draw, weights * draw_weight, weights)

    return weights

# =============================================================================
# Binary format reader - scan offsets and load positions
# =============================================================================

NNUE_MAGIC = b'NNUE'

def _read_binary_header(f_or_mm):
    """Detect versioned (PT-3) vs legacy header and return (num_positions, data_start_offset).

    Versioned format: 4-byte magic 'NNUE' + 1-byte version + 4-byte count  (9 bytes)
    Legacy   format: 4-byte count                                           (4 bytes)
    """
    if hasattr(f_or_mm, 'read'):            # file object
        raw = f_or_mm.read(9)
        if raw[:4] == NNUE_MAGIC:
            version = raw[4]
            num_positions = struct.unpack_from('<I', raw, 5)[0]
            return num_positions, 9
        else:
            num_positions = struct.unpack_from('<I', raw, 0)[0]
            return num_positions, 4
    else:                                    # mmap / bytes
        if f_or_mm[:4] == NNUE_MAGIC:
            version = f_or_mm[4]
            num_positions = struct.unpack_from('<I', f_or_mm, 5)[0]
            return num_positions, 9
        else:
            num_positions = struct.unpack_from('<I', f_or_mm, 0)[0]
            return num_positions, 4
def _build_offset_index(filename):
    """Scan every record to build a complete byte-offset array. Slow but only runs once."""
    import numpy as np
    with open(filename, 'rb') as f:
        num_positions, data_start = _read_binary_header(f)
        print(f"  Total positions: {num_positions:,} (header: {'versioned' if data_start == 9 else 'legacy'})")
        offsets = np.empty(num_positions, dtype=np.int64)
        offset = data_start
        f.seek(offset)
        for i in range(num_positions):
            offsets[i] = offset
            header = f.read(2)
            if len(header) < 2:
                print(f"  Warning: truncated at position {i}")
                offsets = offsets[:i]
                break
            num_features = struct.unpack_from('<H', header, 0)[0]
            skip = 2 * num_features + 1 + 8
            f.seek(skip, 1)
            offset += 2 + skip
            if (i + 1) % 1_000_000 == 0:
                print(f"\r  Scanning {i+1:,}/{num_positions:,}...\033[K", end='', flush=True)
        print(f"\r  Scan complete: {len(offsets):,} positions indexed.\033[K")
    return offsets

def _load_or_build_offset_cache(filename):
    """Load cached offset index if valid, otherwise build and cache it."""
    import numpy as np
    cache_path = filename + ".offidx"
    data_mtime = os.path.getmtime(filename)

    # Try loading existing cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as cf:
                magic = cf.read(8)
                if magic == b'OFFIDX01':
                    saved_mtime = struct.unpack('<d', cf.read(8))[0]
                    saved_count = struct.unpack('<Q', cf.read(8))[0]
                    if abs(saved_mtime - data_mtime) < 0.01:
                        offsets = np.frombuffer(cf.read(), dtype=np.int64)
                        if len(offsets) == saved_count:
                            print(f"  Loaded cached index: {len(offsets):,} offsets ({cache_path})")
                            return offsets
            print(f"  Cache stale or corrupt, rebuilding...")
        except Exception as e:
            print(f"  Cache read error ({e}), rebuilding...")

    # Build from scratch
    print(f"  Building offset index (first time, may take several minutes)...")
    offsets = _build_offset_index(filename)

    # Save cache
    try:
        with open(cache_path, 'wb') as cf:
            cf.write(b'OFFIDX01')                               # magic
            cf.write(struct.pack('<d', data_mtime))             # data file mtime
            cf.write(struct.pack('<Q', len(offsets)))            # count
            cf.write(offsets.tobytes())                          # offset array
        cache_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"  Saved offset cache: {cache_mb:.1f} MB ({cache_path})")
    except Exception as e:
        print(f"  Warning: could not save cache ({e}) - will rescan next time")

    return offsets


# =============================================================================
# Data Sharding - one-time preprocessing for faster training initialization
# =============================================================================

def save_shards_cfg(shard_dir, shard_info):
    """Write shards.cfg manifest file."""
    cfg_path = os.path.join(shard_dir, 'shards.cfg')
    with open(cfg_path, 'w') as f:
        f.write(f"# Sharded training data manifest\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"num_shards={shard_info['num_shards']}\n")
        f.write(f"total_positions={shard_info['total_positions']}\n")
        f.write(f"source_file={shard_info['source_file']}\n")
        f.write(f"seed={shard_info['seed']}\n")
        for i, (path, count) in enumerate(shard_info['shards']):
            f.write(f"shard_{i}={os.path.basename(path)},{count}\n")
    print(f"  Saved manifest: {cfg_path}")


def load_shards_cfg(shard_dir):
    """Read shards.cfg manifest and return shard info dict."""
    cfg_path = os.path.join(shard_dir, 'shards.cfg')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No shards.cfg found in {shard_dir}")
    
    info = {'shards': []}
    with open(cfg_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, val = line.split('=', 1)
            if key == 'num_shards':
                info['num_shards'] = int(val)
            elif key == 'total_positions':
                info['total_positions'] = int(val)
            elif key == 'source_file':
                info['source_file'] = val
            elif key == 'seed':
                info['seed'] = int(val)
            elif key.startswith('shard_'):
                fname, count = val.split(',')
                full_path = os.path.join(shard_dir, fname)
                info['shards'].append((full_path, int(count)))
    return info


def shard_training_data(input_file, num_shards=10, output_dir=None, seed=None):
    """One-time preprocessing: shuffle and split a .bin file into sharded .bin files with .offidx caches.
    
    Pipeline:
    1. Load/build offset index for source file
    2. Shuffle all position indices
    3. Split into N groups
    4. For each group: read positions from source via mmap, write to new .bin file, build .offidx
    5. Write shards.cfg manifest
    """
    import mmap
    
    if seed is None:
        seed = int(time.time()) % (2**31)
    rng = np.random.RandomState(seed)
    
    if output_dir is None:
        output_dir = os.path.splitext(input_file)[0] + '_shards'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Sharding Training Data")
    print(f"{'='*60}")
    print(f"  Source:     {input_file}")
    print(f"  Shards:     {num_shards}")
    print(f"  Output dir: {output_dir}")
    print(f"  Seed:       {seed}")
    print()
    
    # Step 1: Load all offsets
    print("Step 1/4: Loading offset index...")
    all_offsets = _load_or_build_offset_cache(input_file)
    total = len(all_offsets)
    print(f"  {total:,} positions indexed")
    
    # Step 2: Shuffle
    print("Step 2/4: Shuffling positions...")
    indices = np.arange(total, dtype=np.int64)
    rng.shuffle(indices)
    shuffled_offsets = all_offsets[indices]
    
    # Step 3-4: Split and write shards
    print(f"Step 3/4: Writing {num_shards} shards...")
    shard_sizes = [(total // num_shards) + (1 if i < total % num_shards else 0) 
                   for i in range(num_shards)]
    
    shard_manifest = []
    pos_idx = 0
    
    with open(input_file, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        for shard_i in range(num_shards):
            shard_count = shard_sizes[shard_i]
            shard_name = f"shard_{shard_i+1:03d}.bin"
            shard_path = os.path.join(output_dir, shard_name)
            
            # Get this shard's offsets (sort for sequential source reads)
            shard_offsets = shuffled_offsets[pos_idx:pos_idx + shard_count]
            sorted_shard_offsets = np.sort(shard_offsets)
            pos_idx += shard_count
            
            # Write shard .bin file
            new_offsets = np.empty(shard_count, dtype=np.int64)
            with open(shard_path, 'wb') as sf:
                sf.write(NNUE_MAGIC)                       # 4-byte magic
                sf.write(struct.pack('<B', 1))              # version 1
                sf.write(struct.pack('<I', shard_count))    # position count
                write_offset = 9
                
                for j, src_off in enumerate(sorted_shard_offsets):
                    new_offsets[j] = write_offset
                    # Read record from source
                    num_features = struct.unpack_from('<H', mm, src_off)[0]
                    record_size = 2 + 2 * num_features + 1 + 8
                    sf.write(mm[src_off:src_off + record_size])
                    write_offset += record_size
            
            # Write .offidx cache for shard
            offidx_path = shard_path + ".offidx"
            shard_mtime = os.path.getmtime(shard_path)
            with open(offidx_path, 'wb') as cf:
                cf.write(b'OFFIDX01')
                cf.write(struct.pack('<d', shard_mtime))
                cf.write(struct.pack('<Q', shard_count))
                cf.write(new_offsets.tobytes())
            
            shard_manifest.append((shard_path, shard_count))
            
            file_mb = os.path.getsize(shard_path) / (1024 * 1024)
            idx_mb = os.path.getsize(offidx_path) / (1024 * 1024)
            print(f"  [{shard_i+1}/{num_shards}] {shard_name}: {shard_count:,} positions "
                  f"({file_mb:.1f} MB data + {idx_mb:.1f} MB index)")
        
        mm.close()
    
    # Step 4: Write manifest
    print("Step 4/4: Writing manifest...")
    save_shards_cfg(output_dir, {
        'num_shards': num_shards,
        'total_positions': total,
        'source_file': os.path.abspath(input_file),
        'seed': seed,
        'shards': shard_manifest,
    })
    
    total_data_mb = sum(os.path.getsize(p) for p, _ in shard_manifest) / (1024 * 1024)
    total_idx_mb = sum(os.path.getsize(p + ".offidx") for p, _ in shard_manifest) / (1024 * 1024)
    
    print(f"\n{'='*60}")
    print(f"Sharding complete!")
    print(f"  {total:,} positions -> {num_shards} shards")
    print(f"  Data: {total_data_mb:.1f} MB  |  Index: {total_idx_mb:.1f} MB")
    print(f"  Output: {output_dir}")
    print(f"\nTo train from shards, use:")
    print(f"  --data \"{output_dir}\"")
    print(f"{'='*60}\n")


# =============================================================================
# Vectorized HalfKAv2 conversion (chunked for memory safety)
# =============================================================================

_HALFKAV2_CHUNK = 250_000  # positions per chunk — ~600 MB peak per chunk
# AUDIT FIX: Reduced from 2M to 250K so parallel workers don't OOM.
# Each chunk creates ~8-10 intermediate (N, MAX_ACTIVE_FEATURES=64) int32 arrays.
# Per chunk: ~250K × 64 × 4B × 10 arrays ≈ 610 MB.
# With 2 workers: ~1.2 GB total intermediates — safe on 15 GB systems.

def _vectorized_halfkav2_chunk(feat_padded, valid_mask):
    """Convert one chunk of raw 768-features to HalfKAv2 indices.
    
    Args:
        feat_padded: (N, MAX_RAW) uint16 array of raw features (0xFFFF = padding)
        valid_mask:  (N, MAX_RAW) bool array (True where feat != 0xFFFF)
    
    Returns:
        (all_white_idx, all_black_idx, all_phases) — all (N, ...) int32 arrays
    """
    _phase_weights = np.array([0, 1, 1, 2, 4, 0], dtype=np.int32)
    N = len(feat_padded)

    feats = feat_padded.astype(np.int32)

    piece_indices = feats // 64
    squares = feats % 64
    piece_types = piece_indices % 6
    piece_colors = (piece_indices >= 6).astype(np.int32)

    king_mask = (piece_types == 5) & valid_mask
    white_king_mask = king_mask & (piece_colors == 0)
    black_king_mask = king_mask & (piece_colors == 1)

    wk_idx = np.argmax(white_king_mask, axis=1)
    bk_idx = np.argmax(black_king_mask, axis=1)

    # F7.5: Validate that every position has at least one white and one black king.
    # np.argmax returns 0 when no element is True, which silently produces wrong features.
    has_wk = white_king_mask.any(axis=1)
    has_bk = black_king_mask.any(axis=1)
    if not (has_wk.all() and has_bk.all()):
        n_bad = int((~has_wk | ~has_bk).sum())
        print(f"WARNING: {n_bad} positions missing king(s) in HalfKAv2 chunk — features will be incorrect")

    arange_N = np.arange(N)
    wk_sq = squares[arange_N, wk_idx]
    bk_sq = squares[arange_N, bk_idx]

    non_king = valid_mask & ~king_mask

    w_piece_idx = np.where(piece_colors == 0, piece_types, piece_types + 5)
    w_feats = wk_sq[:, None] * 640 + w_piece_idx * 64 + squares

    bk_sq_m = bk_sq ^ 56
    squares_m = squares ^ 56
    b_piece_idx = np.where(piece_colors == 1, piece_types, piece_types + 5)
    b_feats = bk_sq_m[:, None] * 640 + b_piece_idx * 64 + squares_m

    non_king_cumsum = np.cumsum(non_king, axis=1) - 1
    write_mask = non_king & (non_king_cumsum < MAX_ACTIVE_FEATURES)

    row_idx, col_idx_raw = np.where(write_mask)
    dest_col = non_king_cumsum[row_idx, col_idx_raw].astype(np.intp)

    all_white_idx = np.full((N, MAX_ACTIVE_FEATURES), -1, dtype=np.int32)
    all_black_idx = np.full((N, MAX_ACTIVE_FEATURES), -1, dtype=np.int32)
    all_white_idx[row_idx, dest_col] = w_feats[row_idx, col_idx_raw].astype(np.int32)
    all_black_idx[row_idx, dest_col] = b_feats[row_idx, col_idx_raw].astype(np.int32)

    phase_contrib = np.where(non_king, _phase_weights[piece_types], 0)
    all_phase_val = np.sum(phase_contrib, axis=1).astype(np.int32)
    all_phases = np.where(all_phase_val >= 20, 0, np.where(all_phase_val >= 8, 1, 2)).astype(np.int32)

    return all_white_idx, all_black_idx, all_phases


def _vectorized_halfkav2(feat_padded, all_stm, all_result, all_eval, filter_eval_max=0.0, quiet=False):
    """Chunked vectorized HalfKAv2 conversion — memory-safe for large datasets.
    
    Processes positions in chunks of _HALFKAV2_CHUNK to keep peak memory at ~2.4 GB
    instead of ~22 GB for 18M+ positions.

    Args:
        quiet: If True, suppress progress lines (used when loading in parallel).
    """
    N = len(feat_padded)
    t1 = time.time()

    if N <= _HALFKAV2_CHUNK:
        # Small dataset — single pass, no overhead
        if not quiet:
            print(f"  Vectorized HalfKAv2 conversion ({N:,} positions)...")
        valid_mask = feat_padded != 0xFFFF
        w, b, ph = _vectorized_halfkav2_chunk(feat_padded, valid_mask)
        del valid_mask
    else:
        # Large dataset — process in chunks, write directly to memmap to avoid OOM.
        # Without memmap the chunk list + np.concatenate would need 2× the final
        # array size in RAM (e.g. 18 M × 64 × 4 bytes × 2 arrays × 2 = ~17 GB).
        num_chunks = (N + _HALFKAV2_CHUNK - 1) // _HALFKAV2_CHUNK
        if not quiet:
            print(f"  Vectorized HalfKAv2 conversion ({N:,} positions in {num_chunks} chunks, memmap-backed)...")

        w  = _make_memmap((N, MAX_ACTIVE_FEATURES), np.int32, tag='white')
        b  = _make_memmap((N, MAX_ACTIVE_FEATURES), np.int32, tag='black')
        ph = _make_memmap((N,), np.int32, tag='phase')
        # NOTE: No pre-fill needed — every element is written by the chunk loop below.
        # The old w[:] = -1 / b[:] = -1 forced ~17 GB of dirty pages into RAM, causing OOM.

        for ci in range(num_chunks):
            start = ci * _HALFKAV2_CHUNK
            end = min(start + _HALFKAV2_CHUNK, N)
            chunk_feat = feat_padded[start:end]
            chunk_valid = chunk_feat != 0xFFFF  # compute valid_mask per-chunk (saves ~1 GB)
            wc, bc, phc = _vectorized_halfkav2_chunk(chunk_feat, chunk_valid)
            w[start:end] = wc
            b[start:end] = bc
            ph[start:end] = phc
            del chunk_feat, chunk_valid, wc, bc, phc  # free chunk immediately
            if not quiet:
                elapsed_c = time.time() - t1
                rate_c = end / elapsed_c if elapsed_c > 0 else 0
                print(f"\r    Chunk {ci+1}/{num_chunks}: {end:,}/{N:,} ({rate_c:,.0f} pos/s)\033[K", end='', flush=True)
        if not quiet:
            print()
        w.flush(); b.flush(); ph.flush()

    # Free raw inputs — no longer needed after conversion
    del feat_padded
    import gc; gc.collect()

    t_conv = time.time() - t1
    if not quiet:
        print(f"  HalfKAv2 conversion: {t_conv:.1f}s ({N / t_conv:,.0f} pos/s)")

    # Filter extreme evaluations if requested
    kept = N
    if filter_eval_max > 0:
        mask = np.abs(all_eval) <= filter_eval_max
        kept = int(mask.sum())
        if kept < N:
            # For memmap-backed arrays, fancy-indexing would copy to RAM.
            # If the result is still large, write to a new memmap instead.
            kept_bytes = int(kept) * MAX_ACTIVE_FEATURES * 4
            if kept_bytes > _MEMMAP_THRESHOLD_BYTES:
                indices = np.where(mask)[0]
                w_new = _make_memmap((kept, MAX_ACTIVE_FEATURES), np.int32, tag='white_filt')
                b_new = _make_memmap((kept, MAX_ACTIVE_FEATURES), np.int32, tag='black_filt')
                # Copy in slices to avoid large temp arrays
                _COPY_CHUNK = 500_000
                for s in range(0, kept, _COPY_CHUNK):
                    e = min(s + _COPY_CHUNK, kept)
                    idx_slice = indices[s:e]
                    w_new[s:e] = w[idx_slice]
                    b_new[s:e] = b[idx_slice]
                w_new.flush(); b_new.flush()
                w = w_new; b = b_new
            else:
                w = w[mask]
                b = b[mask]
            all_stm = all_stm[mask]
            all_result = all_result[mask]
            all_eval = all_eval[mask]
            ph = np.array(ph[mask])  # phases is small (1D int32)

    if not quiet:
        print(f"  Converting to tensors...")

    phase_counts = np.bincount(ph[:kept], minlength=3)
    if not quiet:
        print(f"Phase distribution: Opening={phase_counts[0]:,}, Middlegame={phase_counts[1]:,}, Endgame={phase_counts[2]:,}")

    return {
        'white':  w,
        'black':  b,
        'stm':    torch.from_numpy(all_stm[:kept].copy()) if kept < N else torch.from_numpy(all_stm),
        'result': torch.from_numpy(all_result[:kept].copy()) if kept < N else torch.from_numpy(all_result),
        'eval':   torch.from_numpy(all_eval[:kept].copy()) if kept < N else torch.from_numpy(all_eval),
        'phases': ph,
        'count':  kept,
    }


def scan_positions(filename, max_positions=0):
    """Fast scan: builds list of byte offsets for random access without loading data.
    
    Uses a cached offset index (.offidx file) to avoid rescanning the entire file
    on every generation. First call builds the cache (~10 min for 66M positions),
    subsequent calls load it in ~1 second.
    
    If max_positions > 0 and max_positions < total, uses reservoir sampling to
    return exactly max_positions uniformly random offsets.
    """
    import numpy as np
    print(f"Scanning {filename} for position offsets...")

    all_offsets = _load_or_build_offset_cache(filename)
    num_positions = len(all_offsets)

    if 0 < max_positions < num_positions:
        # Reservoir sampling from cached offsets (instant in-memory)
        indices = np.random.choice(num_positions, size=max_positions, replace=False)
        sampled = all_offsets[indices]           # numpy array — no tolist() (T-M1)
        print(f"  Sampled {max_positions:,} from {num_positions:,} positions (reservoir)")
        return num_positions, sampled
    else:
        print(f"  Using all {num_positions:,} positions")
        return num_positions, all_offsets        # numpy array — no tolist() (T-M1)

def load_positions_at_offsets(filename, offsets, filter_eval_max=0.0, quiet=False):
    """
    Load positions by byte offsets using memory-mapped file.
    Offsets should be pre-sorted for sequential disk reads.
    If filter_eval_max > 0, positions with |eval| > threshold are excluded.

    Args:
        quiet: If True, suppress \\r progress lines (used when loading in parallel).
    """
    n = len(offsets)

    MAX_RAW = 32
    feat_padded = np.full((n, MAX_RAW), 0xFFFF, dtype=np.uint16)
    all_stm    = np.zeros(n, dtype=np.float32)
    all_result = np.zeros(n, dtype=np.float32)
    all_eval   = np.zeros(n, dtype=np.float32)

    import mmap
    t0 = time.time()

    # Phase 1: Fast raw data extraction from mmap
    with open(filename, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for i, off in enumerate(offsets):
            num_features = struct.unpack_from('<H', mm, off)[0]
            pos = off + 2
            n_copy = min(num_features, MAX_RAW)
            feat_padded[i, :n_copy] = np.frombuffer(mm, dtype=np.uint16, count=n_copy, offset=pos)
            pos += 2 * num_features
            stm = mm[pos]
            pos += 1
            game_result, search_eval = struct.unpack_from('<ff', mm, pos)

            all_stm[i]    = float(stm)
            all_result[i] = game_result
            all_eval[i]   = search_eval

            if not quiet and (i + 1) % 50_000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n - i - 1) / rate if rate > 0 else 0
                print(f"\r    Parsed {i+1:,}/{n:,} positions ({rate:.0f} pos/s, ETA {eta:.0f}s)\033[K", end='', flush=True)

        mm.close()

    if not quiet and n >= 50_000:
        print()

    t_parse = time.time() - t0
    if not quiet:
        rate = f"{n / t_parse:,.0f} pos/s" if t_parse > 0 else "instant"
        print(f"  Parsed {n:,} positions in {t_parse:.1f}s ({rate})")

    # Phase 2: Chunked vectorized HalfKAv2 conversion
    return _vectorized_halfkav2(feat_padded, all_stm, all_result, all_eval,
                                 filter_eval_max=filter_eval_max, quiet=quiet)

# =============================================================================
# Fast preload (small datasets)
# =============================================================================
def load_training_data(filename, max_positions=0, filter_eval_max=0.0, quiet=False):
    """Load positions from file — optimized with bulk I/O and numpy vectorization.

    When max_positions is set and the file contains many more positions than needed,
    uses the offset-cache fast path (scan_positions + load_positions_at_offsets) to
    avoid reading the entire file into RAM. This prevents disk 100% / memory spikes
    on large shard files.

    The offset cache (.offidx) is built once per file on first use (~10s for a 30M
    position shard) and reloaded instantly on subsequent calls.

    Args:
        quiet: If True, suppress \\r progress lines (used when loading in parallel
               threads to avoid terminal flicker).
    """
    if not quiet:
        print(f"Loading positions from {filename}...")

    # Fast path: if we need far fewer positions than the file contains, use the
    # offset-cache + mmap approach to avoid reading the whole file into RAM.
    # Threshold: file must have 4x more positions than requested to justify building
    # the cache (building the cache requires one sequential scan anyway).
    if max_positions > 0:
        with open(filename, 'rb') as _hf:
            _raw_hdr = _hf.read(9)
        total_in_file, _ = _read_binary_header(_raw_hdr)
        if total_in_file > max_positions:
            # FIX 7.3: Always use offset-cache fast path when sampling fewer positions
            # than the file contains. Previous threshold of 4x meant the standard path
            # still scanned all records sequentially (O(N)) for files with 1-4x the
            # needed positions, defeating the purpose of sampling.
            if not quiet:
                print(f"  Total positions: {total_in_file:,} — using offset-cache fast path "
                      f"(avoids reading {total_in_file * 50 // (1024*1024):.0f} MB into RAM)")
            _, sampled_offsets = scan_positions(filename, max_positions)
            sampled_offsets_sorted = np.sort(sampled_offsets)  # sequential reads = faster
            return load_positions_at_offsets(filename, sampled_offsets_sorted,
                                             filter_eval_max=filter_eval_max, quiet=quiet)

    # Standard path: memory-map the file (avoids copying entire file into Python heap)
    import mmap as _mmap
    with open(filename, 'rb') as f:
        _hdr = f.read(9)                       # tiny read for header
    total_in_file, data_start = _read_binary_header(_hdr)
    if not quiet:
        print(f"  Total positions: {total_in_file:,} (header: {'versioned' if data_start == 9 else 'legacy'})")

    # Determine how many to load (with optional reservoir sampling)
    use_reservoir = (max_positions > 0 and max_positions < total_in_file)
    if use_reservoir:
        reservoir_indices = set(random.sample(range(total_in_file), max_positions))
        num_to_load = max_positions
        if not quiet:
            print(f"  Capped to:       {num_to_load:,} (--max-positions, reservoir sampled)")
    else:
        num_to_load = total_in_file

    offset = data_start
    kept = 0
    scan_count = total_in_file if use_reservoir else num_to_load
    t0 = time.time()

    # Phase 1: Fast raw data extraction — mmap keeps data on disk, OS pages in on demand
    MAX_RAW = 32  # max features per position in 768-encoding
    feat_padded = np.full((num_to_load, MAX_RAW), 0xFFFF, dtype=np.uint16)
    all_stm    = np.zeros(num_to_load, dtype=np.float32)
    all_result = np.zeros(num_to_load, dtype=np.float32)
    all_eval   = np.zeros(num_to_load, dtype=np.float32)

    with open(filename, 'rb') as f:
        mm = _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ)
        try:
            for i in range(scan_count):
                nf = struct.unpack_from('<H', mm, offset)[0]

                if use_reservoir and i not in reservoir_indices:
                    offset += 2 + 2 * nf + 1 + 8
                    continue

                feat_offset = offset + 2
                meta_offset = feat_offset + 2 * nf
                stm = mm[meta_offset]
                game_result, search_eval = struct.unpack_from('<ff', mm, meta_offset + 1)
                offset = meta_offset + 9

                if filter_eval_max > 0 and abs(search_eval) > filter_eval_max:
                    continue

                n_copy = min(nf, MAX_RAW)
                feat_padded[kept, :n_copy] = np.frombuffer(mm, dtype=np.uint16, count=n_copy, offset=feat_offset)
                all_stm[kept] = float(stm)
                all_result[kept] = game_result
                all_eval[kept] = search_eval
                kept += 1

                if not quiet and (i + 1) % 500000 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (scan_count - i - 1) / rate if rate > 0 else 0
                    print(f"\r  Parsed {kept:,}/{num_to_load:,} ({rate:,.0f} pos/s, ETA {eta:.0f}s)   ", end='', flush=True)
        except KeyboardInterrupt:
            mm.close()
            print("\nInterrupted during data loading - exiting cleanly.")
            sys.exit(0)
        mm.close()

    elapsed = time.time() - t0
    rate = kept / elapsed if elapsed > 0 else 0
    if not quiet:
        print(f"\r  Parsed {kept:,} positions in {elapsed:.1f}s ({rate:,.0f} pos/s)          ")

    # Trim to actual count
    if kept < num_to_load:
        if not quiet:
            print(f"  Filtered: {num_to_load - kept:,} extreme-eval positions removed, {kept:,} kept")
        feat_padded = feat_padded[:kept]
        all_stm    = all_stm[:kept]
        all_result = all_result[:kept]
        all_eval   = all_eval[:kept]
        num_to_load = kept

    # Phase 2: Chunked vectorized HalfKAv2 conversion
    # Make contiguous copies of the kept slices so we can free the over-sized originals
    _fp = feat_padded[:kept].copy()
    _stm = all_stm[:kept].copy()
    _res = all_result[:kept].copy()
    _ev = all_eval[:kept].copy()
    del feat_padded, all_stm, all_result, all_eval
    import gc; gc.collect()

    return _vectorized_halfkav2(_fp, _stm, _res, _ev,
                                 filter_eval_max=0.0, quiet=quiet)

def _load_dataset_worker(path, max_positions, filter_eval_max):
    """Worker function for parallel dataset loading."""
    return load_training_data(path, max_positions=max_positions,
                              filter_eval_max=filter_eval_max, quiet=True)

# =============================================================================
# Weight I/O - matches C++ binary format exactly
# FIXED: Uses architecture constants instead of hardcoded values
# =============================================================================
def save_weights_cpp(net, filename):
    """Save weights in v5 binary format matching C++ NNUE::Network::loadWeights.
    Format: magic(4B) + version(4B) + L1(4B) + L2(4B) + L3(4B) + WDL(4B) + layer data.
    Writes atomically via .tmp to protect against mid-write crashes."""
    import shutil, struct
    NNUE_MAGIC   = 0x4E4E5545  # "NNUE" in ASCII (little-endian)
    NNUE_VERSION = 5
    tmp_path = filename + '.tmp'
    with open(tmp_path, 'wb') as f:
        # V5 header: magic + version + architecture dimensions
        f.write(struct.pack('<II', NNUE_MAGIC, NNUE_VERSION))
        f.write(struct.pack('<IIII', L1_SIZE, L2_SIZE, L3_SIZE, WDL_SIZE))
        # L1: [NUM_FEATURES, L1_SIZE] floats
        f.write(net.l1_weight.detach().cpu().numpy().astype(np.float32).tobytes())
        f.write(net.l1_bias.detach().cpu().numpy().astype(np.float32).tobytes())
        # L2: C++ stores [L1_SIZE*2][L2_SIZE]; PyTorch weight=[L2_SIZE, L1_SIZE*2] -> transpose
        f.write(net.l2.weight.detach().cpu().numpy().T.astype(np.float32).tobytes())
        f.write(net.l2.bias.detach().cpu().numpy().astype(np.float32).tobytes())
        # L3: C++ stores [L2_SIZE][L3_SIZE]; PyTorch weight=[L3_SIZE, L2_SIZE] -> transpose
        f.write(net.l3.weight.detach().cpu().numpy().T.astype(np.float32).tobytes())
        f.write(net.l3.bias.detach().cpu().numpy().astype(np.float32).tobytes())
        # Phase heads (opening, middlegame, endgame)
        # C++ PhaseHead: weights[WDL_SIZE][L3_SIZE] + biases[WDL_SIZE]
        # PyTorch Linear(L3_SIZE, WDL_SIZE): weight=[WDL_SIZE, L3_SIZE] — no transpose needed
        for head in [net.head_opening, net.head_middlegame, net.head_endgame]:
            f.write(head.weight.detach().cpu().numpy().astype(np.float32).tobytes())
            f.write(head.bias.detach().cpu().numpy().astype(np.float32).tobytes())
        f.flush()
        os.fsync(f.fileno())
    shutil.move(tmp_path, filename)
    print(f"Weights saved to {filename}")

def safe_torch_save(data, path):
    """Save torch checkpoint atomically.
    INFO [7.22]: Consider adding arch constants to checkpoint for resume validation.
    """
    import shutil
    tmp_path = path + '.tmp'
    torch.save(data, tmp_path)
    shutil.move(tmp_path, path)
    print(f"Checkpoint saved to {path}")

def load_weights_cpp(net, filename):
    """Load weights from C++ v4/v5 binary format with automatic architecture migration.
    V4: fixed dims (L1=1024, L2=128, L3=64, WDL=3), no dim header.
    V5: dims stored in header, supports any architecture.
    If file dims differ from current constants, migrates adaptively:
      - Existing neurons: weights copied exactly (preserves learned behavior)
      - New neurons: incoming weights = He-init (breaks symmetry for learning),
        outgoing weights = zero (no immediate effect on network output)
      - Removed neurons: truncated (optimizer adapts quickly)"""
    import struct
    NNUE_MAGIC = 0x4E4E5545

    with open(filename, 'rb') as f:
        # --- Read and verify header ---
        header = f.read(8)
        if len(header) < 8:
            raise ValueError("Weight file too small for header")
        magic, version = struct.unpack_from('<II', header)
        if magic != NNUE_MAGIC:
            raise ValueError(f"Bad magic: 0x{magic:08X} (expected 0x{NNUE_MAGIC:08X})")
        if version in (2, 3):
            raise ValueError(f"Version {version} is incompatible (L1=512 era)")

        # --- Determine file architecture dimensions ---
        if version == 5:
            dims = f.read(16)
            if len(dims) < 16:
                raise ValueError("Weight file truncated in v5 dimension header")
            fl1, fl2, fl3, fwdl = struct.unpack('<IIII', dims)
        elif version == 4:
            fl1, fl2, fl3, fwdl = 1024, 128, 64, 3
        else:
            raise ValueError(f"Unknown weight file version {version}")

        migrating = (fl1 != L1_SIZE or fl2 != L2_SIZE or
                     fl3 != L3_SIZE or fwdl != WDL_SIZE)

        if migrating:
            print(f"NNUE: Migrating weights from "
                  f"{NUM_FEATURES}x{fl1}x{fl2}x{fl3}x{fwdl} -> "
                  f"{NUM_FEATURES}x{L1_SIZE}x{L2_SIZE}x{L3_SIZE}x{WDL_SIZE}")

            rng = np.random.RandomState(12345)  # deterministic for reproducibility

            def adaptive_2d(file_rows, file_cols, tgt_rows, tgt_cols):
                """Read [file_rows x file_cols] from file, return [tgt_rows x tgt_cols]."""
                data = np.frombuffer(f.read(file_rows * file_cols * 4),
                                     dtype=np.float32).reshape(file_rows, file_cols)
                result = np.zeros((tgt_rows, tgt_cols), dtype=np.float32)
                cr, cc = min(file_rows, tgt_rows), min(file_cols, tgt_cols)
                result[:cr, :cc] = data[:cr, :cc]
                return result, cr

            def adaptive_1d(file_size, tgt_size):
                """Read [file_size] bias from file, return [tgt_size]."""
                data = np.frombuffer(f.read(file_size * 4), dtype=np.float32)
                result = np.zeros(tgt_size, dtype=np.float32)
                cs = min(file_size, tgt_size)
                result[:cs] = data[:cs]
                return result

            def he_init_cols(arr, col_start, col_end, row_end, fan_in, fan_out):
                """He-initialize new columns (incoming weights to new neurons)."""
                stddev = np.sqrt(2.0 / (fan_in + fan_out))
                arr[:row_end, col_start:col_end] = rng.normal(
                    0, stddev, size=(row_end, col_end - col_start)).astype(np.float32)

            # --- L1: [NUM_FEATURES x fl1] -> [NUM_FEATURES x L1_SIZE] ---
            l1w, _ = adaptive_2d(NUM_FEATURES, fl1, NUM_FEATURES, L1_SIZE)
            l1b = adaptive_1d(fl1, L1_SIZE)
            if L1_SIZE > fl1:
                he_init_cols(l1w, fl1, L1_SIZE, NUM_FEATURES, NUM_FEATURES, L1_SIZE)
                print(f"  L1: expanded {fl1} -> {L1_SIZE} neurons (He-init)")
            elif L1_SIZE < fl1:
                print(f"  L1: truncated {fl1} -> {L1_SIZE} neurons")
            net.l1_weight.data = torch.from_numpy(l1w)
            net.l1_bias.data = torch.from_numpy(l1b)

            # --- L2: [fl1*2 x fl2] -> [L1_SIZE*2 x L2_SIZE] (C++ layout) ---
            l2w_cpp, _ = adaptive_2d(fl1 * 2, fl2, L1_SIZE * 2, L2_SIZE)
            l2b = adaptive_1d(fl2, L2_SIZE)
            if L2_SIZE > fl2:
                old_rows = min(fl1 * 2, L1_SIZE * 2)
                he_init_cols(l2w_cpp, fl2, L2_SIZE, old_rows, L1_SIZE * 2, L2_SIZE)
                print(f"  L2: expanded {fl2} -> {L2_SIZE} neurons (He-init)")
            elif L2_SIZE < fl2:
                print(f"  L2: truncated {fl2} -> {L2_SIZE} neurons")
            net.l2.weight.data = torch.from_numpy(l2w_cpp.T.copy())  # transpose for PyTorch
            net.l2.bias.data = torch.from_numpy(l2b)

            # --- L3: [fl2 x fl3] -> [L2_SIZE x L3_SIZE] (C++ layout) ---
            l3w_cpp, _ = adaptive_2d(fl2, fl3, L2_SIZE, L3_SIZE)
            l3b = adaptive_1d(fl3, L3_SIZE)
            if L3_SIZE > fl3:
                old_rows = min(fl2, L2_SIZE)
                he_init_cols(l3w_cpp, fl3, L3_SIZE, old_rows, L2_SIZE, L3_SIZE)
                print(f"  L3: expanded {fl3} -> {L3_SIZE} neurons (He-init)")
            elif L3_SIZE < fl3:
                print(f"  L3: truncated {fl3} -> {L3_SIZE} neurons")
            net.l3.weight.data = torch.from_numpy(l3w_cpp.T.copy())  # transpose for PyTorch
            net.l3.bias.data = torch.from_numpy(l3b)

            # --- Phase heads: [fwdl x fl3] -> [WDL_SIZE x L3_SIZE] ---
            # Outgoing weights from new L3 neurons stay zero (correct)
            for head in [net.head_opening, net.head_middlegame, net.head_endgame]:
                hw, _ = adaptive_2d(fwdl, fl3, WDL_SIZE, L3_SIZE)
                hb = adaptive_1d(fwdl, WDL_SIZE)
                head.weight.data = torch.from_numpy(hw.copy())
                head.bias.data = torch.from_numpy(hb)

            print("NNUE: Migration complete. Existing neurons preserved, "
                  "new neurons initialized with He weights.")

        else:
            # --- Fast path: dimensions match exactly ---
            l1w = np.frombuffer(f.read(NUM_FEATURES * L1_SIZE * 4),
                                dtype=np.float32).reshape(NUM_FEATURES, L1_SIZE)
            net.l1_weight.data = torch.from_numpy(l1w.copy())
            l1b = np.frombuffer(f.read(L1_SIZE * 4), dtype=np.float32).copy()
            net.l1_bias.data = torch.from_numpy(l1b)

            l2w = np.frombuffer(f.read(L1_SIZE * 2 * L2_SIZE * 4),
                                dtype=np.float32).reshape(L1_SIZE * 2, L2_SIZE)
            net.l2.weight.data = torch.from_numpy(l2w.T.copy())
            l2b = np.frombuffer(f.read(L2_SIZE * 4), dtype=np.float32).copy()
            net.l2.bias.data = torch.from_numpy(l2b)

            l3w = np.frombuffer(f.read(L2_SIZE * L3_SIZE * 4),
                                dtype=np.float32).reshape(L2_SIZE, L3_SIZE)
            net.l3.weight.data = torch.from_numpy(l3w.T.copy())
            l3b = np.frombuffer(f.read(L3_SIZE * 4), dtype=np.float32).copy()
            net.l3.bias.data = torch.from_numpy(l3b)

            for head in [net.head_opening, net.head_middlegame, net.head_endgame]:
                hw = np.frombuffer(f.read(WDL_SIZE * L3_SIZE * 4),
                                   dtype=np.float32).reshape(WDL_SIZE, L3_SIZE)
                head.weight.data = torch.from_numpy(hw.copy())
                hb = np.frombuffer(f.read(WDL_SIZE * 4), dtype=np.float32).copy()
                head.bias.data = torch.from_numpy(hb)

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
        self.l1_weight       = nn.Parameter(torch.zeros(NUM_FEATURES, L1_SIZE))
        self.l1_bias         = nn.Parameter(torch.zeros(L1_SIZE))
        self.l2              = nn.Linear(L1_SIZE * 2, L2_SIZE)
        self.l3              = nn.Linear(L2_SIZE, L3_SIZE)
        # Three phase heads matching C++ PhaseHead (win/draw/loss per phase)
        self.head_opening    = nn.Linear(L3_SIZE, WDL_SIZE)
        self.head_middlegame = nn.Linear(L3_SIZE, WDL_SIZE)
        self.head_endgame    = nn.Linear(L3_SIZE, WDL_SIZE)
        self.screlu    = SCReLU()
        self.drop2     = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.drop3     = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        import math
        avg_active = 30.0
        nn.init.normal_(self.l1_weight, 0, 1.0 / avg_active)
        nn.init.zeros_(self.l1_bias)
        # T-M3: Use empirical SCReLU gain instead of LeakyReLU.
        # For x ~ N(0,1): Var(clamp(x,0,1)^2) ≈ 0.0521  =>  gain ≈ 4.38
        _screlu_gain = 4.38
        nn.init.normal_(self.l2.weight, 0, _screlu_gain / math.sqrt(self.l2.weight.shape[1]))
        nn.init.zeros_(self.l2.bias)
        nn.init.normal_(self.l3.weight, 0, _screlu_gain / math.sqrt(self.l3.weight.shape[1]))
        nn.init.zeros_(self.l3.bias)
        for head in [self.head_opening, self.head_middlegame, self.head_endgame]:
            nn.init.normal_(head.weight, 0, _screlu_gain / math.sqrt(head.weight.shape[1]))
            nn.init.zeros_(head.bias)

    def forward(self, white_features, black_features, stm, phases=None, return_wdl=False):
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

        # Compute WDL logits from all three phase heads
        wdl_o = self.head_opening(l3_out)     # [batch, 3]
        wdl_m = self.head_middlegame(l3_out)  # [batch, 3]
        wdl_e = self.head_endgame(l3_out)     # [batch, 3]

        if phases is not None:
            # Select per-position head based on game phase (0=opening, 1=mid, 2=end)
            if isinstance(phases, np.ndarray):
                phase_t = torch.from_numpy(phases.astype(np.int64))
            else:
                phase_t = phases.long()
            pm = phase_t.unsqueeze(1)  # [batch, 1]
            wdl = torch.where(pm == 0, wdl_o,
                  torch.where(pm == 1, wdl_m, wdl_e))
        else:
            wdl = (wdl_o + wdl_m + wdl_e) / 3.0

        # Scalar eval proxy: win_logit - loss_logit (maps to centipawns via eval_scale)
        scalar_eval = (wdl[:, 0] - wdl[:, 2]).unsqueeze(1)  # [batch, 1]
        if return_wdl:
            return scalar_eval, wdl  # wdl: [batch, 3] raw logits
        return scalar_eval

# =============================================================================
# Stochastic Weight Averaging (manual, compatible with any PyTorch version)
# =============================================================================
class ManualSWA:
    """Accumulates weight snapshots and averages them for better generalization."""
    def __init__(self):
        self.sum_state = None
        self.count = 0

    def update(self, model):
        """Add current model weights to the running sum (float64 to avoid precision loss)."""
        state = model.state_dict()
        if self.sum_state is None:
            # 7.20: Use float64 (double) for accumulation to prevent precision loss
            # when summing many weight snapshots (float32 loses ~7 decimal digits).
            self.sum_state = {k: v.clone().double() for k, v in state.items()}
        else:
            for k in self.sum_state:
                self.sum_state[k] += state[k].double()
        self.count += 1

    def apply(self, model):
        """Apply averaged weights to the model."""
        if self.sum_state is not None and self.count > 0:
            avg = {k: (v / self.count) for k, v in self.sum_state.items()}
            model.load_state_dict  # INFO [7.22]: If arch changed since checkpoint, this will fail with tensor size mismatch(avg)
            print(f"  SWA: Applied averaged weights ({self.count} snapshots)")
            return True
        return False

# =============================================================================
# WDL Label Generation (Step 3.3)
# =============================================================================
def generate_wdl_targets(game_result, search_eval, eval_scale, wdl_lambda,
                         wdl_draw_elo=100.0):
    """Convert game result + eval into WDL ground truth [P(win), P(draw), P(loss)].

    Uses a draw-band sigmoid model for the eval-based component so that the
    draw logit receives a direct training signal.

    Args:
        game_result: [batch] tensor, 1.0=white win, 0.5=draw, 0.0=black win
        search_eval: [batch] tensor, centipawns from white perspective
        eval_scale:  float, sigmoid temperature (same as existing --eval-scale)
        wdl_lambda:  float in [0,1], blend of eval-based WDL (1.0) vs result-based WDL (0.0)
        wdl_draw_elo: float, draw bandwidth in centipawns (controls how wide
                      the draw band is in the eval-based WDL estimate)

    Returns:
        [batch, 3] tensor of [P(win), P(draw), P(loss)] targets, sums to 1.0
    """
    # --- Result-based WDL (hard labels from game outcome) ---
    # result=1.0 → [1,0,0], result=0.5 → [0,1,0], result=0.0 → [0,0,1]
    result_win  = torch.clamp(game_result * 2.0 - 1.0, 0.0, 1.0)  # 1→1, 0.5→0, 0→0
    result_loss = torch.clamp(1.0 - game_result * 2.0, 0.0, 1.0)  # 1→0, 0.5→0, 0→1
    result_draw = 1.0 - result_win - result_loss                    # 1→0, 0.5→1, 0→0

    # --- Eval-based WDL (soft labels from engine evaluation) ---
    # Draw-band sigmoid: P(win) = σ((eval - d) / s), P(loss) = σ((-eval - d) / s)
    # P(draw) = 1 - P(win) - P(loss) ≥ 0 when d ≥ 0
    eval_win  = torch.sigmoid((search_eval - wdl_draw_elo) / eval_scale)
    eval_loss = torch.sigmoid((-search_eval - wdl_draw_elo) / eval_scale)
    eval_draw = 1.0 - eval_win - eval_loss
    # Clamp to avoid tiny negatives from floating point
    eval_draw = torch.clamp(eval_draw, min=0.0)

    # --- Blend eval-based and result-based ---
    p_win  = wdl_lambda * eval_win  + (1.0 - wdl_lambda) * result_win
    p_draw = wdl_lambda * eval_draw + (1.0 - wdl_lambda) * result_draw
    p_loss = wdl_lambda * eval_loss + (1.0 - wdl_lambda) * result_loss

    return torch.stack([p_win, p_draw, p_loss], dim=1)  # [batch, 3]


# =============================================================================
# Loss Function (with label smoothing support + WDL cross-entropy)
# =============================================================================
def compute_loss(net, white, black, stm, game_result, search_eval,
                 lam, eval_scale, label_smoothing=0.0, phases=None,
                 wdl_alpha=0.0, wdl_lambda=None, wdl_draw_elo=100.0):
    """Compute training loss, optionally blending scalar MSE with WDL cross-entropy.

    Args:
        wdl_alpha:    float in [0,1], weight of WDL CE loss (0.0 = MSE only, 1.0 = CE only)
        wdl_lambda:   float in [0,1], eval vs result blend for WDL targets (None = use lam)
        wdl_draw_elo: float, draw bandwidth in centipawns for eval-based WDL
    """
    use_wdl = (wdl_alpha > 0.0)

    if use_wdl:
        raw_output, wdl_logits = net(white, black, stm, phases=phases, return_wdl=True)
    else:
        raw_output = net(white, black, stm, phases=phases)

    predicted      = raw_output.squeeze(1) * eval_scale
    predicted_white = torch.where(stm < 0.5, predicted, -predicted)
    sig_pred       = torch.sigmoid(predicted_white / eval_scale)
    sig_target     = torch.sigmoid(search_eval / eval_scale)

    # Label smoothing: soften game results toward 0.5
    smoothed_result = game_result
    if label_smoothing > 0:
        smoothed_result = game_result * (1.0 - label_smoothing) + 0.5 * label_smoothing

    # --- Scalar MSE loss (existing approach) ---
    eval_loss   = (sig_pred - sig_target) ** 2
    result_loss = (sig_pred - smoothed_result) ** 2
    mse_loss    = lam * eval_loss + (1.0 - lam) * result_loss

    if not use_wdl:
        return mse_loss

    # --- WDL cross-entropy loss (new: direct training signal for all 3 logits) ---
    effective_wdl_lambda = wdl_lambda if wdl_lambda is not None else lam
    wdl_targets = generate_wdl_targets(game_result, search_eval, eval_scale,
                                        effective_wdl_lambda, wdl_draw_elo)
    # AUDIT FIX H1: Flip WDL win/loss channels for Black-STM positions
    # wdl_targets are White-POV but network WDL head is STM-relative
    stm_is_black = stm > 0.5  # [batch] bool tensor
    wdl_targets_aligned = wdl_targets.clone()
    wdl_targets_aligned[stm_is_black, 0] = wdl_targets[stm_is_black, 2]
    wdl_targets_aligned[stm_is_black, 2] = wdl_targets[stm_is_black, 0]
    wdl_targets = wdl_targets_aligned
    # Apply label smoothing to WDL targets: soften toward uniform [1/3, 1/3, 1/3]
    if label_smoothing > 0:
        wdl_targets = wdl_targets * (1.0 - label_smoothing) + (label_smoothing / 3.0)

    # Cross-entropy: -Σ y_i * log(softmax(logits)_i)
    log_probs = torch.nn.functional.log_softmax(wdl_logits, dim=1)  # [batch, 3]
    ce_loss   = -(wdl_targets * log_probs).sum(dim=1)               # [batch]

    # --- Blend MSE and CE ---
    return (1.0 - wdl_alpha) * mse_loss + wdl_alpha * ce_loss

# =============================================================================
# Training Log & Progress Graph
# =============================================================================
LOG_FILE  = 'training_log.csv'
PLOT_DIR  = os.path.join('.', 'training progress')
PLOT_FILE = os.path.join(PLOT_DIR, 'training_progress.png')

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
    skipped = 0
    with open(LOG_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Skip rows with missing required fields (corrupted CSV lines)
            if 'loss' not in row or row['loss'] is None:
                skipped += 1
                continue
            # Quick sanity check: loss value must be convertible to float
            try:
                float(row['loss'])
            except (ValueError, TypeError):
                print(f"WARNING: Skipping corrupted row {i+2} in {LOG_FILE}: "
                      f"loss='{row.get('loss')}' is not a valid number")
                skipped += 1
                continue
            rows.append(row)
    if skipped:
        print(f"WARNING: Skipped {skipped} corrupted row(s) in {LOG_FILE}")
    return rows

def _safe_float(value, default=0.0):
    """Convert a string to float, returning *default* on malformed values
    (e.g. '922577.23.00' with two decimal points from a corrupted CSV row)."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def generate_plot(log_data=None, show=False):
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed - skipping plot.")
        return
    os.makedirs(PLOT_DIR, exist_ok=True)
    if log_data is None:
        log_data = read_log()
    if not log_data:
        print("No training log data to plot.")
        return

    epochs    = list(range(1, len(log_data) + 1))
    losses    = [_safe_float(row['loss']) for row in log_data]
    lrs       = [_safe_float(row['lr']) for row in log_data]
    val_losses     = [_safe_float(row.get('val_loss', 0)) for row in log_data]
    has_val        = any(v > 0 for v in val_losses)
    epoch_times    = [_safe_float(row.get('epoch_time_s', 0)) for row in log_data]
    has_times      = any(t > 0 for t in epoch_times)
    opening_losses    = [_safe_float(row.get('opening_loss', 0)) for row in log_data]
    middlegame_losses = [_safe_float(row.get('middlegame_loss', 0)) for row in log_data]
    endgame_losses    = [_safe_float(row.get('endgame_loss', 0)) for row in log_data]
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
    ax1.annotate(f'Best train: {losses[best_idx]:.6f}\n(epoch {epochs[best_idx]})',
                 xy=(epochs[best_idx], losses[best_idx]),
                 xytext=(30, 30), textcoords='offset points',
                 fontsize=9, color='#2196F3',
                 arrowprops=dict(arrowstyle='->', color='#2196F3', lw=1))

    # Best val loss annotation — this is the actually saved model
    if has_val:
        valid_val = [(i, v) for i, v in enumerate(val_losses) if v > 0]
        if valid_val:
            best_val_idx = min(valid_val, key=lambda x: x[1])[0]
            ax1.annotate(f'Best val: {val_losses[best_val_idx]:.6f}\n(epoch {epochs[best_val_idx]}) \u2605saved',
                         xy=(epochs[best_val_idx], val_losses[best_val_idx]),
                         xytext=(30, -30), textcoords='offset points',
                         fontsize=9, color='#E91E63', fontweight='bold',
                         arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1.5))

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
            # AUDIT FIX SAFE-4: Use os.startfile instead of shell=True subprocess
            os.startfile(PLOT_FILE)
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
    _primary_is_sharded = False

    # Primary dataset - detect shards directory
    if os.path.isdir(primary_path):
        shards_cfg_file = os.path.join(primary_path, 'shards.cfg')
        if not os.path.exists(shards_cfg_file):
            print(f"ERROR: '{primary_path}' is a directory but contains no shards.cfg")
            sys.exit(1)
        shard_info = load_shards_cfg(primary_path)
        _primary_is_sharded = True
        
        # Shuffle shard order for randomness across generations
        shards = list(shard_info['shards'])
        random.shuffle(shards)
        
        # Select shards to cover the position budget
        if max_positions > 0:
            selected = []
            accumulated = 0
            for spath, scount in shards:
                selected.append((spath, scount))
                accumulated += scount
                if accumulated >= max_positions:
                    break
            if accumulated < max_positions:
                print(f"  Note: all {len(shards)} shards ({accumulated:,} positions) "
                      f"selected (budget: {max_positions:,})")
        else:
            selected = shards
        
        total_selected = sum(c for _, c in selected)
        print(f"  Sharded data: selected {len(selected)}/{len(shards)} shards "
              f"({total_selected:,} positions)")
        
        for spath, scount in selected:
            datasets.append({
                'path': spath,
                'label': os.path.basename(spath),
                'ratio': None,  # filled in below (proportional within primary share)
                'num_positions': scount,
                '_primary_shard': True,
            })
    else:
        # AUDIT FIX BUG-3: Use _read_binary_header() to correctly handle both
        # versioned (magic 'NNUE' + version + count) and legacy (4-byte count) formats.
        # Previously read raw 4 bytes, which for versioned files returns the magic
        # number 0x454E4E55 ≈ 1.16 billion as the position count.
        with open(primary_path, 'rb') as f:
            primary_count, _ = _read_binary_header(f)
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
            # AUDIT FIX BUG-3: Use _read_binary_header() for versioned format support
            with open(filepath, 'rb') as f:
                count, _ = _read_binary_header(f)
            extra_total_ratio += ratio
            datasets.append({
                'path': filepath,
                'label': os.path.basename(filepath),
                'ratio': ratio,
                'num_positions': count,
            })

    if extra_total_ratio >= 1.0:
        print(f"WARNING: Extra dataset ratios sum to {extra_total_ratio:.2f} (>=1.0). Normalising.")
        for ds in datasets:
            if ds.get('ratio') is not None:
                ds['ratio'] = ds['ratio'] / (extra_total_ratio + 0.01)
        extra_total_ratio = sum(ds['ratio'] for ds in datasets if ds.get('ratio') is not None)

    # Primary share(s) get the remainder
    primary_ratio = 1.0 - extra_total_ratio
    primary_datasets = [ds for ds in datasets if ds['ratio'] is None]
    if len(primary_datasets) == 1:
        primary_datasets[0]['ratio'] = primary_ratio
    else:
        # Multiple primary shards: split proportionally by position count
        total_primary_pos = sum(ds['num_positions'] for ds in primary_datasets)
        for ds in primary_datasets:
            ds['ratio'] = primary_ratio * (ds['num_positions'] / total_primary_pos) if total_primary_pos > 0 else primary_ratio / len(primary_datasets)

    # Apply max_positions budget across ALL sources proportionally by ratio.
    # This prevents massive draw datasets (e.g. 35M draws vs 10K decisive)
    # from creating an imbalance that hurts training.
    total_available = sum(ds['num_positions'] for ds in datasets)
    if max_positions > 0 and max_positions < total_available:
        print(f"  --max-positions: distributing {max_positions:,} budget across {len(datasets)} sources by ratio")
        # First pass: allocate budget proportionally, cap at actual count
        budget_remaining = max_positions
        uncapped = []
        for ds in datasets:
            target = int(max_positions * ds['ratio'])
            if target >= ds['num_positions']:
                # F2.5: Source has fewer positions than its budget — use all, mark capped
                budget_remaining -= ds['num_positions']
                ds['_capped'] = True
            else:
                uncapped.append(ds)
                ds['_capped'] = False
        # Second pass: distribute remaining budget to uncapped sources
        # BUGFIX: Use max(1, ...) to prevent rounding to 0, which the loader
        # misinterprets as "no limit" and loads the entire file (OOM).
        uncapped_ratio_sum = sum(ds['ratio'] for ds in uncapped)
        for ds in uncapped:
            if uncapped_ratio_sum > 0:
                target = max(1, int(budget_remaining * (ds['ratio'] / uncapped_ratio_sum)))
            else:
                target = max(1, budget_remaining // len(uncapped))
            old_count = ds['num_positions']
            ds['num_positions'] = min(target, old_count)
            print(f"    {ds['label']}: {old_count:,} -> {ds['num_positions']:,} (ratio {ds['ratio']:.1%})")
        # F2.5: Post-hoc trim if rounding caused total to exceed budget
        total_allocated = sum(ds['num_positions'] for ds in datasets)
        if total_allocated > max_positions:
            excess = total_allocated - max_positions
            # Trim from largest uncapped source
            largest = max(uncapped, key=lambda d: d['num_positions'])
            largest['num_positions'] = max(1, largest['num_positions'] - excess)
            print(f"    (trimmed {largest['label']} by {excess} to stay within budget)")
        total_available = sum(ds['num_positions'] for ds in datasets)

    # Print summary
    print(f"\nDatasets:{len(datasets):>14} sources (ratio-based sampling)")
    for ds in datasets:
        print(f"  - {ds['label']:<40s} ratio={ds['ratio']:.1%}  pos={ds['num_positions']:,}")
    print(f"  Total positions: {total_available:,}\n")

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
                  eval_soft_cap=0.0, draw_weight=1.0, mate_boost=0.0, phases=None,
                  wdl_alpha=0.0, wdl_lambda=None, wdl_draw_elo=100.0):
    """Process one mini-batch with gradient accumulation and sample rebalancing."""
    loss_tensor = compute_loss(net, white, black, stm, result, eval_t,
                               lam, eval_scale, label_smoothing, phases=phases,
                               wdl_alpha=wdl_alpha, wdl_lambda=wdl_lambda,
                               wdl_draw_elo=wdl_draw_elo)

    # Apply per-sample rebalancing weights
    sample_weights = compute_sample_weights(eval_t, result, eval_soft_cap, draw_weight, mate_boost)
    if sample_weights is not None:
        loss_tensor = loss_tensor * sample_weights

    loss = loss_tensor.mean()
    scaled_loss = loss / accum_steps
    scaled_loss.backward()
    return loss.item()

# =============================================================================
# Stratified Sampler (Phase 1.4)
# =============================================================================
class StratifiedSampler:
    """Samples training positions proportionally from categorized pools.
    
    Pools: decisive-opening, decisive-middlegame, decisive-endgame,
           drawn-opening, drawn-middlegame, drawn-endgame
    """
    
    def __init__(self, phases, results, 
                 draw_ratio=0.15, 
                 phase_ratios=None,
                 draw_oversample_epochs=0,
                 draw_oversample_factor=3.0):
        """
        Args:
            phases: numpy array of phase labels (0=opening, 1=mid, 2=end)
            results: numpy array of game results
            draw_ratio: target fraction of draws in each batch (0.0-1.0)
            phase_ratios: dict {0: r_op, 1: r_mg, 2: r_eg} target ratios (sum to 1.0)
            draw_oversample_epochs: for first N epochs, oversample draws
            draw_oversample_factor: multiplier for draw sampling in early epochs
        """
        self.draw_ratio = draw_ratio
        self.phase_ratios = phase_ratios or {0: 0.33, 1: 0.34, 2: 0.33}
        self.draw_oversample_epochs = draw_oversample_epochs
        self.draw_oversample_factor = draw_oversample_factor
        
        # Normalize phase ratios
        total = sum(self.phase_ratios.values())
        self.phase_ratios = {k: v / total for k, v in self.phase_ratios.items()}
        
        # Build pools: (is_draw, phase) -> list of indices
        is_draw = np.abs(results - 0.5) < 0.01
        self.pools = {}
        for draw_flag in [False, True]:
            for phase in [0, 1, 2]:
                mask = (is_draw == draw_flag) & (phases == phase)
                indices = np.where(mask)[0]
                self.pools[(draw_flag, phase)] = indices
        
        # Report pool sizes
        for key, indices in self.pools.items():
            draw_str = "drawn" if key[0] else "decisive"
            phase_str = {0: "opening", 1: "middlegame", 2: "endgame"}[key[1]]
            if len(indices) > 0:
                pass  # silent
    
    def sample_epoch_indices(self, total_samples, epoch=0):
        """Sample indices for one epoch with stratified proportions.
        
        Args:
            total_samples: total number of samples to return
            epoch: current epoch (for draw oversampling schedule)
        
        Returns:
            numpy array of shuffled indices
        """
        # Determine effective draw ratio
        effective_draw_ratio = self.draw_ratio
        if epoch < self.draw_oversample_epochs:
            effective_draw_ratio = min(0.5, self.draw_ratio * self.draw_oversample_factor)
        
        decisive_ratio = 1.0 - effective_draw_ratio
        
        all_indices = []
        
        for phase in [0, 1, 2]:
            phase_r = self.phase_ratios[phase]
            
            # Decisive positions for this phase
            n_decisive = int(total_samples * decisive_ratio * phase_r)
            pool = self.pools[(False, phase)]
            if len(pool) > 0 and n_decisive > 0:
                chosen = np.random.choice(pool, size=n_decisive, replace=(n_decisive > len(pool)))
                all_indices.append(chosen)
            
            # Drawn positions for this phase
            n_drawn = int(total_samples * effective_draw_ratio * phase_r)
            pool = self.pools[(True, phase)]
            if len(pool) > 0 and n_drawn > 0:
                chosen = np.random.choice(pool, size=n_drawn, replace=(n_drawn > len(pool)))
                all_indices.append(chosen)
        
        if not all_indices:
            # Fallback: return all available indices
            all_available = np.concatenate([v for v in self.pools.values() if len(v) > 0])
            np.random.shuffle(all_available)
            return all_available[:total_samples]
        
        result = np.concatenate(all_indices)
        np.random.shuffle(result)
        
        # Pad or trim to exact total_samples
        if len(result) < total_samples:
            extra = np.random.choice(result, size=total_samples - len(result), replace=True)
            result = np.concatenate([result, extra])
        elif len(result) > total_samples:
            result = result[:total_samples]
        
        return result

def deduplicate_chunk(white_feats, black_feats, stm, result, evals, phases):
    """Remove duplicate positions within a training chunk using hash-based dedup.
    Positions with identical feature sets are collapsed to one instance."""
    n = len(stm)
    if n == 0:
        return white_feats, black_feats, stm, result, evals, phases

    # Hash based on stm + eval + result (fast proxy for position identity)
    seen = set()
    keep = []
    for i in range(n):
        key = (int(stm[i].item() if hasattr(stm[i], 'item') else stm[i]),
               int(evals[i].item() if hasattr(evals[i], 'item') else evals[i]),
               int(result[i].item() if hasattr(result[i], 'item') else result[i]),
               int(phases[i]) if phases is not None and len(phases) > i else 0)
        if key not in seen:
            seen.add(key)
            keep.append(i)

    if len(keep) == n:
        return white_feats, black_feats, stm, result, evals, phases

    keep_idx = np.array(keep)
    deduped_white = white_feats[keep_idx]
    deduped_black = black_feats[keep_idx]
    deduped_stm = stm[keep_idx] if isinstance(stm, (np.ndarray, torch.Tensor)) else stm
    deduped_result = result[keep_idx] if isinstance(result, (np.ndarray, torch.Tensor)) else result
    deduped_evals = evals[keep_idx] if isinstance(evals, (np.ndarray, torch.Tensor)) else evals
    deduped_phases = phases[keep_idx] if phases is not None else None

    removed = n - len(keep)
    if removed > 0:
        print(f"    Dedup: removed {removed} duplicates ({removed*100/n:.1f}%) from chunk")

    return deduped_white, deduped_black, deduped_stm, deduped_result, deduped_evals, deduped_phases


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

    # Force streaming if dense arrays would exceed ~4 GB RAM or file is much larger than cap
    estimated_ram_gb = (num_positions * MAX_ACTIVE_FEATURES * 2 * 2) / (1024**3)  # sparse white+black idx arrays
    # If --max-positions caps us to a small amount that fits in RAM, use preload mode
    # even if the file is huge (mmap-based preload only touches the first N records).
    file_much_larger = False
    if args.max_positions > 0 and os.path.isfile(args.data):
        with open(args.data, 'rb') as f:
            actual_primary, _ = _read_binary_header(f)
        if actual_primary > args.max_positions * 3:  # file is 3x+ larger than cap
            # Only force streaming if the capped amount itself is too large for preload
            file_much_larger = (args.max_positions > MAX_PRELOAD_POSITIONS)
    streaming_mode = num_positions > MAX_PRELOAD_POSITIONS or estimated_ram_gb > 4.0 or file_much_larger
    if streaming_mode:
        print(f"  Mode: STREAMING (dataset too large for RAM preload)")
        print(f"  Chunk size: {args.chunk_size:,} positions per chunk")
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

    # torch.compile() for +20-40% training speed (PyTorch 2.0+)
    #
    # Fix Inductor cache dir: default path C:\Users\... contains \U which
    # Python interprets as a Unicode escape in generated code → SyntaxError.
    # Redirect to a safe path without \U.
    if os.name == 'nt':
        # Use forward slashes so g++ doesn't interpret \t as tab etc.
        # in generated #include directives.
        _safe_cache = 'C:/Temp/torchinductor'
        os.makedirs(_safe_cache, exist_ok=True)
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = _safe_cache
        try:
            import torch._inductor.config as _ind_cfg
            _ind_cfg.cache_dir = _safe_cache
        except Exception:
            pass

    # g++ (MinGW-w64) is on PATH — Inductor finds it automatically.

    # Fix MinGW M_PI: PyTorch headers use M_PI but MinGW in strict C++ mode
    # doesn't define it unless _USE_MATH_DEFINES is set.  Inductor ignores
    # CXXFLAGS and has no reliable cflags config, so we monkey-patch
    # compile_file() in codecache to inject the flag into every g++ command.
    if os.name == 'nt':
        try:
            import torch._inductor.codecache as _cc
            import sysconfig, sys
            _orig_compile_file = _cc.compile_file
            # Locate Python libs directory (contains python3XX.lib or .dll.a)
            _py_libs_dir = os.path.join(sys.prefix, 'libs').replace('\\', '/')
            _py_ver = f'{sys.version_info.major}{sys.version_info.minor}'
            def _patched_compile_file(input_path, output_path, cmd):
                if isinstance(cmd, list) and len(cmd) > 1:
                    # -D flags go early (before source), -L/-l go at END
                    # (GCC linker resolves symbols left-to-right, so libs
                    #  must come after the objects that reference them)
                    cmd = [cmd[0], '-D_USE_MATH_DEFINES'] + cmd[1:] + \
                          [f'-L{_py_libs_dir}', f'-lpython{_py_ver}',
                           '-static-libgcc', '-static-libstdc++']
                    # Output .pyd instead of .so — Windows importlib won't
                    # recognise .so as a native extension module
                    cmd = [c[:-3] + '.pyd' if c.endswith('.so') else c for c in cmd]
                    output_path = output_path[:-3] + '.pyd' if output_path.endswith('.so') else output_path
                return _orig_compile_file(input_path, output_path, cmd)
            _cc.compile_file = _patched_compile_file

            # Register MinGW DLL directory so Python can find runtime libs
            # (Python 3.8+ no longer searches PATH for DLL dependencies)
            _mingw_bin = r'C:\msys64\ucrt64\bin'
            if os.path.isdir(_mingw_bin):
                os.add_dll_directory(_mingw_bin)

            # Patch the module loader so it looks for .pyd instead of .so
            _CppPyBindCache = _cc.CppPythonBindingsCodeCache
            _orig_load_inner = _CppPyBindCache._load_library_inner.__func__
            @classmethod
            def _patched_load_inner(cls, path, key):
                if path.endswith('.so'):
                    pyd_path = path[:-3] + '.pyd'
                    if os.path.exists(pyd_path):
                        path = pyd_path
                return _orig_load_inner(cls, path, key)
            _CppPyBindCache._load_library_inner = _patched_load_inner
        except Exception:
            pass

    # Fix Windows path mangling: PyTorch Inductor uses shlex.split(cmd)
    # which defaults to POSIX mode — this treats backslashes as escape
    # characters, turning C:\Temp into C:Temp. Replace the shlex
    # reference inside codecache so all shlex.split() calls use posix=False.
    if os.name == 'nt':
        try:
            import torch._inductor.codecache as _cc
            import shlex as _shlex, types as _types
            _real_split = _shlex.split
            _fake_shlex = _types.ModuleType('shlex')
            _fake_shlex.split = lambda s, comments=False, posix=True: \
                _real_split(s, comments=comments, posix=False)
            # Copy any other shlex attributes Inductor might use
            for _attr in dir(_shlex):
                if not hasattr(_fake_shlex, _attr):
                    setattr(_fake_shlex, _attr, getattr(_shlex, _attr))
            _cc.shlex = _fake_shlex
        except Exception:
            pass

    if hasattr(torch, 'compile') and not getattr(args, 'no_compile', False):
        try:
            net = torch.compile(net, mode='reduce-overhead')
            print("  torch.compile:     ENABLED (reduce-overhead mode)")
        except Exception as e:
            print(f"  torch.compile:     SKIPPED ({e})")

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
    # F2.4: Account for warmup epochs in cosine schedule so LR reaches lr_min on time.
    # Warmup is step-based; estimate warmup_epochs from steps.
    warmup_epochs_est = 0
    if args.warmup_steps > 0 and num_positions > 0:
        steps_per_epoch = max(1, num_positions // (args.batch_size * args.grad_accum))
        warmup_epochs_est = max(1, args.warmup_steps // steps_per_epoch)

    if args.cosine_lr:
        if args.cosine_restarts:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=args.cosine_t0,
                T_mult=args.cosine_t_mult,
                eta_min=args.lr_min
            )
            schedule_name = f"Cosine Warm Restarts (T0={args.cosine_t0}, Tmult={args.cosine_t_mult})"
        else:
            effective_cosine_epochs = max(1, args.epochs - warmup_epochs_est)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=effective_cosine_epochs,
                eta_min=args.lr_min
            )
            schedule_name = f"Cosine Decay (T_max={effective_cosine_epochs}, warmup_epochs~{warmup_epochs_est})"
    else:
        scheduler = None
        schedule_name = "Constant LR"

    # F2.1: Initialize training state defaults BEFORE checkpoint may override them
    best_loss         = float('inf')
    epochs_no_improve = 0
    global_step       = 0

    # --- Load optimizer checkpoint ---
    if not args.fresh and not args.load_weights and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            optimizer.load_state_dict(ckpt['optimizer'])
            if scheduler is not None and 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
                print(f"Resumed optimizer state (LR: {scheduler.get_last_lr()[0]:.6f})")
            else:
                print(f"Resumed optimizer state (LR: {optimizer.param_groups[0]['lr']:.6f})")
            # AUDIT FIX M5: Restore full training state on resume
            if 'epoch' in ckpt:
                epochs_done = ckpt['epoch']
                print(f"  Resumed from epoch {epochs_done}")
            if 'global_step' in ckpt:
                global_step = ckpt['global_step']
            if 'best_loss' in ckpt:
                best_loss = ckpt['best_loss']
            if 'epochs_no_improve' in ckpt:
                epochs_no_improve = ckpt['epochs_no_improve']
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
    draw_w_str = f"{args.draw_weight:.1f}x" if args.draw_weight != 1.0 else "OFF"  # FIX: show active for any non-1.0 value
    print(f"  Draw weight:       {draw_w_str}")
    mate_b_str = f"{args.mate_boost:.1f}x" if args.mate_boost > 0 else "OFF"
    print(f"  Mate boost:        {mate_b_str}")
    print(f"  Async prefetch:    {'ON' if streaming_mode else 'N/A (preload mode)'}")
    if args.wdl_alpha > 0:
        wdl_lam_str = f"{args.wdl_lambda:.2f}" if args.wdl_lambda is not None else f"{lam} (from --lam)"
        print(f"  WDL CE loss:       ON (alpha={args.wdl_alpha}, lambda={wdl_lam_str}, draw_elo={args.wdl_draw_elo})")
        if args.adaptive_lambda:
            print(f"  Adaptive lambda:   {args.lambda_start:.2f} -> {args.lambda_end:.2f} over {args.epochs} epochs")
    else:
        print(f"  WDL CE loss:       OFF (scalar MSE only)")
    print(f"  Activation:        SCReLU (squared clipped ReLU)")
    print(f"  Architecture:      {NUM_FEATURES}->{L1_SIZE}->{L2_SIZE}->{L3_SIZE}->WDL(3)x3heads")
    if hasattr(args, 'stratified') and args.stratified:
        print(f"  Stratified:        ON (draw_ratio={args.draw_ratio}, "
              f"oversample_epochs={args.draw_oversample_epochs})")
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
            # Pass max_positions to scan_positions for memory-efficient reservoir sampling
            ds_count, ds_offsets = scan_positions(ds['path'], max_positions=ds['num_positions'])

            # 10% val, capped proportionally
            ds_indices = np.arange(len(ds_offsets), dtype=np.int64)  # T-M1: numpy indices
            # AUDIT FIX: Use np.random.shuffle for numpy arrays (rng.shuffle uses
            # slow Python-level element access on numpy arrays)
            np.random.shuffle(ds_indices)
            ds_val_size = max(1, len(ds_indices) // 10)
            ds_val_size = min(ds_val_size, int(50_000 * ds['ratio']))

            ds_val_idx   = ds_indices[:ds_val_size]
            ds_train_idx = ds_indices[ds_val_size:]

            ds_val_offsets   = ds_offsets[ds_val_idx]    # numpy fancy indexing (T-M1)
            ds_train_offsets = ds_offsets[ds_train_idx]  # numpy fancy indexing (T-M1)

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
            part = load_positions_at_offsets(path, sorted(offsets), filter_eval_max=args.filter_eval_max)
            val_parts.append(part)

        # Merge val parts
        if len(val_parts) == 1:
            val_data = val_parts[0]
        else:
            val_data = {
                'white':  _safe_concat_np([p['white']  for p in val_parts], tag='val_w'),
                'black':  _safe_concat_np([p['black']  for p in val_parts], tag='val_b'),
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
        if len(datasets) > 1:
            # M-D5 FIX: Honor dataset ratios in preload mode via weighted index selection.
            print("  Multi-dataset preload: ratios will be applied during epoch sampling.")
        all_data_parts = [None] * len(datasets)
        loaded_counts = [0] * len(datasets)
        if len(datasets) > 1:
            # AUDIT FIX (v7): Load datasets SEQUENTIALLY.  Parallel loading
            # causes disk thrash + OS page-cache pressure from simultaneous
            # memmap writes, which can freeze Windows when pagefile.sys
            # competes for the same physical disk.  Sequential loading keeps
            # peak RAM to one dataset at a time and lets the OS flush memmap
            # pages between loads.
            print(f"  Loading {len(datasets)} datasets sequentially...")
            for idx, ds in enumerate(datasets):
                part = load_training_data(ds['path'], max_positions=ds['num_positions'],
                                          filter_eval_max=args.filter_eval_max, quiet=False)
                all_data_parts[idx] = part
                loaded_counts[idx] = part['count']
                print(f"  [{ds['label']}] loaded: {part['count']:,} positions")
                import gc; gc.collect()          # free transient buffers before next load
        else:
            ds = datasets[0]
            part = load_training_data(ds['path'], max_positions=ds['num_positions'],
                                      filter_eval_max=args.filter_eval_max)
            all_data_parts[0] = part
            loaded_counts[0] = part['count']
            print(f"  [{ds['label']}] loaded: {part['count']:,} positions")

        # Merge all parts
        if len(all_data_parts) == 1:
            data = all_data_parts[0]
        else:
            data = {
                'white':  _safe_concat_np([p['white']  for p in all_data_parts], tag='merge_w'),
                'black':  _safe_concat_np([p['black']  for p in all_data_parts], tag='merge_b'),
                'stm':    torch.cat([p['stm']    for p in all_data_parts]),
                'result': torch.cat([p['result'] for p in all_data_parts]),
                'eval':   torch.cat([p['eval']   for p in all_data_parts]),
                'phases': np.concatenate([p['phases'] for p in all_data_parts]),
                'count':  sum(p['count'] for p in all_data_parts),
            }
        del all_data_parts

        # M-D5: Track per-dataset boundaries for ratio-aware epoch sampling
        # Use post-filter loaded_counts, not pre-filter ds['num_positions']
        ds_boundaries = []
        cum = 0
        for i, ds in enumerate(datasets):
            ds_count = loaded_counts[i]
            ds_boundaries.append((cum, cum + ds_count, ds['ratio']))
            cum += ds_count
        num_positions = data['count']
        all_indices = np.arange(num_positions, dtype=np.int64)
        # AUDIT FIX: Use numpy for large index shuffling (faster than Python list)
        np.random.shuffle(all_indices)
        val_size     = max(1, num_positions // 10)
        val_size     = min(val_size, 50_000)
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
            'phases': data['phases'][val_indices],
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
    val_phases = val_data['phases']

    start_time        = time.time()
    run_id            = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_log_data      = read_log()
    has_phase_data    = any(len(v) > 0 for v in val_phase_indices.values())

    # SWA setup
    swa = ManualSWA() if args.swa else None

    # Global optimizer step counter (for warmup) — F2.1: initialized before checkpoint load
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

    if not streaming_mode and hasattr(args, 'stratified') and args.stratified:
        phase_ratios = {
            0: args.phase_ratio_opening,
            1: args.phase_ratio_middlegame,
            2: args.phase_ratio_endgame,
        }
        stratified_sampler = StratifiedSampler(
            phases=data['phases'][train_indices],
            results=data['result'][train_indices].numpy(),
            draw_ratio=args.draw_ratio,
            phase_ratios=phase_ratios,
            draw_oversample_epochs=args.draw_oversample_epochs,
            draw_oversample_factor=args.draw_oversample_factor,
        )
        # Print pool sizes
        for key, pool_indices in stratified_sampler.pools.items():
            draw_str = "drawn" if key[0] else "decisive"
            phase_str = {0: "opening", 1: "middlegame", 2: "endgame"}[key[1]]
            print(f"    Stratified pool [{draw_str}-{phase_str}]: {len(pool_indices):,}")

    # T-L2: Removed redundant "Mode: eager" print — mode already shown at L1329/1332

    # --- Smart ETA state ---
    ema_epoch_time  = None
    EMA_ALPHA       = 0.15  # Smoothed EMA (was 0.3 with trend; lower = more stable)

    # --- Async prefetch helper for streaming mode ---
    def load_chunk_async(filename, offsets_slice, filt_max):
        sorted_slice = sorted(offsets_slice)
        return load_positions_at_offsets(filename, sorted_slice, filter_eval_max=filt_max)

    epoch = 0
    try:
        for epoch in range(args.epochs):
            epoch_start = time.time()
            net.train()

            # Adaptive lambda scheduling: blend from eval-heavy to result-heavy
            if args.adaptive_lambda:
                progress = epoch / max(args.epochs - 1, 1)
                current_wdl_lambda = args.lambda_start + progress * (args.lambda_end - args.lambda_start)
                print(f"  Adaptive lambda: {current_wdl_lambda:.4f} "
                      f"(epoch {epoch+1}/{args.epochs})")
            else:
                current_wdl_lambda = args.wdl_lambda

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
                    np.random.shuffle(src_offsets)          # T-M1: np.random.shuffle for numpy arrays
                    # Sample ratio * total_train positions from this source
                    n_sample = int(src['ratio'] * total_train)
                    if n_sample >= len(src_offsets):
                        # Oversample with replacement if needed
                        sampled = np.random.choice(src_offsets, size=n_sample, replace=True)  # T-M1
                    else:
                        sampled = src_offsets[:n_sample]
                    for off in sampled:
                        combined_tagged_offsets.append((src['path'], off))

                random.shuffle(combined_tagged_offsets)

                chunk_size = args.chunk_size
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
                        'white':  _safe_concat_np([p['white']  for p in parts], tag='stream_w'),
                        'black':  _safe_concat_np([p['black']  for p in parts], tag='stream_b'),
                        'stm':    torch.cat([p['stm']    for p in parts]),
                        'result': torch.cat([p['result'] for p in parts]),
                        'eval':   torch.cat([p['eval']   for p in parts]),
                        'phases': np.concatenate([p['phases'] for p in parts]),
                        'count':  sum(p['count'] for p in parts),
                    }

                # Guard: if all sources sampled 0 positions, skip this epoch cleanly
                if not chunk_slices:
                    print(f"[WARN] Epoch {epoch+1}: no training positions sampled "
                          f"(total_train={total_train}, sources={len(all_train_sources)}) "
                          f"— skipping epoch", flush=True)
                    continue

                # Start async prefetch of first chunk
                with ThreadPoolExecutor(max_workers=1) as executor:
                    next_future = executor.submit(load_multi_chunk_async,
                                                    chunk_slices[0], args.filter_eval_max)

                    chunk_times = []
                    epoch_batches_done = 0
                    epoch_batches_total = sum(
                        (len(cs) + batch_size - 1) // batch_size for cs in chunk_slices
                    )

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
                        phases_all = chunk_data['phases']
                        chunk_n    = chunk_data['count']

                        if args.dedup:
                            white_all, black_all, stm_all, result_all, eval_all, phases_all = \
                                deduplicate_chunk(white_all, black_all, stm_all, result_all, eval_all, phases_all)
                            chunk_n = len(stm_all)

                        chunk_indices = list(range(chunk_n))
                        random.shuffle(chunk_indices)
                        idx_tensor = torch.tensor(chunk_indices, dtype=torch.long)

                        chunk_batches = (chunk_n + batch_size - 1) // batch_size
                        chunk_batch_i = 0

                        for batch_start in range(0, chunk_n, batch_size):
                            batch_idx = idx_tensor[batch_start:batch_start + batch_size]

                            batch_idx_np = batch_idx.numpy()
                            loss_val = process_batch(
                                net, optimizer,
                                densify_batch(white_all[batch_idx_np]),
                                densify_batch(black_all[batch_idx_np]),
                                stm_all[batch_idx], result_all[batch_idx],
                                eval_all[batch_idx],
                                lam, eval_scale, label_smoothing,
                                accum_steps, args.grad_clip,
                                eval_soft_cap=args.eval_soft_cap,
                                draw_weight=args.draw_weight,
                                mate_boost=args.mate_boost,
                                phases=phases_all[batch_idx_np],
                                wdl_alpha=args.wdl_alpha,
                                wdl_lambda=current_wdl_lambda,
                                wdl_draw_elo=args.wdl_draw_elo,
                            )
                            total_loss  += loss_val
                            num_batches += 1
                            accum_count += 1
                            chunk_batch_i += 1
                            epoch_batches_done += 1

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

                            # Batch progress (every 10 batches)
                            if chunk_batch_i % 10 == 0 or chunk_batch_i == chunk_batches:
                                batch_elapsed = time.time() - chunk_start_time
                                batch_rate = chunk_batch_i / max(batch_elapsed, 0.01)
                                chunk_batch_eta = (chunk_batches - chunk_batch_i) / max(batch_rate, 0.01)
                                epoch_rate = epoch_batches_done / max(time.time() - epoch_start, 0.01)
                                epoch_batch_eta = (epoch_batches_total - epoch_batches_done) / max(epoch_rate, 0.01)
                                running_loss = total_loss / max(num_batches, 1)
                                print(f"\r  Ep {epoch+1} | Ch {chunk_i+1}/{num_chunks} "
                                      f"[{chunk_batch_i}/{chunk_batches}] | "
                                      f"Loss: {running_loss:.6f} | "
                                      f"ChETA: {format_time(chunk_batch_eta)} | "
                                      f"EpETA: {format_time(epoch_batch_eta)}\033[K", end='', flush=True)

                        # Free chunk memory
                        del chunk_data, white_all, black_all, stm_all, result_all, eval_all

                        chunk_elapsed = time.time() - chunk_start_time
                        chunk_times.append(chunk_elapsed)

                    print()  # newline after \r progress

            else:
                # === PRELOAD MODE ===
                if hasattr(args, 'stratified') and args.stratified and 'stratified_sampler' in locals():
                    relative_indices = stratified_sampler.sample_epoch_indices(len(train_indices), epoch=epoch)
                    indices = [train_indices[ri] for ri in relative_indices.tolist()]
                elif multi_dataset and len(ds_boundaries) > 1:
                    # M-D5 FIX: Sample train indices proportionally by dataset ratio
                    indices = []
                    for ds_start, ds_end, ds_ratio in ds_boundaries:
                        ds_train = [ti for ti in train_indices if ds_start <= ti < ds_end]
                        n_sample = max(1, int(len(train_indices) * ds_ratio))
                        if ds_train:
                            indices.extend(random.choices(ds_train, k=n_sample))
                    random.shuffle(indices)
                elif args.phase_balanced:
                    indices = cached_balanced.copy()
                    random.shuffle(indices)
                else:
                    indices = train_indices.copy()
                    random.shuffle(indices)

                idx_tensor  = torch.tensor(indices, dtype=torch.long)
                num_samples = len(indices)
                total_batches_epoch = (num_samples + batch_size - 1) // batch_size

                white_idx_all = data['white']   # numpy int32 (n, MAX_ACTIVE_FEATURES)
                black_idx_all = data['black']   # numpy int32 (n, MAX_ACTIVE_FEATURES)
                stm_all    = data['stm']
                result_all = data['result']
                eval_all   = data['eval']

                batch_count_epoch = 0
                for batch_start in range(0, num_samples, batch_size):
                    batch_idx = idx_tensor[batch_start:batch_start + batch_size]
                    batch_idx_np = batch_idx.numpy()

                    loss_val = process_batch(
                        net, optimizer,
                        densify_batch(white_idx_all[batch_idx_np]),
                        densify_batch(black_idx_all[batch_idx_np]),
                        stm_all[batch_idx], result_all[batch_idx],
                        eval_all[batch_idx],
                        lam, eval_scale, label_smoothing,
                        accum_steps, args.grad_clip,
                        eval_soft_cap=args.eval_soft_cap,
                        draw_weight=args.draw_weight,
                        mate_boost=args.mate_boost,
                        phases=data['phases'][batch_idx_np],
                        wdl_alpha=args.wdl_alpha,
                        wdl_lambda=current_wdl_lambda,
                        wdl_draw_elo=args.wdl_draw_elo,
                    )
                    total_loss  += loss_val
                    num_batches += 1
                    accum_count += 1
                    batch_count_epoch += 1

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

                    # Batch progress (every 10 batches)
                    if batch_count_epoch % 10 == 0 or batch_count_epoch == total_batches_epoch:
                        batch_elapsed = time.time() - epoch_start
                        batch_rate = batch_count_epoch / max(batch_elapsed, 0.01)
                        batch_eta = (total_batches_epoch - batch_count_epoch) / max(batch_rate, 0.01)
                        running_loss = total_loss / max(num_batches, 1)
                        pct = 100.0 * batch_count_epoch / total_batches_epoch
                        print(f"\r  Ep {epoch+1} | Batch {batch_count_epoch}/{total_batches_epoch} "
                              f"({pct:.0f}%) | Loss: {running_loss:.6f} | "
                              f"ETA: {format_time(batch_eta)}\033[K", end='', flush=True)

                print()  # newline after \r progress

            # Flush remaining accumulated gradients
            if accum_count > 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0
                global_step += 1  # T-L1: count the final partial accumulation step

            # F2.4: Step the LR scheduler only after warmup is complete
            if scheduler is not None and (warmup_steps == 0 or global_step > warmup_steps):
                scheduler.step()

            avg_loss   = total_loss / max(num_batches, 1)
            epoch_time = time.time() - epoch_start
            elapsed    = time.time() - start_time

            # --- Validation (batched to avoid OOM; no label smoothing) ---
            net.eval()
            val_n = val_white.shape[0]
            val_batch_size = 8192
            all_val_losses = []
            all_val_pred_probs = []

            with torch.no_grad():
                for vb_start in range(0, val_n, val_batch_size):
                    vb_end = min(vb_start + val_batch_size, val_n)
                    vw = densify_batch(val_white[vb_start:vb_end])
                    vb = densify_batch(val_black[vb_start:vb_end])
                    vs = val_stm[vb_start:vb_end]
                    vr = val_result[vb_start:vb_end]
                    ve = val_eval[vb_start:vb_end]

                    val_phases_batch = val_phases[vb_start:vb_end]
                    loss_batch = compute_loss(net, vw, vb, vs, vr, ve, lam, eval_scale,
                                               label_smoothing=0.0, phases=val_phases_batch,
                                               wdl_alpha=args.wdl_alpha, wdl_lambda=current_wdl_lambda,
                                               wdl_draw_elo=args.wdl_draw_elo)
                    all_val_losses.append(loss_batch)

                    # AUDIT FIX C1: Pass phases, apply STM flip, remove /eval_scale
                    pred_eval_b = net(vw, vb, vs, phases=val_phases_batch)
                    pred_raw = pred_eval_b.squeeze(1)
                    pred_white = torch.where(vs < 0.5, pred_raw, -pred_raw)  # STM→White flip
                    pred_prob_b = torch.sigmoid(pred_white)
                    all_val_pred_probs.append(pred_prob_b)

            val_loss_tensor = torch.cat(all_val_losses)
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

            # --- Compute accuracy (prediction vs outcome agreement) ---
            pred_prob = torch.cat(all_val_pred_probs)
            # Classify: pred > 0.55 = win, pred < 0.45 = loss, else draw
            pred_class = torch.where(pred_prob > 0.55, torch.ones_like(pred_prob),
                         torch.where(pred_prob < 0.45, -torch.ones_like(pred_prob),
                         torch.zeros_like(pred_prob)))
            # T-L3: Use same 0.55/0.45 thresholds as predictions for comparability
            actual_class = torch.where(val_result > 0.55, torch.ones_like(val_result),
                           torch.where(val_result < 0.45, -torch.ones_like(val_result),
                           torch.zeros_like(val_result)))
            val_accuracy = (pred_class == actual_class).float().mean().item()

            # --- SWA update ---
            if swa is not None and (epoch + 1) >= args.swa_start:
                swa.update(net)

            epochs_done = epoch + 1
            epochs_left = args.epochs - epochs_done
            current_lr  = optimizer.param_groups[0]['lr']

            # --- Smart ETA (EWMA-only, no trend extrapolation) ---
            if ema_epoch_time is None:
                ema_epoch_time = epoch_time
            else:
                ema_epoch_time = EMA_ALPHA * epoch_time + (1 - EMA_ALPHA) * ema_epoch_time

            next_epoch_s = ema_epoch_time
            total_eta_s  = ema_epoch_time * epochs_left
            next_eta_str  = format_time(next_epoch_s) if next_epoch_s > 0 else "?"
            total_eta_str = format_time(total_eta_s)  if total_eta_s  > 0 else "?"

            swa_marker = " [SWA]" if swa is not None and epochs_done >= args.swa_start else ""
            pos_per_sec = num_batches * batch_size / max(epoch_time, 0.01)
            improved_marker = " *" if val_loss < best_loss else ""

            # T-M5: Note when train loss is rebalanced (weighted) and not directly comparable to val
            _rebalanced = (args.eval_soft_cap > 0 or args.draw_weight != 1.0 or args.mate_boost > 0)  # FIX
            _reb_note = " (rebalanced)" if _rebalanced else ""
            print(f"{'-'*70}")
            print(f"  Epoch {epochs_done:4d}/{args.epochs} | Train: {avg_loss:.6f}{_reb_note} | "
                  f"Val: {val_loss:.6f}{improved_marker} | LR: {current_lr:.6f} | Acc: {val_accuracy:.4f}")
            print(f"  Time: {format_time(epoch_time)} | "
                  f"Elapsed: {format_time(elapsed)} | "
                  f"{pos_per_sec:,.0f} pos/s | "
                  f"Next: ~{next_eta_str} | Total ETA: {total_eta_str}{swa_marker}")
            if has_phase_data:
                print(f"  Phase loss -> Opening: {phase_losses.get('opening',0):.6f}  "
                      f"Middlegame: {phase_losses.get('middlegame',0):.6f}  "
                      f"Endgame: {phase_losses.get('endgame',0):.6f}")
            if epochs_no_improve > 0 and args.early_stop > 0:
                print(f"  No improvement for {epochs_no_improve}/{args.early_stop} epochs")

            append_log(epochs_done, avg_loss, val_loss, current_lr, epoch_time,
                       run_id, phase_losses)
            all_log_data.append({
                'timestamp': datetime.now().isoformat(),
                'run_id': run_id, 'epoch': str(epochs_done),
                'loss': f'{avg_loss:.8f}', 'val_loss': f'{val_loss:.8f}', 'accuracy': f'{val_accuracy:.4f}',
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
                # AUDIT FIX M5: Save full training state for proper resume
                ckpt_data = {'optimizer': optimizer.state_dict(),
                             'epoch': epochs_done,
                             'global_step': global_step,
                             'best_loss': best_loss,
                             'epochs_no_improve': epochs_no_improve}
                if scheduler is not None:
                    ckpt_data['scheduler'] = scheduler.state_dict()
                safe_torch_save(ckpt_data, checkpoint_path)
            else:
                epochs_no_improve += 1
                if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
                    print(f"\nEarly stopping: no improvement for {args.early_stop} epochs.")
                    break

            if epochs_done % args.save_every == 0:
                periodic_path = os.path.splitext(args.output)[0] + '_checkpoint.bin'
                save_weights_cpp(net, periodic_path)
                # AUDIT FIX M5: Save full training state for proper resume
                ckpt_data = {'optimizer': optimizer.state_dict(),
                             'epoch': epochs_done,
                             'global_step': global_step,
                             'best_loss': best_loss,
                             'epochs_no_improve': epochs_no_improve}
                if scheduler is not None:
                    ckpt_data['scheduler'] = scheduler.state_dict()
                safe_torch_save(ckpt_data, checkpoint_path)

    except KeyboardInterrupt:
        print(f"\n\nStopped by user after epoch {epoch + 1}.")
        save_weights_cpp(net, args.output)
        # F2.2: Save full training state so resume works correctly
        ckpt_data = {'optimizer': optimizer.state_dict(),
                     'epoch': epoch + 1,
                     'global_step': global_step,
                     'best_loss': best_loss,
                     'epochs_no_improve': epochs_no_improve}
        if scheduler is not None:
            ckpt_data['scheduler'] = scheduler.state_dict()
        safe_torch_save(ckpt_data, checkpoint_path)
        print(f"Total time: {format_time(time.time() - start_time)}")
        if swa is not None and swa.count > 0:
            print("Applying SWA averaged weights...")
            swa.apply(net)
            swa_path = os.path.splitext(args.output)[0] + '_swa.bin'
            save_weights_cpp(net, swa_path)
            print(f"  SWA weights also saved to {swa_path}")
        return
    except Exception as _fatal_exc:
        import traceback as _tb
        print(f"\n[FATAL] Unhandled exception in training loop at epoch {epoch + 1}:", flush=True)
        _tb.print_exc()
        print(f"[FATAL] Training aborted. See traceback above.", flush=True)
        sys.exit(1)

    # --- End of training ---
    if swa is not None and swa.count > 0:
        # Best model is already saved at args.output — don't overwrite it
        print(f"Applying SWA averaged weights ({swa.count} snapshots)...")
        swa.apply(net)
        swa_path = os.path.splitext(args.output)[0] + '_swa.bin'
        save_weights_cpp(net, swa_path)
        print(f"  SWA weights saved to {swa_path}")
        print(f"  Best validation weights remain at {args.output}")

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
                        help='Primary training data file (.bin) or shards directory (default: assets/training_data.bin)')
    parser.add_argument('--shard-data',   type=int,   default=0, metavar='N',
                        help='Shard the --data file into N pieces for faster loading, then exit. '
                             'Creates a shards/ directory next to the source file.')
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
    parser.add_argument('--cosine-lr', action='store_true', default=True,
                        help='Use cosine annealing LR schedule (default)')
    parser.add_argument('--no-cosine-lr', dest='cosine_lr', action='store_false',
                        help='Use constant LR (no scheduler)')
    parser.add_argument('--cosine-t0',     type=int,   default=50)
    parser.add_argument('--cosine-t-mult', type=int,   default=2)
    parser.add_argument('--lr-min',        type=float, default=1e-6)
    parser.add_argument('--cosine-restarts', action='store_true', default=None,  # T-M4: sentinel
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

    # Stratified sampler (Phase 1.4)
    parser.add_argument('--stratified', action='store_true', default=False,
                        help='Use stratified sampler (replaces phase-balanced)')
    parser.add_argument('--draw-ratio', type=float, default=0.15,
                        help='Target fraction of draws per batch (default: 0.15)')
    parser.add_argument('--phase-ratio-opening', type=float, default=0.33,
                        help='Target sampling ratio for opening positions (default: 0.33)')
    parser.add_argument('--phase-ratio-middlegame', type=float, default=0.34,
                        help='Target sampling ratio for middlegame positions (default: 0.34)')
    parser.add_argument('--phase-ratio-endgame', type=float, default=0.33,
                        help='Target sampling ratio for endgame positions (default: 0.33)')
    parser.add_argument('--draw-oversample-epochs', type=int, default=0,
                        help='Oversample draws for first N epochs (default: 0=off)')
    parser.add_argument('--draw-oversample-factor', type=float, default=3.0,
                        help='Draw oversampling multiplier for early epochs (default: 3.0)')

    # Stopping & saving
    parser.add_argument('--early-stop',  type=int, default=15)
    parser.add_argument('--save-every',  type=int, default=10)

    # Plotting
    parser.add_argument('--plot',       action='store_true', help='Generate final plot after training (intermediate plots saved regardless)')
    parser.add_argument('--show-plot',  action='store_true', help='Open the plot image after saving (requires --plot)')
    parser.add_argument('--plot-every', type=int, default=10)
    parser.add_argument('--clear-log', action='store_true', help='Delete training_log.csv before training for a clean plot')

    # Streaming
    parser.add_argument('--chunk-size', type=int, default=STREAM_CHUNK_SIZE,
                        help=f'Positions per chunk in streaming mode (default: {STREAM_CHUNK_SIZE:,})')
    parser.add_argument('--max-positions', type=int, default=0,
                        help='Limit training to first N positions (0=all)')

    # === WDL CROSS-ENTROPY (Step 3.3) ===
    parser.add_argument('--wdl-alpha', type=float, default=0.0,
                        help='Weight of WDL cross-entropy loss vs scalar MSE loss. '
                             '0.0=MSE only (legacy), 1.0=CE only. (default: 0.0, recommended: 0.5)')
    parser.add_argument('--wdl-lambda', type=float, default=None,
                        help='Eval vs result blend for WDL targets. '
                             'None=use --lam value. 0.0=pure game result, 1.0=pure eval-based. '
                             '(default: None, uses --lam)')
    parser.add_argument('--wdl-draw-elo', type=float, default=100.0,
                        help='Draw bandwidth in centipawns for eval-based WDL targets. '
                             'Controls how wide the draw band is around eval=0. '
                             '0=no eval-based draw signal. (default: 100.0)')
    parser.add_argument('--adaptive-lambda', action='store_true', default=False,
                        help='Enable adaptive lambda scheduling: linearly ramp wdl_lambda '
                             'from --lambda-start to --lambda-end over training epochs')
    parser.add_argument('--lambda-start', type=float, default=1.0,
                        help='Starting wdl_lambda value (epoch 0). Default: 1.0 (pure eval)')
    parser.add_argument('--lambda-end', type=float, default=0.5,
                        help='Ending wdl_lambda value (last epoch). Default: 0.5 (balanced)')
    parser.add_argument('--dedup', action='store_true', default=False,
                        help='Enable hash-based deduplication of training positions within chunks')

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
    parser.add_argument('--draw-weight', type=float, default=None,
                        help='Loss multiplier for drawn positions (1.0=off, enhanced: 3.0). '
                             'Compensates for draw underrepresentation in dataset')
    parser.add_argument('--mate-boost', type=float, default=3.0,
                        help='Weight multiplier for high-eval (|eval|>2000cp) decisive '
                             'positions. Teaches mate patterns. 0=disabled. (default: 3.0)')

    args = parser.parse_args()

    # F2.3: Track which args user explicitly set (vs argparse defaults)
    # FRAGILE: This detection checks if option_strings appear in sys.argv, but
    # --no-<flag> style boolean flags (e.g. --no-cosine-lr) won't match the
    # dest name, so they may not be detected as explicitly set.
    _explicitly_set = {a.dest for a in parser._actions
                       if any(t in sys.argv for t in a.option_strings)}

    # Apply --enhanced defaults (only override if not explicitly set)
    if args.enhanced:
        if 'warmup_steps' not in _explicitly_set:
            args.warmup_steps = 1000
        if 'grad_accum' not in _explicitly_set:
            args.grad_accum = 4
        if 'swa' not in _explicitly_set:
            args.swa = True
        if 'label_smoothing' not in _explicitly_set:
            args.label_smoothing = 0.02
        if 'filter_eval_max' not in _explicitly_set:
            args.filter_eval_max = 1000.0  # centipawns: ±10 pawns
        if 'eval_soft_cap' not in _explicitly_set:
            args.eval_soft_cap = 800.0  # centipawns: ±8 pawns
        if 'draw_weight' not in _explicitly_set:
            args.draw_weight = 0.5
        # Enable WDL cross-entropy with sensible defaults
        if 'wdl_alpha' not in _explicitly_set:
            args.wdl_alpha = 0.5
        # Single cosine decay is better for fine-tuning; only override if user didn't pass flag (T-M4)
        if args.cosine_restarts is None:
            args.cosine_restarts = False
        dw_display = args.draw_weight if args.draw_weight is not None else 0.5
        print(f"=== Enhanced mode v3: warmup={args.warmup_steps}, grad_accum={args.grad_accum}, SWA=ON, "
              f"label_smooth={args.label_smoothing}, filter_eval={args.filter_eval_max}, "
              f"eval_soft_cap={args.eval_soft_cap}, draw_weight={dw_display}x, "
              f"wdl_alpha={args.wdl_alpha}, wdl_draw_elo={args.wdl_draw_elo}, cosine_decay ===\n")

    # Default draw_weight if neither user nor --enhanced set it
    if args.draw_weight is None:
        args.draw_weight = 1.0
    # T-M4: Resolve cosine_restarts sentinel — default ON unless --enhanced/--no-cosine-restarts set it
    if args.cosine_restarts is None:
        args.cosine_restarts = True

    # F6.5: Input validation
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not 0.0 <= args.wdl_alpha <= 1.0:
        parser.error("--wdl-alpha must be in [0, 1]")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.grad_accum < 1:
        parser.error("--grad-accum must be >= 1")

    if args.plot and not HAS_MATPLOTLIB:
        print("WARNING: matplotlib not found -- plot will be skipped. Install with: pip install matplotlib")

    # Handle --shard-data: one-time preprocessing, then exit
    if args.shard_data > 0:
        if not os.path.isfile(args.data):
            print(f"ERROR: --shard-data requires --data to point to a .bin file, got: {args.data}")
            sys.exit(1)
        shard_training_data(args.data, num_shards=args.shard_data)
        sys.exit(0)

    if args.clear_log and os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        print(f"Cleared {LOG_FILE} for fresh run.")
    train(args)

    if args.plot and HAS_MATPLOTLIB:
        generate_plot(show=args.show_plot)
