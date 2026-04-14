"""
training_format.py — Shared constants and helpers for the NNUE binary training format.

AUDIT FIX PT-3: Binary format version header.
AUDIT FIX PT-6: Shared feature encoding utilities.

Format v1 layout:
    [4 bytes]  MAGIC   = b'NNUE'
    [1 byte ]  VERSION = 1
    [4 bytes]  uint32  position_count
    [variable] position records...

Each position record:
    [2 bytes]  uint16  num_features
    [2*N bytes] uint16[] feature_indices
    [1 byte ]  uint8   side_to_move (0=White, 1=Black)
    [4 bytes]  float   game_result (1.0=White win, 0.5=draw, 0.0=Black win)
    [4 bytes]  float   search_eval (centipawns, white POV)

Legacy format (v0) has no magic header — starts directly with uint32 position_count.
Readers should detect format by checking the first 4 bytes against MAGIC.
"""

import struct
import os

# ── Format constants ─────────────────────────────────────────────────────────

MAGIC   = b'NNUE'       # 4-byte magic number
VERSION = 1             # format version
HEADER_SIZE = 4 + 1 + 4  # magic(4) + version(1) + count(4) = 9 bytes

# ── Feature encoding (HalfKAv2: 12 piece types × 64 squares = 768) ──────────

def feature_index(piece_type_0based: int, color_is_black: bool, square: int) -> int:
    """
    Convert a piece on a square to a 0..767 feature index.

    Args:
        piece_type_0based: 0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King
        color_is_black:    True if the piece is black
        square:            0..63 (rank*8 + file, a1=0)
    """
    pt_idx = piece_type_0based
    if color_is_black:
        pt_idx += 6
    return pt_idx * 64 + square


# ── Format detection ─────────────────────────────────────────────────────────

def detect_format_version(filepath: str) -> int:
    """
    Detect the format version of a binary training file.
    Returns 1 for versioned format, 0 for legacy format.
    """
    with open(filepath, 'rb') as f:
        header = f.read(4)
        if header == MAGIC:
            return 1
        return 0


def read_header(filepath: str):
    """
    Read the file header and return (version, position_count, data_offset).
    Works with both legacy (v0) and versioned (v1) formats.
    """
    with open(filepath, 'rb') as f:
        magic = f.read(4)
        if magic == MAGIC:
            version = struct.unpack('<B', f.read(1))[0]
            count = struct.unpack('<I', f.read(4))[0]
            return version, count, HEADER_SIZE
        else:
            # Legacy format: first 4 bytes are the count
            count = struct.unpack('<I', magic)[0]
            return 0, count, 4


def write_header(f, count: int):
    """Write a versioned header to an open file handle."""
    f.write(MAGIC)
    f.write(struct.pack('<B', VERSION))
    f.write(struct.pack('<I', count))


def pack_position(features, stm, result, eval_val):
    """Pack a single position into binary bytes (no header).
    
    Args:
        features: list of uint16 feature indices
        stm: 0=White, 1=Black
        result: float game result (1.0=White win, 0.5=draw, 0.0=Black win)
        eval_val: float search eval in centipawns (white POV)
    """
    nf = len(features)
    return struct.pack(f"<H{nf}HBff", nf, *features, stm, result, eval_val)
