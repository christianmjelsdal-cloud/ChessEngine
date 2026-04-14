#!/usr/bin/env python3
"""
merge_training_data.py - Merge multiple NNUE training binary files

Binary format (little-endian, matching SelfPlayGen / train_nnue.py):
  Header:  [uint32_t  position_count]
  Records: [uint16_t  num_features]
           [uint16_t  feature_indices[num_features]]  (sorted ascending)
           [uint8_t   stm]     0 = white, 1 = black
           [float32   result]  1.0 = white wins, 0.5 = draw, 0.0 = black wins
           [float32   eval]    White-perspective eval in centipawns

Usage:
  py -3.10 merge_training_data.py file1.bin file2.bin ... -o output.bin
"""

import sys
import os
import struct
import argparse
from training_format import read_header as tf_read_header, write_header as tf_write_header


def read_header(filepath):
    """Read position count and data offset, handling both legacy and versioned formats.
    FIX H-1: Was hardcoded to 4-byte legacy header; now uses training_format.py."""
    version, count, data_offset = tf_read_header(filepath)
    return count, data_offset


def main():
    parser = argparse.ArgumentParser(description='Merge NNUE training data files')
    parser.add_argument('inputs', nargs='+', help='Input .bin files')
    parser.add_argument('-o', '--output', required=True, help='Output .bin file')
    args = parser.parse_args()

    total_positions = 0
    valid_files = []

    print("Scanning input files...")
    for path in args.inputs:
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        size = os.path.getsize(path)
        if size < 4:
            print(f"  WARNING: {path} too small ({size} bytes), skipping")
            continue
        count, data_offset = read_header(path)
        size_mb = size / (1024 * 1024)
        print(f"  {path}: {count:,} positions ({size_mb:.1f} MB)")
        valid_files.append((path, count, data_offset))
        total_positions += count

    if not valid_files:
        print("No valid input files found.")
        sys.exit(1)

    print(f"\nTotal: {total_positions:,} positions -> {args.output}")

    CHUNK = 1024 * 1024  # 1 MB streaming buffer

    with open(args.output, 'wb') as out:
        # FIX H-1: Write versioned header instead of legacy 4-byte header
        tf_write_header(out, total_positions)

        for path, count, data_offset in valid_files:
            print(f"  Merging {path} ({count:,} positions)...")
            with open(path, 'rb') as inp:
                inp.seek(data_offset)  # FIX H-1: skip correct header size (4 or 9 bytes)
                while True:
                    chunk = inp.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)

    out_size = os.path.getsize(args.output)
    verify_count, _ = read_header(args.output)
    print(f"\nDone! Output: {args.output}")
    print(f"  Size: {out_size / (1024**2):.1f} MB ({out_size / (1024**3):.2f} GB)")
    print(f"  Header says: {verify_count:,} positions")

    if verify_count != total_positions:
        print(f"  WARNING: Header mismatch! Expected {total_positions:,}, got {verify_count:,}")
        sys.exit(1)


if __name__ == '__main__':
    main()
