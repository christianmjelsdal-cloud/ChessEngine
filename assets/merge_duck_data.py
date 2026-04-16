#!/usr/bin/env python3
"""
merge_duck_data.py - Merge duck chess self-play binary files

Format (written by SelfPlayGen.cpp):
  Header: 4 bytes magic ('NNUE') + 1 byte version + 4 bytes uint32 position count
  Records (variable length per position):
    uint16  nfeatures
    uint16  features[nfeatures]
    uint8   stm  (0=white, 1=black)
    float32 result
    float32 eval

Usage:
  py merge_duck_data.py file1.bin file2.bin ... -o output.bin [--include-base]

Examples:
  # Merge all selfplay gens into one file
  py merge_duck_data.py duck_selfplay_gen*.bin -o duck_all_selfplay.bin

  # Also include the base training data
  py merge_duck_data.py duck_selfplay_gen*.bin -o duck_all_selfplay.bin --base duck_training_data.bin
"""

import sys
import os
import struct
import argparse
import glob

MAGIC = b'NNUE'
FORMAT_VERSION = 1
HEADER_SIZE = 4 + 1 + 4  # magic + version + count


def read_header(f):
    """Read and validate file header. Returns position count or raises."""
    magic = f.read(4)
    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic!r} (expected b'NNUE')")
    version = struct.unpack('<B', f.read(1))[0]
    if version != FORMAT_VERSION:
        raise ValueError(f"Unknown version: {version}")
    count = struct.unpack('<I', f.read(4))[0]
    return count


def iter_records_raw(f, count):
    """Yield raw bytes for each record (including the nfeatures prefix)."""
    for _ in range(count):
        nf_bytes = f.read(2)
        if len(nf_bytes) < 2:
            break
        nf = struct.unpack('<H', nf_bytes)[0]
        # features + stm + result + eval
        rest_size = nf * 2 + 1 + 4 + 4
        rest = f.read(rest_size)
        if len(rest) < rest_size:
            break
        yield nf_bytes + rest


def scan_file(path):
    """Return (count, byte_size_of_records) for a file."""
    size = os.path.getsize(path)
    if size < HEADER_SIZE:
        return 0, 0
    with open(path, 'rb') as f:
        try:
            count = read_header(f)
        except ValueError:
            return 0, 0
        record_bytes = size - HEADER_SIZE
    return count, record_bytes


def main():
    parser = argparse.ArgumentParser(description='Merge duck chess training data files')
    parser.add_argument('inputs', nargs='+', help='Input .bin files (supports globs)')
    parser.add_argument('-o', '--output', required=True, help='Output .bin file')
    parser.add_argument('--base', help='Optional base dataset to include (e.g. duck_training_data.bin)')
    args = parser.parse_args()

    # Expand globs (useful when called from a shell that doesn't expand them)
    input_paths = []
    for pattern in args.inputs:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            input_paths.extend(expanded)
        elif os.path.exists(pattern):
            input_paths.append(pattern)
        else:
            print(f"  WARNING: no files matched '{pattern}', skipping")

    if args.base:
        if os.path.exists(args.base):
            input_paths.insert(0, args.base)
        else:
            print(f"  WARNING: base file '{args.base}' not found, skipping")

    # Deduplicate while preserving order
    seen = set()
    unique_paths = []
    for p in input_paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            unique_paths.append(p)
    input_paths = unique_paths

    if not input_paths:
        print("No input files found.")
        sys.exit(1)

    # Scan all files
    print("Scanning input files...")
    valid_files = []
    total_positions = 0
    for path in input_paths:
        count, rec_bytes = scan_file(path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if count == 0:
            print(f"  SKIP  {path}  (unrecognised format or empty)")
            continue
        print(f"  OK    {path}  {count:>8,} positions  ({size_mb:.1f} MB)")
        valid_files.append((path, count))
        total_positions += count

    if not valid_files:
        print("No valid files to merge.")
        sys.exit(1)

    out_path = args.output
    if os.path.abspath(out_path) in {os.path.abspath(p) for p, _ in valid_files}:
        print(f"ERROR: output '{out_path}' is the same as one of the inputs.")
        sys.exit(1)

    print(f"\nMerging {total_positions:,} positions -> {out_path}")

    with open(out_path, 'wb') as out:
        # Write header with total count
        out.write(MAGIC)
        out.write(struct.pack('<B', FORMAT_VERSION))
        out.write(struct.pack('<I', min(total_positions, 0xFFFFFFFF)))

        written = 0
        for path, count in valid_files:
            with open(path, 'rb') as f:
                read_header(f)  # skip header
                for raw in iter_records_raw(f, count):
                    out.write(raw)
                    written += 1

    out_size = os.path.getsize(out_path)
    print(f"\nDone!")
    print(f"  Positions written : {written:,}")
    print(f"  Output size       : {out_size / (1024*1024):.1f} MB  ({out_path})")


if __name__ == '__main__':
    main()
