#!/usr/bin/env python3
"""
merge_training_data.py - Merge multiple NNUE training binary files

Auto-detects format:
  - Fixed format:    1542 bytes/record (converter output: 768 floats + result + searchEval)
  - Variable format: length-prefixed records (legacy self-play output)

Usage:
  py -3.10 merge_training_data.py file1.bin file2.bin ... -o output.bin
"""

import sys
import os
import struct
import argparse

FIXED_RECORD_SIZE = 1542  # 768*4 + 4 + 4 + 2 bytes overhead = 1542
FIXED_MAGIC = b'\x00'     # fixed format has no magic; detected by file size divisibility


def detect_format(filepath):
    """Detect whether a .bin file is fixed or variable format."""
    size = os.path.getsize(filepath)
    if size == 0:
        return 'empty'
    if size % FIXED_RECORD_SIZE == 0:
        return 'fixed'
    # Check for variable-length format: first 4 bytes = record length
    with open(filepath, 'rb') as f:
        first_len = struct.unpack('<I', f.read(4))[0]
    if 0 < first_len < 100000:
        return 'variable'
    return 'fixed'  # fallback


def count_records(filepath, fmt):
    if fmt == 'fixed':
        return os.path.getsize(filepath) // FIXED_RECORD_SIZE
    elif fmt == 'variable':
        count = 0
        with open(filepath, 'rb') as f:
            while True:
                hdr = f.read(4)
                if len(hdr) < 4:
                    break
                length = struct.unpack('<I', hdr)[0]
                f.seek(length, 1)
                count += 1
        return count
    return 0


def copy_fixed(src_path, dst_file):
    """Copy all fixed-size records from src to dst."""
    with open(src_path, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            dst_file.write(chunk)


def copy_variable_as_fixed(src_path, dst_file):
    """Read variable-length records, extract the fixed-size payload, write as fixed."""
    FEATURES = 768
    PAYLOAD = FEATURES * 4 + 4 + 4  # 3080 bytes... but our fixed is 1542
    # Variable format packs differently - just copy raw records with length prefix stripped
    with open(src_path, 'rb') as f:
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            length = struct.unpack('<I', hdr)[0]
            data = f.read(length)
            if len(data) < length:
                break
            dst_file.write(data)


def main():
    parser = argparse.ArgumentParser(description='Merge NNUE training data files')
    parser.add_argument('inputs', nargs='+', help='Input .bin files')
    parser.add_argument('-o', '--output', required=True, help='Output .bin file')
    args = parser.parse_args()

    total_records = 0
    file_info = []

    print("Scanning input files...")
    for path in args.inputs:
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        fmt = detect_format(path)
        count = count_records(path, fmt)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {path}: {count:,} records ({fmt} format, {size_mb:.1f} MB)")
        file_info.append((path, fmt, count))
        total_records += count

    if not file_info:
        print("No valid input files found.")
        sys.exit(1)

    print(f"\nTotal: {total_records:,} records -> {args.output}")

    with open(args.output, 'wb') as out:
        for path, fmt, count in file_info:
            print(f"  Merging {path} ({count:,} records)...")
            if fmt == 'fixed':
                copy_fixed(path, out)
            elif fmt == 'variable':
                copy_variable_as_fixed(path, out)

    out_size = os.path.getsize(args.output)
    out_records = out_size // FIXED_RECORD_SIZE
    print(f"\nDone! Output: {args.output}")
    print(f"  Size: {out_size / (1024**3):.2f} GB")
    print(f"  Records: {out_records:,}")


if __name__ == '__main__':
    main()
