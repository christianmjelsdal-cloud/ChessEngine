"""
merge_training_data.py
Merges two or more training_data.bin files into one, shuffling the result.

Usage:
  py -3.10 merge_training_data.py file1.bin file2.bin -o merged.bin

The merged file can then be used directly as your training data.
"""

import struct
import random
import argparse
import os


def read_binary(filename: str) -> list:
    """Read all raw position bytes from a .bin file."""
    positions = []
    with open(filename, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        for _ in range(n):
            # read numFeatures
            nf_bytes = f.read(2)
            nf = struct.unpack("<H", nf_bytes)[0]
            # read features + stm + gameResult + searchEval
            body = f.read(nf * 2 + 1 + 4 + 4)
            positions.append(nf_bytes + body)
    print(f"  Read {len(positions):,} positions from {filename}")
    return positions


def write_binary(positions: list, filename: str):
    """Write raw position bytes to a .bin file."""
    with open(filename, "wb") as f:
        f.write(struct.pack("<I", len(positions)))
        for raw in positions:
            f.write(raw)
    print(f"  Wrote {len(positions):,} positions → {filename}")


def main():
    parser = argparse.ArgumentParser(description="Merge NNUE training .bin files")
    parser.add_argument("inputs", nargs="+", help="Input .bin files")
    parser.add_argument("-o", "--output", required=True, help="Output .bin file")
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    all_positions = []
    for f in args.inputs:
        if not os.path.exists(f):
            print(f"  !! File not found: {f}")
            continue
        all_positions.extend(read_binary(f))

    print(f"\nTotal: {len(all_positions):,} positions")

    if not args.no_shuffle:
        print("Shuffling...")
        random.shuffle(all_positions)

    write_binary(all_positions, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
