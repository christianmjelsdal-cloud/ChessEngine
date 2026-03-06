"""
Lichess Elite Database → NNUE Training Data Converter
=====================================================

Downloads and converts high-rated (2500+) Lichess games to your binary training format.

Usage:
  1. Download a PGN from https://database.nikonoel.fr/ (Lichess Elite Database)
     - Pick any month(s), e.g. "lichess_elite_2025-01.pgn.zst" or .pgn.bz2
     - Extract to a .pgn file (use 7-Zip or similar)
  
  2. Run:  py -3.10 convert_lichess_elite.py input.pgn --max-games 100000
  
  3. Merge with existing data:
     py -3.10 merge_training_data.py assets/training_data.bin lichess_elite_training.bin merged.bin
     copy merged.bin assets\\training_data.bin

Requirements:  pip install chess
"""

import struct, os, random, time, argparse
import chess, chess.pgn

MIN_PLY = 8       # skip opening book moves
MAX_PLY = 200     # skip ultra-long endgames
SAMPLE_EVERY = 2  # take every 2nd position (reduces redundancy)

PIECE_MAT = [0, 100, 320, 330, 500, 900, 0]
RESULTS = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def pack_position(board, result):
    """Pack a board position into your NNUE binary format."""
    feats = []
    score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            pt = p.piece_type - 1 + (6 if p.color == chess.BLACK else 0)
            feats.append(pt * 64 + chess.square_rank(sq) * 8 + chess.square_file(sq))
            score += PIECE_MAT[p.piece_type] * (1 if p.color == chess.WHITE else -1)
    nf = len(feats)
    return struct.pack(f"<H{nf}HBff", nf, *feats,
                       0 if board.turn == chess.WHITE else 1, result, float(score))


def convert(pgn_path, out_path, max_games):
    positions = []
    gc = 0
    skipped = 0
    t0 = time.time()

    print(f"Reading: {pgn_path}")
    print(f"Max games: {max_games:,}\n")

    with open(pgn_path, "r", errors="ignore") as f:
        while gc < max_games:
            try:
                g = chess.pgn.read_game(f)
            except Exception:
                skipped += 1
                continue
            if g is None:
                break

            r = RESULTS.get(g.headers.get("Result"))
            if r is None:
                skipped += 1
                continue

            gc += 1
            board = g.board()
            ply = 0
            for move in g.mainline_moves():
                board.push(move)
                ply += 1
                if ply < MIN_PLY:
                    continue
                if ply > MAX_PLY:
                    break
                if (ply - MIN_PLY) % SAMPLE_EVERY != 0:
                    continue
                if board.is_check():
                    continue
                positions.append(pack_position(board, r))

            if gc % 5000 == 0:
                el = time.time() - t0
                rate = gc / el
                eta = (max_games - gc) / rate if rate > 0 else 0
                print(f"  {gc:,} games → {len(positions):,} positions "
                      f"({rate:.0f} g/s, ETA {eta/60:.0f}m)")

    elapsed = time.time() - t0
    print(f"\nDone: {gc:,} games → {len(positions):,} positions in {elapsed:.0f}s")
    if skipped:
        print(f"  ({skipped:,} games skipped)")

    print("Shuffling...")
    random.shuffle(positions)

    print(f"Writing to {out_path}...")
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", len(positions)))
        for p in positions:
            f.write(p)

    sz = os.path.getsize(out_path)
    print(f"Saved {len(positions):,} positions ({sz / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Lichess PGN to NNUE training data")
    parser.add_argument("pgn", help="Path to .pgn file")
    parser.add_argument("--output", "-o", default="lichess_elite_training.bin",
                        help="Output binary file (default: lichess_elite_training.bin)")
    parser.add_argument("--max-games", "-n", type=int, default=999999999,
                        help="Max games to convert (default: all)")
    args = parser.parse_args()
    convert(args.pgn, args.output, args.max_games)
