"""
pgn_to_training.py
Converts PGN game files to training_data.bin format for the chess NNUE engine.

Binary format (matching NNUETrainer.cpp saveTrainingData):
  uint32_t numPositions
  For each position:
    uint16_t numFeatures
    uint16_t[numFeatures] featureIndices  (0..767)
    uint8_t sideToMove  (0=White, 1=Black)
    float   gameResult  (1.0=white win, 0.5=draw, 0.0=black win)
    float   searchEval  (pawn units, white's POV, approximate material count)

Feature encoding (768 features = 12 piece types * 64 squares):
  featureIndex = pieceIndex * 64 + rank * 8 + file
  pieceIndex   = pieceType-1  (Pawn=0..King=5)  for White
               = pieceType-1 + 6                for Black
  rank  = chess.square_rank(sq)  (0=rank1 / white back rank .. 7=rank8)
  file  = chess.square_file(sq)  (0=a .. 7=h)

Usage:
  uv run --with chess python3 pgn_to_training.py [pgn_file1] [pgn_file2] ... -o output.bin
  Or run without args to use hardcoded sources.
"""

import struct
import sys
import os
import io
import zipfile
import subprocess
import chess
import chess.pgn
import random
import argparse
from pathlib import Path

# ── constants ────────────────────────────────────────────────────────────────
PIECE_MATERIAL = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,
}

# top pgnmentor player files ordered by game count (more = more diverse data)
PGN_MENTOR_PLAYERS = [
    ("Fischer",         "https://www.pgnmentor.com/players/Fischer.zip"),
    ("Capablanca",      "https://www.pgnmentor.com/players/Capablanca.zip"),
    ("Tal",             "https://www.pgnmentor.com/players/Tal.zip"),
    ("Kasparov",        "https://www.pgnmentor.com/players/Kasparov.zip"),
    ("Karpov",          "https://www.pgnmentor.com/players/Karpov.zip"),
    ("Carlsen",         "https://www.pgnmentor.com/players/Carlsen.zip"),
    ("Nakamura",        "https://www.pgnmentor.com/players/Nakamura.zip"),
    ("Kramnik",         "https://www.pgnmentor.com/players/Kramnik.zip"),
]

# ── feature extraction ────────────────────────────────────────────────────────

def feature_index(piece: chess.Piece, sq: int) -> int:
    """Convert a piece on a square to a 0..767 feature index."""
    pt_idx = piece.piece_type - 1          # Pawn=0 .. King=5
    if piece.color == chess.BLACK:
        pt_idx += 6                        # Black pieces: 6..11
    rank = chess.square_rank(sq)           # 0=rank1 .. 7=rank8
    file = chess.square_file(sq)           # 0=file-a .. 7=file-h
    return pt_idx * 64 + rank * 8 + file


def extract_features(board: chess.Board):
    """Return list of active feature indices (one per piece on the board)."""
    feats = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            feats.append(feature_index(p, sq))
    return feats


def material_eval(board: chess.Board) -> float:
    """Simple material count in pawn units, white's POV.
    (Normalized: pawn = 1.0, knight = 3.2, etc.)"""
    score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            v = PIECE_MATERIAL[p.piece_type]
            score += v if p.color == chess.WHITE else -v
    return float(score) / 100.0  # normalize centipawns → pawn units


def result_to_float(result_str: str) -> float | None:
    """'1-0' -> 1.0, '0-1' -> 0.0, '1/2-1/2' -> 0.5, else None."""
    if result_str == "1-0":   return 1.0
    if result_str == "0-1":   return 0.0
    if result_str == "1/2-1/2": return 0.5
    return None


# ── Draw source detection ────────────────────────────────────────────────────
# Draw source types (must match DrawSource enum)
# 0 = NOT_DRAW, 1 = UNKNOWN, 2 = STALEMATE, 3 = INSUFFICIENT,
# 4 = FIFTY_MOVE, 5 = REPETITION, 6 = AGREEMENT, 7 = TABLEBASE
DRAW_SOURCE_NAMES = {
    0: 'NOT_DRAW', 1: 'UNKNOWN', 2: 'STALEMATE', 3: 'INSUFFICIENT',
    4: 'FIFTY_MOVE', 5: 'REPETITION', 6: 'AGREEMENT', 7: 'TABLEBASE',
}

def detect_draw_source(game: chess.pgn.Game, board: chess.Board) -> int:
    """Detect draw source type from a PGN game's final position.
    
    Returns draw source code (0-7). Call after replaying all moves.
    """
    result = game.headers.get("Result", "*")
    if result != "1/2-1/2":
        return 0  # NOT_DRAW

    # Check terminal position for specific draw types
    if board.is_stalemate():
        return 2  # STALEMATE
    if board.is_insufficient_material():
        return 3  # INSUFFICIENT
    if board.can_claim_fifty_moves():
        return 4  # FIFTY_MOVE
    if board.can_claim_threefold_repetition():
        return 5  # REPETITION

    return 6  # AGREEMENT (default for draws without clear reason)


# ── game → positions ──────────────────────────────────────────────────────────

# Global draw source statistics (accumulated across all games)
_draw_source_stats = {}

def game_to_positions(game: chess.pgn.Game,
                      min_ply: int = 10,
                      max_ply: int = 100,
                      sample_every: int = 3):
    """
    Walk through a game, extract every sample_every-th position
    in [min_ply, max_ply].  Returns list of (features, stm, gameResult, searchEval).
    Also tracks draw source statistics.
    """
    result = result_to_float(game.headers.get("Result", "*"))
    if result is None:
        return []

    board = game.board()
    positions = []
    ply = 0

    for move in game.mainline_moves():
        board.push(move)
        ply += 1
        if ply < min_ply:
            continue
        if ply > max_ply:
            break
        if (ply - min_ply) % sample_every != 0:
            continue

        # Skip positions in check (noisy) and terminal
        if board.is_check() or board.is_game_over():
            continue

        feats  = extract_features(board)
        stm    = 0 if board.turn == chess.WHITE else 1
        s_eval = material_eval(board)

        positions.append((feats, stm, result, s_eval))

    # Detect draw source for the game (replay remaining moves to reach final position)
    while True:
        # INFO [7.23]: Reuse board already advanced through mainline
        final_board = board
        break
    draw_source = detect_draw_source(game, final_board)
    _draw_source_stats[draw_source] = _draw_source_stats.get(draw_source, 0) + 1

    return positions


# ── binary I/O ────────────────────────────────────────────────────────────────

def write_binary(positions: list, filename: str):
    """Write positions to a .bin file in the NNUETrainer format.
    AUDIT FIX PT-3: Now writes versioned header (magic + version + count).
    """
    from training_format import write_header
    with open(filename, "wb") as f:
        write_header(f, len(positions))
        for feats, stm, game_result, search_eval in positions:
            f.write(struct.pack("<H", len(feats)))
            for feat in feats:
                f.write(struct.pack("<H", feat))
            f.write(struct.pack("<B", stm))
            f.write(struct.pack("<f", game_result))
            f.write(struct.pack("<f", search_eval))
    print(f"  Wrote {len(positions):,} positions → {filename}")


def append_binary(positions: list, filename: str):
    """Append positions to an existing .bin file, updating the count header.
    AUDIT FIX PT-3: Supports both versioned (v1) and legacy (v0) files.
    AUDIT FIX F1.3: Crash-safe — writes data first, updates header count last.
    """
    from training_format import MAGIC, HEADER_SIZE, write_header
    if not os.path.exists(filename):
        write_binary(positions, filename)
        return

    with open(filename, "r+b") as f:
        magic = f.read(4)
        if magic == MAGIC:
            # Versioned format: count at offset 5
            f.read(1)  # skip version
            existing = struct.unpack("<I", f.read(4))[0]
            count_offset = 5
        else:
            # Legacy format: count at offset 0
            existing = struct.unpack("<I", magic)[0]
            count_offset = 0

        # Step 1: Append position data FIRST (before updating count)
        f.seek(0, 2)  # seek to end
        for feats, stm, game_result, search_eval in positions:
            f.write(struct.pack("<H", len(feats)))
            for feat in feats:
                f.write(struct.pack("<H", feat))
            f.write(struct.pack("<B", stm))
            f.write(struct.pack("<f", game_result))
            f.write(struct.pack("<f", search_eval))
        f.flush()
        os.fsync(f.fileno())

        # Step 2: Update header count AFTER data is safely on disk
        new_count = existing + len(positions)
        f.seek(count_offset)
        f.write(struct.pack("<I", new_count))
        f.flush()
        os.fsync(f.fileno())

    print(f"  Appended {len(positions):,} positions (total {new_count:,}) → {filename}")


# ── PGN processing ────────────────────────────────────────────────────────────

def process_pgn_text(pgn_text: str,
                     source_name: str = "?",
                     max_games: int = 99999,
                     min_ply: int = 10,
                     max_ply: int = 100,
                     sample_every: int = 3) -> list:
    """Parse PGN text, extract positions from all games."""
    # TODO: Consider using python-chess library's PGN parser for robust edge case handling.
    # Current parser may not handle: promotion with check (e8=Q+), null moves (--),
    # long algebraic notation, or nested variations.
    pgn_io = io.StringIO(pgn_text)
    all_positions = []
    game_count = 0
    error_count = 0
    max_consecutive_errors = 100
    consecutive_errors = 0

    while game_count < max_games:
        try:
            game = chess.pgn.read_game(pgn_io)
        except Exception:
            error_count += 1
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                print(f"Warning: {max_consecutive_errors} consecutive parse errors, stopping")
                break
            continue
        if game is None:
            break
        game_count += 1
        try:
            positions = game_to_positions(game, min_ply, max_ply, sample_every)
            all_positions.extend(positions)
            consecutive_errors = 0  # FIX 7.17: reset after successful game_to_positions
        except Exception:
            error_count += 1
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                print(f"Warning: {max_consecutive_errors} consecutive parse errors, stopping")
                break

    print(f"  {source_name}: {game_count} games → {len(all_positions):,} positions  (errors: {error_count})")
    return all_positions


def process_pgn_file(filepath: str, **kwargs) -> list:
    """Read a .pgn file and process it using streaming (no full-file read)."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return process_pgn_stream(f, source_name=os.path.basename(filepath), **kwargs)


def process_pgn_stream(pgn_file,
                       source_name: str = "?",
                       max_games: int = 99999,
                       min_ply: int = 10,
                       max_ply: int = 100,
                       sample_every: int = 3) -> list:
    """Parse PGN from a file handle using streaming reads, flushing chunks periodically."""
    CHUNK_SIZE = 500_000
    all_positions = []
    game_count = 0
    error_count = 0
    max_consecutive_errors = 100
    consecutive_errors = 0

    while game_count < max_games:
        try:
            game = chess.pgn.read_game(pgn_file)
        except Exception:
            error_count += 1
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                print(f"Warning: {max_consecutive_errors} consecutive parse errors, stopping")
                break
            continue
        if game is None:
            break
        game_count += 1
        try:
            positions = game_to_positions(game, min_ply, max_ply, sample_every)
            all_positions.extend(positions)
            consecutive_errors = 0  # FIX 7.17: reset after successful game_to_positions
        except Exception:
            error_count += 1
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                print(f"Warning: {max_consecutive_errors} consecutive parse errors, stopping")
                break

        if len(all_positions) >= CHUNK_SIZE:
            print(f"  {source_name}: chunk flushed ({len(all_positions):,} positions, {game_count} games so far)")

    print(f"  {source_name}: {game_count} games → {len(all_positions):,} positions  (errors: {error_count})")
    return all_positions


def download_and_process_zip(url: str, name: str, **kwargs) -> list:
    """Download a .zip from pgnmentor, extract the first .pgn, process it."""
    print(f"  Downloading {name}...")
    try:
        result = subprocess.run(
            ["curl", "-L", "-s", "--max-time", "60",
             "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
             url],
            capture_output=True, timeout=65
        )
        if result.returncode != 0:
            print(f"  !! curl failed for {name}")
            return []
        data = result.stdout
    except Exception as e:
        print(f"  !! Failed to download {name}: {e}")
        return []

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            pgn_names = [n for n in z.namelist() if n.lower().endswith(".pgn")]
            if not pgn_names:
                print(f"  !! No .pgn in zip for {name}")
                return []
            with z.open(pgn_names[0]) as pf:
                text = pf.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"  !! Failed to read zip for {name}: {e}")
        return []

    return process_pgn_text(text, source_name=name, **kwargs)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert PGN files to NNUE training_data.bin")
    parser.add_argument("pgn_files", nargs="*", help="Local .pgn files to process")
    parser.add_argument("-o", "--output", default="/tmp/pgn_training_data.bin",
                        help="Output .bin file")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip downloading from pgnmentor")
    parser.add_argument("--max-games", type=int, default=99999)
    parser.add_argument("--min-ply",   type=int, default=10,
                        help="First ply to include (skip opening)")
    parser.add_argument("--max-ply",   type=int, default=100,
                        help="Last ply to include (skip endgames)")
    parser.add_argument("--sample-every", type=int, default=3,
                        help="Sample one position every N plies")
    args = parser.parse_args()

    all_positions = []

    # ── 1. Local PGN files (including uploaded ones) ──────────────────────────
    for pgn_path in args.pgn_files:
        if not os.path.exists(pgn_path):
            print(f"  !! File not found: {pgn_path}")
            continue
        print(f"Processing local file: {pgn_path}")
        positions = process_pgn_file(pgn_path,
                                     max_games=args.max_games,
                                     min_ply=args.min_ply,
                                     max_ply=args.max_ply,
                                     sample_every=args.sample_every)
        all_positions.extend(positions)

    # ── 2. Download from pgnmentor ────────────────────────────────────────────
    if not args.no_download:
        print("\nDownloading from pgnmentor.com...")
        for name, url in PGN_MENTOR_PLAYERS:
            positions = download_and_process_zip(url, name,
                                                 max_games=args.max_games,
                                                 min_ply=args.min_ply,
                                                 max_ply=args.max_ply,
                                                 sample_every=args.sample_every)
            all_positions.extend(positions)

    # ── 3. Shuffle and write ──────────────────────────────────────────────────
    print(f"\nTotal positions collected: {len(all_positions):,}")
    random.shuffle(all_positions)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_binary(all_positions, args.output)

    # ── 4. Print draw source statistics ─────────────────────────────────────
    total_games = sum(_draw_source_stats.values())
    if total_games > 0:
        print(f"\n  {'='*50}")
        print(f"  DRAW SOURCE STATISTICS ({total_games:,} games)")
        print(f"  {'='*50}")
        for code in sorted(_draw_source_stats.keys()):
            count = _draw_source_stats[code]
            name = DRAW_SOURCE_NAMES.get(code, f'UNKNOWN({code})')
            pct = count / total_games * 100
            print(f"    {name:<16}: {count:>8,}  ({pct:5.1f}%)")
        # Summary
        draw_games = sum(v for k, v in _draw_source_stats.items() if k != 0)
        non_draw = _draw_source_stats.get(0, 0)
        print(f"  {'─'*50}")
        print(f"    {'Decisive':<16}: {non_draw:>8,}  ({non_draw/total_games*100:5.1f}%)")
        print(f"    {'Drawn':<16}: {draw_games:>8,}  ({draw_games/total_games*100:5.1f}%)")
        print(f"  {'='*50}")

    print("\nDone!")


if __name__ == "__main__":
    main()
