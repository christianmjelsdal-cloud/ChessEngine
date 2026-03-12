#!/usr/bin/env python3
"""
generate_selfplay.py — Self-play data generator for NNUE training.

Runs your chess engine against itself via UCI, recording positions with
evaluations and game outcomes in the training binary format.

Binary format (compatible with train_nnue.py):
  Header:  [4B: uint32 position_count]
  Record:  [2B: num_features] [num_features * 2B: feature_indices] [1B: stm] [4B: result] [4B: eval]

Usage:
    # Basic (500 games, depth 8):
    py -3.10 generate_selfplay.py --engine path/to/engine.exe --games 500

    # Faster, shallower search:
    py -3.10 generate_selfplay.py --engine engine.exe --games 1000 --depth 6 --workers 4

    # With opening book and specific output:
    py -3.10 generate_selfplay.py --engine engine.exe --games 500 --depth 8 \
        --opening-book openings.epd --output assets/selfplay_v1.bin

    # With time control instead of depth:
    py -3.10 generate_selfplay.py --engine engine.exe --games 500 --movetime 100

    # Merge into training data:
    py -3.10 generate_selfplay.py --engine engine.exe --games 500 --merge assets/training_data.bin

Dependencies: python-chess (pip install python-chess)
"""

import argparse
import chess
import os
import random
import signal
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# =============================================================================
# Graceful shutdown support
# =============================================================================

shutdown_event = threading.Event()

def _signal_handler(signum, frame):
    """Handle Ctrl+C: signal all workers to stop after their current game."""
    if shutdown_event.is_set():
        # Second Ctrl+C: force exit
        print("\n\n⚠ Force quit! Data NOT saved.")
        sys.exit(1)
    shutdown_event.set()
    print("\n\n🛑 Shutdown requested — finishing current games and saving data...")
    print("   (press Ctrl+C again to force quit without saving)\n")

signal.signal(signal.SIGINT, _signal_handler)

# =============================================================================
# Feature encoding (must match train_nnue.py / generate_draws.py)
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
    """Convert board to sorted feature indices (piece_index * 64 + square)."""
    features = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            idx = PIECE_TO_INDEX[(piece.piece_type, piece.color)]
            features.append(idx * 64 + square)
    return sorted(features)

# =============================================================================
# UCI engine interface
# =============================================================================

class UCIEngine:
    """Manages a UCI engine subprocess."""

    def __init__(self, engine_path, options=None):
        self.path = engine_path
        self.process = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")

        # Apply custom UCI options
        if options:
            for name, value in options.items():
                self._send(f"setoption name {name} value {value}")

        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd):
        try:
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()
        except (OSError, BrokenPipeError):
            raise RuntimeError("Engine process died")

    def _wait_for(self, token, timeout=30):
        """Read lines until we see one starting with token. Aborts on shutdown."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if shutdown_event.is_set():
                raise InterruptedError("Shutdown requested")
            line = self.process.stdout.readline().strip()
            if not line and self.process.poll() is not None:
                raise RuntimeError("Engine process died")
            if line.startswith(token):
                return line
        raise TimeoutError(f"Engine didn't respond with '{token}' within {timeout}s")

    def new_game(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")

    def search(self, board, depth=None, movetime=None, nodes=None):
        """
        Search a position. Returns (best_move_uci, score_cp, score_mate).
        score_cp is centipawns from engine's perspective (side to move).
        score_mate is mate-in-N (positive = engine mates, negative = engine gets mated).
        """
        fen = board.fen()
        self._send(f"position fen {fen}")

        go_cmd = "go"
        if depth is not None:
            go_cmd += f" depth {depth}"
        elif movetime is not None:
            go_cmd += f" movetime {movetime}"
        elif nodes is not None:
            go_cmd += f" nodes {nodes}"
        else:
            go_cmd += " depth 8"

        self._send(go_cmd)

        best_move = None
        score_cp = 0
        score_mate = None

        while True:
            line = self.process.stdout.readline().strip()
            if not line:
                continue

            if line.startswith("info") and " score " in line:
                parts = line.split()
                try:
                    si = parts.index("score")
                    if parts[si + 1] == "cp":
                        score_cp = int(parts[si + 2])
                        score_mate = None
                    elif parts[si + 1] == "mate":
                        score_mate = int(parts[si + 2])
                        score_cp = 0
                except (IndexError, ValueError):
                    pass

            if line.startswith("bestmove"):
                best_move = line.split()[1]
                break

        return best_move, score_cp, score_mate

    def quit(self):
        try:
            self._send("quit")
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()

# =============================================================================
# Opening book
# =============================================================================

def load_opening_book(path):
    """Load an EPD/FEN opening book. Returns list of FEN strings."""
    fens = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # EPD format: pieces active castling ep [opcodes...]
            # FEN format: pieces active castling ep halfmove fullmove
            parts = line.split()
            if len(parts) >= 4:
                fen = ' '.join(parts[:4])
                # Add halfmove/fullmove if missing
                if len(parts) >= 6 and parts[4].isdigit():
                    fen = ' '.join(parts[:6])
                else:
                    fen += ' 0 1'
                fens.append(fen)
    return fens

def random_opening(rng, book_fens=None, random_plies=8):
    """
    Get a starting position. Uses book if available, otherwise plays
    random_plies random moves from startpos for variety.
    """
    if book_fens:
        return chess.Board(rng.choice(book_fens))

    board = chess.Board()
    for _ in range(random_plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    return board

# =============================================================================
# Self-play game
# =============================================================================

def play_game(engine, depth, movetime, nodes, rng, book_fens, random_plies,
              skip_plies, resign_cp, resign_count, draw_adjudicate_cp,
              draw_adjudicate_count, max_plies):
    """
    Play one self-play game. Returns list of (features, stm, eval_cp) tuples
    and the game result (1.0 = white win, 0.5 = draw, 0.0 = black win).
    """
    board = random_opening(rng, book_fens, random_plies)
    engine.new_game()

    positions = []  # (features, stm, eval_white_cp)
    resign_streak = 0
    draw_streak = 0
    ply = 0

    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        best_move, score_cp, score_mate = engine.search(
            board, depth=depth, movetime=movetime, nodes=nodes
        )

        if best_move is None or best_move == "(none)":
            break

        # Handle mate scores
        if score_mate is not None:
            if score_mate > 0:
                score_cp = 10000  # Winning mate
            else:
                score_cp = -10000  # Losing mate

        # Convert to white's perspective
        eval_white_cp = score_cp if board.turn == chess.WHITE else -score_cp

        # Record position (skip early plies and checks for training quality)
        if ply >= skip_plies and not board.is_check():
            features = board_to_features(board)
            stm = 0 if board.turn == chess.WHITE else 1
            positions.append((features, stm, eval_white_cp))

        # Resign adjudication
        if resign_cp > 0:
            if score_cp < -resign_cp:
                resign_streak += 1
            else:
                resign_streak = 0
            if resign_streak >= resign_count:
                # Side to move is losing
                result = 0.0 if board.turn == chess.WHITE else 1.0
                return positions, result

        # Draw adjudication
        if draw_adjudicate_cp > 0 and ply >= 40:
            if abs(score_cp) < draw_adjudicate_cp:
                draw_streak += 1
            else:
                draw_streak = 0
            if draw_streak >= draw_adjudicate_count:
                return positions, 0.5

        # Make the move
        try:
            move = chess.Move.from_uci(best_move)
            if move not in board.legal_moves:
                break
            board.push(move)
        except Exception:
            break

        ply += 1

    # Game ended naturally
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        result = 0.5
    elif outcome.winner == chess.WHITE:
        result = 1.0
    else:
        result = 0.0

    return positions, result

# =============================================================================
# Binary I/O
# =============================================================================

def write_positions_bin(all_positions, filename):
    """Write positions to binary file (compatible with train_nnue.py)."""
    with open(filename, 'wb') as f:
        f.write(struct.pack('<I', len(all_positions)))
        for features, stm, result, eval_val in all_positions:
            f.write(struct.pack('<H', len(features)))
            for feat in features:
                f.write(struct.pack('<H', feat))
            f.write(struct.pack('B', stm))
            f.write(struct.pack('<ff', result, eval_val))

def merge_bin_files(source, dest):
    """Append positions from source into dest."""
    with open(source, 'rb') as f:
        src_count = struct.unpack('<I', f.read(4))[0]
        src_data = f.read()

    with open(dest, 'r+b') as f:
        dst_count = struct.unpack('<I', f.read(4))[0]
        new_count = dst_count + src_count
        f.seek(0)
        f.write(struct.pack('<I', new_count))
        f.seek(0, 2)  # End of file
        f.write(src_data)

    print(f"  Merged: {dest} now has {new_count:,} positions ({dst_count:,} + {src_count:,} new)")

# =============================================================================
# Progress tracker with smart ETA
# =============================================================================

class ProgressTracker:
    """Thread-safe progress tracker with rolling-average ETA."""

    def __init__(self, total_games):
        self.total_games = total_games
        self.completed = 0
        self.total_positions = 0
        self.results = {'white': 0, 'draw': 0, 'black': 0}
        self.start_time = time.time()
        self._lock = threading.Lock()
        # Rolling window of recent game durations for smooth ETA
        self._recent_times = deque(maxlen=50)
        self._last_game_time = self.start_time

    def game_done(self, num_positions, result):
        """Record a completed game and print progress."""
        now = time.time()
        with self._lock:
            game_dur = now - self._last_game_time
            # Only track per-game time for single-worker or approximate for multi
            self._recent_times.append(game_dur)
            self._last_game_time = now

            self.completed += 1
            self.total_positions += num_positions
            if result == 1.0:
                self.results['white'] += 1
            elif result == 0.0:
                self.results['black'] += 1
            else:
                self.results['draw'] += 1

            # Print progress every 10 games or at milestones
            if self.completed % 10 == 0 or self.completed == self.total_games:
                self._print_progress(now)

    def _print_progress(self, now):
        """Print a single progress line with ETA (must be called under lock)."""
        elapsed = now - self.start_time
        remaining = self.total_games - self.completed
        pct = self.completed / self.total_games * 100

        # ETA: blend overall average with rolling recent average
        overall_rate = self.completed / max(elapsed, 0.001)  # games/sec

        if len(self._recent_times) >= 5:
            # Use recent rolling average (more responsive)
            recent_avg = sum(self._recent_times) / len(self._recent_times)
            recent_rate = 1.0 / max(recent_avg, 0.001)
            # Blend: 70% recent, 30% overall (adapts to speed changes)
            blended_rate = 0.7 * recent_rate + 0.3 * overall_rate
        else:
            blended_rate = overall_rate

        eta_secs = remaining / max(blended_rate, 0.001)

        # Format ETA nicely
        if eta_secs < 60:
            eta_str = f"{eta_secs:.0f}s"
        elif eta_secs < 3600:
            eta_str = f"{eta_secs/60:.1f}m"
        else:
            h = int(eta_secs // 3600)
            m = int((eta_secs % 3600) // 60)
            eta_str = f"{h}h{m:02d}m"

        elapsed_str = f"{elapsed:.0f}s" if elapsed < 60 else f"{elapsed/60:.1f}m"

        w, d, b = self.results['white'], self.results['draw'], self.results['black']
        pos_per_game = self.total_positions / max(self.completed, 1)

        print(f"\r  [{self.completed:,}/{self.total_games:,}] {pct:.0f}%  "
              f"{w}W/{d}D/{b}B  "
              f"{self.total_positions:,} pos ({pos_per_game:.0f}/game)  "
              f"elapsed {elapsed_str}  ETA {eta_str}  "
              f"({blended_rate:.1f} games/s)   ", flush=True)

    def get_results(self):
        """Return final tallies."""
        with self._lock:
            return self.total_positions, dict(self.results)


# Global tracker set by main(), used by workers
_progress_tracker = None


# =============================================================================
# Worker (for parallel games)
# =============================================================================

def worker_play_games(worker_id, engine_path, uci_options, num_games, args, seed):
    """Worker function: creates engine, plays N games, returns positions."""
    rng = random.Random(seed)
    engine = None
    all_positions = []
    results = {'white': 0, 'draw': 0, 'black': 0}

    try:
        if shutdown_event.is_set():
            return all_positions, results

        engine = UCIEngine(engine_path, uci_options)
        book_fens = None
        if args.opening_book and os.path.exists(args.opening_book):
            book_fens = load_opening_book(args.opening_book)

        for g in range(num_games):
            if shutdown_event.is_set():
                print(f"  [Worker {worker_id}] Stopping early ({g}/{num_games} games completed)")
                break

            try:
                positions, result = play_game(
                    engine=engine,
                    depth=args.depth,
                    movetime=args.movetime,
                    nodes=args.nodes,
                    rng=rng,
                    book_fens=book_fens,
                    random_plies=args.random_plies,
                    skip_plies=args.skip_plies,
                    resign_cp=args.resign_cp,
                    resign_count=args.resign_count,
                    draw_adjudicate_cp=args.draw_adjudicate_cp,
                    draw_adjudicate_count=args.draw_adjudicate_count,
                    max_plies=args.max_plies,
                )
            except (InterruptedError, RuntimeError, TimeoutError, KeyboardInterrupt, OSError):
                # Engine died or shutdown mid-game — keep what we have
                print(f"  [Worker {worker_id}] Stopping after {g} games (engine interrupted)")
                break

            # Assign game result to all positions
            game_positions = []
            for features, stm, eval_cp in positions:
                eval_pawns = eval_cp / 100.0
                game_positions.append((features, stm, result, eval_pawns))
            all_positions.extend(game_positions)

            if result == 1.0:
                results['white'] += 1
            elif result == 0.0:
                results['black'] += 1
            else:
                results['draw'] += 1

            # Report to global progress tracker
            if _progress_tracker is not None:
                _progress_tracker.game_done(len(game_positions), result)

    except (InterruptedError, RuntimeError, TimeoutError, OSError, KeyboardInterrupt) as e:
        # Engine failed to start or died during init — return what we have
        print(f"  [Worker {worker_id}] Engine error: {e}")
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass

    return all_positions, results

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate self-play training data for NNUE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py -3.10 generate_selfplay.py --engine engine.exe --games 500
  py -3.10 generate_selfplay.py --engine engine.exe --games 1000 --depth 6 --workers 4
  py -3.10 generate_selfplay.py --engine engine.exe --games 500 --merge assets/training_data.bin
        """
    )

    # Required
    parser.add_argument('--engine', required=True, help='Path to UCI engine executable')

    # Search settings (pick one)
    search = parser.add_mutually_exclusive_group()
    search.add_argument('--depth', type=int, default=8, help='Search depth per move (default: 8)')
    search.add_argument('--movetime', type=int, help='Search time in ms per move')
    search.add_argument('--nodes', type=int, help='Search nodes per move')

    # Game settings
    parser.add_argument('--games', type=int, default=500, help='Number of games to play (default: 500)')
    parser.add_argument('--workers', type=int, default=1, help='Parallel engine instances (default: 1)')
    parser.add_argument('--opening-book', type=str, help='EPD/FEN opening book file')
    parser.add_argument('--random-plies', type=int, default=8,
                        help='Random opening plies if no book (default: 8)')
    parser.add_argument('--skip-plies', type=int, default=16,
                        help='Skip first N plies for recording (default: 16)')
    parser.add_argument('--max-plies', type=int, default=400,
                        help='Max plies per game before adjudicating draw (default: 400)')

    # Adjudication
    parser.add_argument('--resign-cp', type=int, default=800,
                        help='Resign threshold in centipawns (default: 800, 0=disabled)')
    parser.add_argument('--resign-count', type=int, default=5,
                        help='Consecutive moves below resign threshold (default: 5)')
    parser.add_argument('--draw-adjudicate-cp', type=int, default=10,
                        help='Draw adjudication threshold in cp (default: 10, 0=disabled)')
    parser.add_argument('--draw-adjudicate-count', type=int, default=8,
                        help='Consecutive moves within draw threshold (default: 8)')

    # UCI options
    parser.add_argument('--uci-option', nargs=2, action='append', metavar=('NAME', 'VALUE'),
                        help='Set UCI option (repeatable, e.g. --uci-option Hash 128)')

    # Output
    parser.add_argument('--output', type=str, default='assets/selfplay_v1.bin',
                        help='Output file (default: assets/selfplay_v1.bin)')
    parser.add_argument('--merge', type=str, metavar='FILE',
                        help='After generating, merge into this file')

    parser.add_argument('--seed', type=int, default=None, help='Random seed')

    args = parser.parse_args()

    # Parse UCI options
    uci_options = {}
    if args.uci_option:
        for name, value in args.uci_option:
            uci_options[name] = value

    seed = args.seed if args.seed is not None else random.randint(0, 2**31)

    # Print config
    print("=" * 60)
    print("  NNUE Self-Play Data Generator")
    print("=" * 60)
    print(f"  Engine:            {args.engine}")
    print(f"  Games:             {args.games:,}")
    print(f"  Workers:           {args.workers}")
    if args.depth:
        print(f"  Search:            depth {args.depth}")
    elif args.movetime:
        print(f"  Search:            {args.movetime}ms/move")
    elif args.nodes:
        print(f"  Search:            {args.nodes:,} nodes/move")
    print(f"  Skip plies:        {args.skip_plies}")
    print(f"  Resign:            {args.resign_cp}cp x {args.resign_count} moves")
    print(f"  Draw adjudicate:   {args.draw_adjudicate_cp}cp x {args.draw_adjudicate_count} moves")
    if args.opening_book:
        print(f"  Opening book:      {args.opening_book}")
    else:
        print(f"  Random openings:   {args.random_plies} random plies")
    print(f"  Output:            {args.output}")
    if args.merge:
        print(f"  Merge into:        {args.merge}")
    if uci_options:
        print(f"  UCI options:       {uci_options}")
    print(f"  Seed:              {seed}")
    print("=" * 60)

    start_time = time.time()

    # Set up global progress tracker
    global _progress_tracker
    _progress_tracker = ProgressTracker(args.games)

    # Distribute games across workers
    games_per_worker = [args.games // args.workers] * args.workers
    for i in range(args.games % args.workers):
        games_per_worker[i] += 1

    all_positions = []
    total_results = {'white': 0, 'draw': 0, 'black': 0}

    if args.workers == 1:
        # Single-threaded (simpler, better for debugging)
        try:
            positions, results = worker_play_games(
                0, args.engine, uci_options, args.games, args, seed
            )
            all_positions = positions
            total_results = results
        except KeyboardInterrupt:
            shutdown_event.set()
    else:
        # Multi-threaded
        executor = ThreadPoolExecutor(max_workers=args.workers)
        futures = []
        for w in range(args.workers):
            f = executor.submit(
                worker_play_games, w, args.engine, uci_options,
                games_per_worker[w], args, seed + w
            )
            futures.append(f)

        try:
            for f in as_completed(futures):
                try:
                    positions, results = f.result()
                    all_positions.extend(positions)
                    for k in total_results:
                        total_results[k] += results[k]
                except Exception as e:
                    print(f"  Worker error (data from other workers preserved): {e}")
        except KeyboardInterrupt:
            # Signal already handled, just collect whatever finished
            shutdown_event.set()
            print("\n  Collecting completed results...")
            for f in futures:
                try:
                    if f.done():
                        positions, results = f.result()
                        all_positions.extend(positions)
                        for k in total_results:
                            total_results[k] += results[k]
                except Exception:
                    pass
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    elapsed = time.time() - start_time
    print()  # newline after \r progress line

    # Statistics
    total_games = total_results['white'] + total_results['draw'] + total_results['black']
    draw_rate = total_results['draw'] / max(total_games, 1) * 100
    pos_per_game = len(all_positions) / max(total_games, 1)

    was_interrupted = shutdown_event.is_set()

    print(f"\n{'='*60}")
    if was_interrupted:
        print(f"  Generation stopped early (Ctrl+C)")
    else:
        print(f"  Generation complete!")
    print(f"{'='*60}")
    print(f"  Games:         {total_games:,} ({total_results['white']}W / {total_results['draw']}D / {total_results['black']}B)")
    print(f"  Draw rate:     {draw_rate:.1f}%")
    print(f"  Positions:     {len(all_positions):,} ({pos_per_game:.1f}/game)")
    print(f"  Time:          {elapsed:.1f}s ({elapsed/max(total_games,1):.2f}s/game)")
    print(f"  Throughput:    {len(all_positions)/max(elapsed,1):.0f} pos/sec")

    # Compute eval distribution
    evals = [abs(p[3]) for p in all_positions]
    if evals:
        avg_eval = sum(evals) / len(evals)
        extreme = sum(1 for e in evals if e > 10.0)
        print(f"  Avg |eval|:    {avg_eval:.2f} pawns")
        print(f"  Extreme evals: {extreme:,} ({extreme/len(evals)*100:.1f}%)")

    print(f"{'='*60}")

    # Write output
    if not all_positions:
        print("\nNo positions collected — nothing to save.")
        return

    print(f"\nWriting {len(all_positions):,} positions to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    write_positions_bin(all_positions, args.output)
    file_size = os.path.getsize(args.output)
    print(f"  Written: {file_size / 1024 / 1024:.1f} MB")

    # Optional merge
    if args.merge:
        if os.path.exists(args.merge):
            print(f"\nMerging into {args.merge}...")
            merge_bin_files(args.output, args.merge)
        else:
            print(f"\nWarning: {args.merge} not found, skipping merge.")

    print("\nDone! ✓")

if __name__ == '__main__':
    main()
