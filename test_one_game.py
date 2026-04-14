#!/usr/bin/env python3
"""
Verbose single-game test — shows every move as it happens.
Run from project root:
    py -3.10 test_one_game.py
"""
import subprocess, time, threading, queue, sys, os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--engine', default=r"x64\Release\ChessEngine.exe", help="Path to engine binary")
args = parser.parse_args()
ENGINE = args.engine
success = True  # Track overall success for exit code

# ---------- minimal engine wrapper ----------
def reader_thread(proc, q):
    try:
        for line in proc.stdout:
            q.put(line)
    except Exception:
        pass
    q.put(None)

def send(proc, cmd):
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()

def wait_for(q, token, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            line = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if line is None:
            raise RuntimeError("Engine died")
        line = line.strip()
        if line.startswith(token):
            return line
    raise TimeoutError(f"No '{token}' in {timeout}s")

def search(proc, q, fen, depth=8):
    send(proc, f"position fen {fen}")
    send(proc, f"go depth {depth}")
    score_cp = 0
    score_mate = None
    while True:
        try:
            line = q.get(timeout=30)
        except queue.Empty:
            raise TimeoutError("No bestmove in 30s")
        if line is None:
            raise RuntimeError("Engine died during search")
        line = line.strip()
        if line.startswith("info") and " score " in line:
            parts = line.split()
            try:
                si = parts.index("score")
                if parts[si+1] == "cp":
                    score_cp = int(parts[si+2])
                    score_mate = None
                elif parts[si+1] == "mate":
                    score_mate = int(parts[si+2])
            except Exception:
                pass
        if line.startswith("bestmove"):
            move = line.split()[1]
            return move, score_cp, score_mate

# ---------- main ----------
# WARNING (AUDIT 11.13): Auto-installing pip packages in a test script is a
# security risk in CI environments (arbitrary code execution, unversioned
# dependency).  Prefer using a requirements.txt or pre-installed venv instead
# of auto-installing at runtime.
try:
    import chess
except ImportError:
    print("Installing python-chess...")
    import subprocess as _sp
    _sp.check_call([sys.executable, "-m", "pip", "install", "python-chess", "-q"])
    import chess

print(f"Starting engine: {ENGINE}")
proc = subprocess.Popen(
    [ENGINE, "--uci"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, bufsize=1,
    cwd="."  # use project root so engine can find assets/
)

q = queue.Queue()
t = threading.Thread(target=reader_thread, args=(proc, q), daemon=True)
t.start()

# F5.1: Ensure engine process is always cleaned up
import atexit
def _cleanup_engine():
    if proc.poll() is None:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass
atexit.register(_cleanup_engine)

send(proc, "uci")
wait_for(q, "uciok")
print("✓ uciok")
send(proc, "isready")
wait_for(q, "readyok")
print("✓ readyok")

# Random opening
import random
board = chess.Board()
rng = random.Random(42)
for i in range(8):
    moves = list(board.legal_moves)
    if not moves:
        break
    board.push(rng.choice(moves))

print(f"\nOpening FEN: {board.fen()}")
print(f"Starting game (depth 8)...\n")

send(proc, "ucinewgame")
send(proc, "isready")
wait_for(q, "readyok")

ply = 0
game_start = time.time()
resign_streak = 0
resign_sign = 0

while not board.is_game_over(claim_draw=True) and ply < 500:
    t0 = time.time()
    move_uci, score_cp, score_mate = search(proc, q, board.fen(), depth=8)
    elapsed = time.time() - t0

    if score_mate is not None:
        score_str = f"mate {score_mate}"
        if score_mate > 0:
            score_cp = 10000
        else:
            score_cp = -10000
    else:
        score_str = f"{score_cp}cp"

    eval_white = score_cp if board.turn == chess.WHITE else -score_cp
    side = "W" if board.turn == chess.WHITE else "B"
    
    print(f"  Ply {ply:3d} [{side}] {move_uci:6s}  eval={score_str:>10s}  ({elapsed:.1f}s)  total={time.time()-game_start:.0f}s")

    if move_uci == "(none)" or move_uci == "0000":
        print("  → No legal move, stopping")
        success = False
        break

    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            print(f"  → ILLEGAL MOVE: {move_uci}")
            print(f"    FEN: {board.fen()}")
            print(f"    Legal moves: {[m.uci() for m in board.legal_moves]}")
            success = False
            break
        search_turn = board.turn
        board.push(move)
    except Exception as e:
        print(f"  → Move parse error: {e}")
        success = False
        break

    # Check resign (800cp x 5) — convert to White perspective for consistency
    white_eval = score_cp if search_turn == chess.WHITE else -score_cp
    if white_eval < -800:
        if resign_sign == -1:
            resign_streak += 1
        else:
            resign_sign = -1
            resign_streak = 1
    elif white_eval > 800:
        if resign_sign == 1:
            resign_streak += 1
        else:
            resign_sign = 1
            resign_streak = 1
    else:
        resign_streak = 0
        resign_sign = 0
    if resign_streak >= 5:
        winner = "White" if resign_sign == 1 else "Black"
        print(f"\n  → Resign adjudication: {winner} wins")
        break

    ply += 1

total = time.time() - game_start
print(f"\n{'='*50}")
print(f"Game finished: {ply} plies in {total:.1f}s ({total/max(ply,1):.2f}s/ply)")

outcome = board.outcome(claim_draw=True)
if outcome:
    print(f"Result: {outcome.result()}")
else:
    print(f"Result: adjudicated or stopped")

send(proc, "quit")
try:
    proc.wait(timeout=5)
except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait()
print("Done ✓")
sys.exit(0 if success else 1)
