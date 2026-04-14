#!/usr/bin/env python3
"""
test_engine_uci.py — Diagnose UCI communication issues.

Run from project root:
    py -3.10 test_engine_uci.py

Or specify a custom engine path:
    py -3.10 test_engine_uci.py --engine x64\Release\ChessEngine.exe
"""

import argparse
import os
import queue
import subprocess
import sys
import threading
import time

def main():
    parser = argparse.ArgumentParser()
    # AUDIT 11.17: Default path is configurable via ENGINE_PATH env var or --engine arg
    default_engine = os.environ.get("ENGINE_PATH", r"x64\Release\ChessEngine.exe")
    parser.add_argument("--engine", default=default_engine)
    args = parser.parse_args()

    engine_path = os.path.abspath(args.engine)

    # Detect project root (two levels up from x64/Release/)
    engine_dir = os.path.dirname(engine_path)
    project_root = os.path.dirname(os.path.dirname(engine_dir))
    if not os.path.isdir(os.path.join(project_root, "assets")):
        project_root = engine_dir if os.path.isdir(os.path.join(engine_dir, "assets")) else os.getcwd()

    print(f"Engine:      {engine_path}")
    print(f"Working dir: {project_root}")
    print(f"Exists:      {os.path.isfile(engine_path)}")
    print(f"Assets dir:  {os.path.isdir(os.path.join(project_root, 'assets'))}")
    print()

    # Start engine
    print("Starting engine process...")
    t0 = time.time()
    proc = subprocess.Popen(
        [engine_path, "--uci"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=project_root,
    )
    print(f"Engine PID: {proc.pid}  (started in {time.time()-t0:.2f}s)")

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

    print()

    # Shared line queue + reader thread (same design as generate_selfplay.py)
    q = queue.Queue()

    def reader():
        try:
            while True:
                line = proc.stdout.readline()
                q.put(line)
                if not line:  # EOF
                    return
        except Exception as e:
            q.put(f"[READER ERROR: {e}]")
            q.put("")

    # Also capture stderr
    stderr_lines = []
    def stderr_reader():
        try:
            while True:
                line = proc.stderr.readline()
                if not line:
                    return
                stderr_lines.append(line.strip())
        except:
            pass

    threading.Thread(target=reader, daemon=True).start()
    threading.Thread(target=stderr_reader, daemon=True).start()

    def send(cmd):
        print(f"  >>> {cmd}")
        try:
            proc.stdin.write(cmd + "\n")
            proc.stdin.flush()
        except (OSError, BrokenPipeError) as e:
            print(f"  !!! SEND FAILED: {e}")
            return False
        return True

    def wait_for(token, timeout=10):
        deadline = time.time() + timeout
        lines_seen = []
        while time.time() < deadline:
            try:
                line = q.get(timeout=0.5)
            except queue.Empty:
                rc = proc.poll()
                if rc is not None:
                    print(f"  !!! ENGINE DIED (exit code {rc})")
                    if stderr_lines:
                        print(f"  !!! stderr: {'; '.join(stderr_lines[-5:])}")
                    return None
                continue

            raw = line.rstrip('\n\r')
            if raw:
                print(f"  <<< {raw}")
                lines_seen.append(raw)

            if not line:  # EOF
                rc = proc.poll()
                print(f"  !!! ENGINE EOF (exit code {rc})")
                if stderr_lines:
                    print(f"  !!! stderr: {'; '.join(stderr_lines[-5:])}")
                return None

            if raw.startswith(token):
                return raw

        print(f"  !!! TIMEOUT after {timeout}s waiting for '{token}'")
        print(f"  !!! Lines seen so far: {lines_seen}")
        rc = proc.poll()
        print(f"  !!! Engine alive: {rc is None} (poll={rc})")
        return None

    results = {}

    # ─── Test 1: UCI handshake ───
    print("=" * 60)
    print("TEST 1: uci → uciok")
    print("=" * 60)
    send("uci")
    r = wait_for("uciok", timeout=15)
    results["uci"] = bool(r)
    print(f"  Result: {'✅ PASS' if r else '❌ FAIL'}\n")

    if not r:
        print("Engine didn't respond to 'uci'. Cannot continue.")
        proc.kill()
        return

    # ─── Test 2: isready ───
    print("=" * 60)
    print("TEST 2: isready → readyok")
    print("=" * 60)
    send("isready")
    r = wait_for("readyok", timeout=10)
    results["isready"] = bool(r)
    print(f"  Result: {'✅ PASS' if r else '❌ FAIL'}\n")

    # ─── Test 3: ucinewgame + isready ───
    print("=" * 60)
    print("TEST 3: ucinewgame + isready → readyok")
    print("=" * 60)
    send("ucinewgame")
    send("isready")
    r = wait_for("readyok", timeout=10)
    results["newgame"] = bool(r)
    print(f"  Result: {'✅ PASS' if r else '❌ FAIL'}\n")

    # ─── Test 4: search from startpos, depth 4 ───
    print("=" * 60)
    print("TEST 4: search startpos depth 4")
    print("=" * 60)
    send("position startpos")
    send("go depth 4")
    t1 = time.time()
    r = wait_for("bestmove", timeout=30)
    dt = time.time() - t1
    results["search_startpos"] = bool(r)
    print(f"  Time: {dt:.2f}s")
    print(f"  Result: {'✅ PASS' if r else '❌ FAIL'}\n")

    # ─── Test 5: search from simple FEN, depth 8 ───
    print("=" * 60)
    print("TEST 5: search FEN depth 8 (1. e4 position)")
    print("=" * 60)
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    print(f"  FEN: {fen}")
    send("ucinewgame")
    send("isready")
    r = wait_for("readyok", timeout=10)
    if r:
        send(f"position fen {fen}")
        send("go depth 8")
        t1 = time.time()
        r = wait_for("bestmove", timeout=30)
        dt = time.time() - t1
        results["search_fen"] = bool(r)
        print(f"  Time: {dt:.2f}s")
    else:
        results["search_fen"] = False
    print(f"  Result: {'✅ PASS' if results['search_fen'] else '❌ FAIL'}\n")

    # ─── Test 6: search from random opening (simulates self-play) ───
    print("=" * 60)
    print("TEST 6: search random opening (8 random plies), depth 8")
    print("=" * 60)
    try:
        import chess
        import random
        board = chess.Board()
        rng = random.Random(42)
        for _ in range(8):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        fen = board.fen()
    except ImportError:
        fen = "rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2"
        print("  (python-chess not available, using hardcoded FEN)")

    print(f"  FEN: {fen}")
    send("ucinewgame")
    send("isready")
    r = wait_for("readyok", timeout=10)
    if r:
        send(f"position fen {fen}")
        send("go depth 8")
        t1 = time.time()
        r = wait_for("bestmove", timeout=30)
        dt = time.time() - t1
        results["search_random"] = bool(r)
        print(f"  Time: {dt:.2f}s")
    else:
        results["search_random"] = False
    print(f"  Result: {'✅ PASS' if results['search_random'] else '❌ FAIL'}\n")

    # ─── Test 7: two consecutive searches (simulates a game) ───
    print("=" * 60)
    print("TEST 7: two consecutive searches (simulates game turns)")
    print("=" * 60)
    send("ucinewgame")
    send("isready")
    r = wait_for("readyok", timeout=10)
    passed = True
    if r:
        # Move 1
        send("position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        send("go depth 6")
        t1 = time.time()
        r = wait_for("bestmove", timeout=30)
        if r:
            move1 = r.split()[1]
            print(f"  Move 1: {move1} ({time.time()-t1:.2f}s)")

            # Move 2 (after making move 1)
            # Use position startpos moves <move1> for simplicity
            send(f"position startpos moves {move1}")
            send("go depth 6")
            t1 = time.time()
            r = wait_for("bestmove", timeout=30)
            if r:
                move2 = r.split()[1]
                print(f"  Move 2: {move2} ({time.time()-t1:.2f}s)")
            else:
                passed = False
        else:
            passed = False
    else:
        passed = False
    results["consecutive"] = passed
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}\n")

    # ─── Cleanup ───
    send("quit")
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()

    # ─── Summary ───
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_pass = False

    if stderr_lines:
        print(f"\n  Stderr output ({len(stderr_lines)} lines):")
        for line in stderr_lines[:10]:
            print(f"    {line}")

    print()
    if all_pass:
        print("All tests passed! Engine UCI communication is working correctly.")
        print("If self-play still fails, the issue is likely in the parallel/threading layer.")
    else:
        print("Some tests FAILED. The output above shows exactly where communication broke down.")
        print("Share this output for further diagnosis.")

    if not all_pass:
        sys.exit(1)

if __name__ == "__main__":
    main()
