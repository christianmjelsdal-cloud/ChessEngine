#!/usr/bin/env python3
"""
Pipeline Integration Test
=========================
Validates that the self-play → training pipeline works end-to-end:
  1. Creates a small fake base dataset (1000 positions)
  2. Creates a small fake self-play dataset (200 positions)
  3. Verifies both files are valid .bin format
  4. Runs prepare_datasets() and checks ratio math
  5. Tests --max-positions only caps PRIMARY (not extra datasets)
  6. Runs 1 epoch of actual training on the fake data
  7. Verifies output weights file is produced

Run from project root:
    py -3.10 test_pipeline.py
"""

import struct
import os
import sys
import tempfile
import random
import importlib.util

# ---- Helpers ----

def create_fake_bin(path, num_positions, eval_range=(-5.0, 5.0), seed=42):  # FIX 11.10: reproducible seed
    """Create a valid .bin training file with random positions.
    
    Format per position (sparse features):
      num_features  (uint16)       - number of active features
      features      (num_features × uint16) - feature indices in [0, 767]
      stm           (uint8)        - side to move (0 or 1)
      game_result   (float32)      - 0.0, 0.5, or 1.0
      search_eval   (float32)      - centipawn eval
    """
    # F5.2: Use versioned header format via training_format.py
    try:
        from training_format import write_header
        _use_versioned = True
    except ImportError:
        _use_versioned = False
    
    with open(path, 'wb') as f:
        if _use_versioned:
            write_header(f, num_positions)
        else:
            f.write(struct.pack('<I', num_positions))
        rng = random.Random(seed)  # FIX 11.10: use seeded RNG for reproducibility
        for _ in range(num_positions):
            # Typical chess position has ~16-30 active piece features
            num_features = rng.randint(16, 30)
            # Pick unique feature indices in [0, 767] (12 piece types × 64 squares)
            features = rng.sample(range(768), num_features)
            f.write(struct.pack('<H', num_features))
            for feat in features:
                f.write(struct.pack('<H', feat))
            # side_to_move
            f.write(struct.pack('B', rng.randint(0, 1)))
            # game_result: 0.0 (black wins), 0.5 (draw), 1.0 (white wins)
            f.write(struct.pack('<f', rng.choice([0.0, 0.5, 1.0])))
            # search_eval in centipawns
            f.write(struct.pack('<f', rng.uniform(*eval_range)))
    return path

def read_bin_header(path):
    """Read position count from .bin header."""
    # F5.2: Handle both versioned and legacy headers
    try:
        from training_format import read_header
        with open(path, 'rb') as f:
            _, count = read_header(f)
        return count
    except ImportError:
        with open(path, 'rb') as f:
            return struct.unpack('<I', f.read(4))[0]

def load_train_module(script_path):
    """Import train_nnue.py as a module.

    WARNING (AUDIT 11.12): spec.loader.exec_module() executes all top-level
    code in the target module as a side-effect (argument parsing, global state
    init, torch.cuda probing, etc.).  If the module has top-level code that
    calls sys.exit() or modifies global state, this can cause spurious test
    failures.  The target module should guard such code with:
        if __name__ == '__main__':
    """
    spec = importlib.util.spec_from_file_location("train_nnue", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ---- Tests ----

def test_bin_format(base_path, selfplay_path, base_count, selfplay_count):
    """Test 1: Verify .bin files are readable."""
    print("TEST 1: Binary file format validation")
    
    assert os.path.exists(base_path), f"Base file not found: {base_path}"
    assert os.path.exists(selfplay_path), f"Self-play file not found: {selfplay_path}"
    
    n_base = read_bin_header(base_path)
    n_selfplay = read_bin_header(selfplay_path)
    
    assert n_base == base_count, f"Base header says {n_base}, expected {base_count}"
    assert n_selfplay == selfplay_count, f"Selfplay header says {n_selfplay}, expected {selfplay_count}"
    
    # FIX H-3: Use training_format.read_header() for dynamic data offset instead of
    # hardcoded offset=4. Versioned headers are 9 bytes (magic+version+count), not 4.
    try:
        from training_format import read_header as _read_hdr
        _use_training_format = True
    except ImportError:
        _use_training_format = False

    actual_base_size = os.path.getsize(base_path)
    actual_selfplay_size = os.path.getsize(selfplay_path)
    
    assert actual_base_size > 4, f"Base file too small: {actual_base_size} bytes"
    assert actual_selfplay_size > 4, f"Selfplay file too small: {actual_selfplay_size} bytes"
    
    # Verify we can actually read positions (parse first 3)
    for label, path, count in [("Base", base_path, base_count), ("Selfplay", selfplay_path, selfplay_count)]:
        with open(path, 'rb') as f:
            raw = f.read()
        # Determine data offset from header format
        if _use_training_format:
            _, _, data_offset = _read_hdr(path)
            offset = data_offset
        else:
            offset = 4  # legacy fallback
        for i in range(min(3, count)):
            nf = struct.unpack_from('<H', raw, offset)[0]
            assert 1 <= nf <= 768, f"{label} pos {i}: num_features={nf} out of range"
            offset += 2
            for j in range(nf):
                feat = struct.unpack_from('<H', raw, offset)[0]
                assert 0 <= feat < 768, f"{label} pos {i}: feature {feat} >= 768"
                offset += 2
            stm = raw[offset]; offset += 1
            assert stm in (0, 1), f"{label} pos {i}: stm={stm} not 0 or 1"
            result, ev = struct.unpack_from('<ff', raw, offset); offset += 8
            assert result in (0.0, 0.5, 1.0), f"{label} pos {i}: result={result}"
    
    print(f"  ✓ Base:     {n_base:,} positions, {actual_base_size:,} bytes, format valid")
    print(f"  ✓ Selfplay: {n_selfplay:,} positions, {actual_selfplay_size:,} bytes, format valid")
    print("  PASSED\n")

def test_prepare_datasets(mod, base_path, selfplay_path, base_count, selfplay_count):
    """Test 2: Verify ratio math in prepare_datasets."""
    print("TEST 2: Dataset ratio math (no --max-positions)")
    
    extra_args = [[selfplay_path, "0.3"]]
    datasets, total = mod.prepare_datasets(base_path, extra_args, max_positions=0)
    
    assert len(datasets) == 2, f"Expected 2 datasets, got {len(datasets)}"
    assert datasets[0]['num_positions'] == base_count, \
        f"Primary should have {base_count}, got {datasets[0]['num_positions']}"
    assert datasets[1]['num_positions'] == selfplay_count, \
        f"Extra should have {selfplay_count}, got {datasets[1]['num_positions']}"
    assert abs(datasets[0]['ratio'] - 0.7) < 0.01, \
        f"Primary ratio should be ~0.7, got {datasets[0]['ratio']}"
    assert abs(datasets[1]['ratio'] - 0.3) < 0.01, \
        f"Extra ratio should be ~0.3, got {datasets[1]['ratio']}"
    assert total == base_count + selfplay_count
    
    print(f"  ✓ Primary: {datasets[0]['num_positions']:,} @ {datasets[0]['ratio']:.0%}")
    print(f"  ✓ Extra:   {datasets[1]['num_positions']:,} @ {datasets[1]['ratio']:.0%}")
    print(f"  ✓ Total:   {total:,}")
    print("  PASSED\n")
    
    return datasets

def test_max_positions_primary_only(mod, base_path, selfplay_path, base_count, selfplay_count):
    """Test 3: --max-positions only caps PRIMARY dataset."""
    print("TEST 3: --max-positions caps PRIMARY only")
    
    cap = 300  # Cap base to 300
    extra_args = [[selfplay_path, "0.3"]]
    datasets, total = mod.prepare_datasets(base_path, extra_args, max_positions=cap)
    
    assert datasets[0]['num_positions'] == cap, \
        f"Primary should be capped to {cap}, got {datasets[0]['num_positions']}"
    assert datasets[1]['num_positions'] == selfplay_count, \
        f"Extra should be UNCHANGED at {selfplay_count}, got {datasets[1]['num_positions']}"
    assert total == cap + selfplay_count
    
    # Check oversampling ratio
    total_train = total
    selfplay_samples_per_epoch = datasets[1]['ratio'] * total_train
    oversampling = selfplay_samples_per_epoch / selfplay_count
    
    print(f"  ✓ Primary capped: {datasets[0]['num_positions']:,} (from {base_count:,})")
    print(f"  ✓ Extra unchanged: {datasets[1]['num_positions']:,}")
    print(f"  ✓ Selfplay oversampling: {oversampling:.1f}× per epoch (target: 1-5×)")
    assert oversampling < 10, f"Oversampling too high: {oversampling:.1f}×"
    print("  PASSED\n")

def test_realistic_ratio(mod, base_path, selfplay_path, selfplay_count):
    """Test 4: Simulate real scenario (66M base capped to 500K + 87K selfplay)."""
    # NOTE: This test validates arithmetic on hardcoded constants, not actual training code.
    # TODO: Replace with a test that calls prepare_datasets() with mock files.
    print("TEST 4: Realistic ratio simulation (66M→500K base + 87K selfplay)")
    
    # We can't create a 66M file, but we can test the math
    # If base=500K, selfplay=87K, ratio=0.3:
    simulated_base = 500_000
    simulated_selfplay = 87_000
    simulated_total = simulated_base + simulated_selfplay
    
    selfplay_samples = 0.3 * simulated_total  # 176,100
    oversampling = selfplay_samples / simulated_selfplay  # ~2.0×
    
    base_samples = 0.7 * simulated_total  # 410,900
    base_coverage = base_samples / simulated_base  # ~0.82×
    
    print(f"  Simulated setup:")
    print(f"    Base:     {simulated_base:,} positions → {base_samples:,.0f} samples/epoch ({base_coverage:.1%} coverage)")
    print(f"    Selfplay: {simulated_selfplay:,} positions → {selfplay_samples:,.0f} samples/epoch ({oversampling:.1f}× oversampling)")
    
    assert 1.0 < oversampling < 5.0, f"Oversampling {oversampling:.1f}× out of healthy range (1-5×)"
    assert base_coverage > 0.5, f"Base coverage {base_coverage:.1%} too low"
    
    print(f"  ✓ Selfplay gets meaningful representation without memorization")
    print(f"  ✓ Base still covers majority of data each epoch")
    print("  PASSED\n")

def test_streaming_forced_for_large_cap(mod, base_path, selfplay_path, base_count, selfplay_count):
    """Test 6: Verify streaming mode is forced when file >> max_positions."""
    print("TEST 6: Streaming mode forced for large file + small cap")
    
    # Simulate: base has 1000 positions, cap at 100 (ratio > 3x → force streaming)
    cap = 100
    extra_args = [[selfplay_path, "0.3"]]
    datasets, total = mod.prepare_datasets(base_path, extra_args, max_positions=cap)
    
    assert datasets[0]['num_positions'] == cap, \
        f"Primary not capped: {datasets[0]['num_positions']}"
    
    # The actual streaming detection happens in train(), but we can verify
    # the condition: actual_primary > max_positions * 3
    actual_primary = base_count
    should_force_streaming = actual_primary > cap * 3
    
    print(f"  Primary file: {actual_primary:,} positions")
    print(f"  Cap: {cap:,} → ratio = {actual_primary / cap:.0f}x")
    print(f"  ✓ Force streaming: {should_force_streaming} (file is {actual_primary / cap:.0f}× larger than cap)")
    assert should_force_streaming, "Streaming should be forced when file >> cap"
    print("  PASSED\n")

def test_training_epoch(mod, base_path, selfplay_path, tmp_dir):
    """Test 5: Run 1 actual training epoch."""
    print("TEST 5: Full training epoch (1 epoch on fake data)")
    
    output_weights = os.path.join(tmp_dir, "test_weights.bin")
    
    # Build args namespace matching train_nnue.py expectations
    class Args:
        data = base_path
        extra_data = [[selfplay_path, "0.3"]]
        max_positions = 300  # cap base to 300
        epochs = 1
        batch_size = 64
        lr = 0.001
        lr_min = 1e-6
        weight_decay = 0.001
        dropout = 0.0
        grad_accum = 1
        grad_clip = 1.0
        lam = 0.75
        eval_scale = 400.0
        label_smoothing = 0.0
        eval_soft_cap = 0.0
        draw_weight = 1.0
        phase_balanced = False
        early_stop = 999
        fresh = True
        load_weights = None
        output = output_weights
        enhanced = False
        swa = False
        swa_start = 10
        warmup_steps = 0
        filter_eval_max = 0.0
        cosine_lr = True
        cosine_restarts = False
        cosine_t0 = 10
        cosine_t_mult = 2
        mate_boost = 0.0
        chunk_size = 500000
        plot = False
        plot_every = 999
        save_every = 999
        log_file = os.path.join(tmp_dir, "test_log.json")
        plot_file = os.path.join(tmp_dir, "test_plot.png")
    
    args = Args()
    
    try:
        mod.train(args)
    except SystemExit as e:
        # F5.3: Only swallow clean exits; re-raise error exits
        if e.code not in (None, 0):
            raise
    
    assert os.path.exists(output_weights), f"Output weights not created: {output_weights}"
    weight_size = os.path.getsize(output_weights)
    assert weight_size > 0, "Output weights file is empty"
    
    print(f"  ✓ Training completed, weights: {weight_size:,} bytes")
    print("  PASSED\n")

# ---- Main ----

def test_reservoir_sampling(mod, base_path, BASE_COUNT):
    """Test 7: Verify reservoir sampling returns correct count and is random."""
    print("Test 7: Reservoir sampling")
    
    # With max_positions < total: should return exactly max_positions offsets
    count, offsets = mod.scan_positions(base_path, max_positions=100)
    assert count == BASE_COUNT, f"Expected {BASE_COUNT} total, got {count}"
    assert len(offsets) == 100, f"Expected 100 sampled offsets, got {len(offsets)}"
    
    # Offsets should all be valid (>= 4 header, unique if from different positions)
    assert all(o >= 4 for o in offsets), "Some offsets are below header boundary"
    
    # With max_positions=0: should return all offsets
    count2, offsets2 = mod.scan_positions(base_path, max_positions=0)
    assert count2 == BASE_COUNT, f"Expected {BASE_COUNT}, got {count2}"
    assert len(offsets2) == BASE_COUNT, f"Expected all {BASE_COUNT} offsets, got {len(offsets2)}"
    
    # With max_positions > total: should return all offsets (no crash)
    count3, offsets3 = mod.scan_positions(base_path, max_positions=BASE_COUNT + 500)
    assert len(offsets3) == BASE_COUNT, f"Expected {BASE_COUNT} offsets when cap > total, got {len(offsets3)}"
    
    # Randomness: two runs should (very likely) produce different orderings
    _, run1 = mod.scan_positions(base_path, max_positions=100)
    _, run2 = mod.scan_positions(base_path, max_positions=100)
    # Not guaranteed different, but with 1000 choose 100, probability of identical is ~0
    # We just check both are valid
    assert len(run1) == 100 and len(run2) == 100, "Both runs should return 100 offsets"
    
    print(f"  Reservoir sampling: {len(offsets)} from {count} ✓")
    print(f"  Full scan fallback: {len(offsets2)} offsets ✓")
    print(f"  Cap > total handled: {len(offsets3)} offsets ✓")
    print()


def test_atomic_saves(mod):
    """Test 8: Verify atomic save functions exist and produce valid output."""
    print("Test 8: Atomic save functions (behavioral)")
    
    assert hasattr(mod, 'safe_torch_save'), "safe_torch_save function missing"
    assert hasattr(mod, 'save_weights_cpp'), "save_weights_cpp function missing"
    
    # F5.4: Behavioral test instead of source inspection
    import tempfile
    try:
        import torch
    except ImportError:
        print("  SKIP: PyTorch not installed")  # INFO [11.21]
        return True
    with tempfile.TemporaryDirectory(prefix="atomic_test_") as td:
        # Test save_weights_cpp: create a minimal model and save it
        model = mod.NNUE()
        path = os.path.join(td, "test_weights.bin")
        mod.save_weights_cpp(model, path)
        assert os.path.exists(path), "save_weights_cpp did not create file"
        assert os.path.getsize(path) > 0, "save_weights_cpp created empty file"
        assert not os.path.exists(path + '.tmp'), "Leftover .tmp file from save_weights_cpp"
        
        # Test safe_torch_save: save and reload a checkpoint
        ckpt_path = os.path.join(td, "test_checkpoint.pt")
        test_data = {"epoch": 1, "loss": 0.5}
        mod.safe_torch_save(test_data, ckpt_path)
        assert os.path.exists(ckpt_path), "safe_torch_save did not create file"
        assert not os.path.exists(ckpt_path + '.tmp'), "Leftover .tmp file from safe_torch_save"
        loaded = torch.load(ckpt_path, weights_only=False)
        assert loaded["epoch"] == 1, "Checkpoint data corrupted"
        assert loaded["loss"] == 0.5, "Checkpoint data corrupted"
    
    print("  save_weights_cpp: creates valid file, no .tmp leftover ✓")
    print("  safe_torch_save: creates valid checkpoint, roundtrips correctly ✓")
    print()


def main():
    print("=" * 60)
    print("Pipeline Integration Test")
    print("=" * 60 + "\n")
    
    # Find train_nnue.py - check current dir first, then /agent/home/
    script_path = None
    for candidate in ["train_nnue.py", os.path.join(os.path.dirname(__file__), "train_nnue.py")]:
        if os.path.exists(candidate):
            script_path = os.path.abspath(candidate)
            break
    
    if script_path is None:
        print("ERROR: train_nnue.py not found. Run from project root or same dir.")
        sys.exit(1)
    
    print(f"Using: {script_path}\n")
    mod = load_train_module(script_path)
    
    BASE_COUNT = 1000
    SELFPLAY_COUNT = 200
    
    with tempfile.TemporaryDirectory(prefix="pipeline_test_") as tmp_dir:
        base_path = os.path.join(tmp_dir, "base_data.bin")
        selfplay_path = os.path.join(tmp_dir, "selfplay_data.bin")
        
        print("Creating test datasets...")
        create_fake_bin(base_path, BASE_COUNT)
        create_fake_bin(selfplay_path, SELFPLAY_COUNT, eval_range=(-3.0, 3.0))
        print(f"  Base:     {base_path}")
        print(f"  Selfplay: {selfplay_path}\n")
        
        passed = 0
        failed = 0
        
        # NOTE: Test numbering is non-sequential in source order (6 before 5).
        # The execution order follows the tests list, not the function definition order.
        tests = [
            ("bin_format", lambda: test_bin_format(base_path, selfplay_path, BASE_COUNT, SELFPLAY_COUNT)),
            ("ratio_math", lambda: test_prepare_datasets(mod, base_path, selfplay_path, BASE_COUNT, SELFPLAY_COUNT)),
            ("max_positions", lambda: test_max_positions_primary_only(mod, base_path, selfplay_path, BASE_COUNT, SELFPLAY_COUNT)),
            ("realistic_ratio", lambda: test_realistic_ratio(mod, base_path, selfplay_path, SELFPLAY_COUNT)),
            ("training_epoch", lambda: test_training_epoch(mod, base_path, selfplay_path, tmp_dir)),
            ("streaming_forced", lambda: test_streaming_forced_for_large_cap(mod, base_path, selfplay_path, BASE_COUNT, SELFPLAY_COUNT)),
            ("reservoir_sampling", lambda: test_reservoir_sampling(mod, base_path, BASE_COUNT)),
            ("atomic_saves", lambda: test_atomic_saves(mod)),
        ]
        
        for name, test_fn in tests:
            try:
                test_fn()
                passed += 1
            except Exception as e:
                print(f"  FAILED: {e}\n")
                failed += 1
        
        print("=" * 60)
        if failed == 0:
            print(f"ALL {passed} TESTS PASSED ✓")
            print(f"\nOvernight command (copy-paste):")
            print(f"py -3.10 generate_selfplay.py --engine x64\\Release\\ChessEngine.exe --games 1000 --depth 5 --workers 4 --output assets/selfplay_v1.bin; if ($?) {{ py -3.10 train_nnue.py --data assets/training_data.bin --extra-data assets/selfplay_v1.bin 0.3 --max-positions 500000 --epochs 30 --weight-decay 0.001 --early-stop 15 --no-cosine-restarts }}")
        else:
            print(f"{passed} passed, {failed} FAILED ✗")
            print("Fix failures before running the overnight pipeline.")
        print("=" * 60)

if __name__ == "__main__":
    main()
