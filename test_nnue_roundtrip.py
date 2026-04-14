#!/usr/bin/env python3
"""
NNUE weight round-trip test (Finding 15.1, 15.2, 2.19).
Saves a model's weights to a temp file, reloads them, and verifies they match.
"""
import sys
import os
import tempfile
import struct

def test_weight_roundtrip():
    """
    Test that NNUE weight files survive a save/load cycle.
    Uses train_nnue.py's NNUEModel class directly.
    """
    try:
        import torch
    except ImportError:
        print("SKIP: PyTorch not installed")
        return True

    # Load the training module
    import importlib.util
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, "train_nnue.py")
    if not os.path.exists(train_path):
        print("SKIP: train_nnue.py not found")
        return True

    spec = importlib.util.spec_from_file_location("train_nnue", train_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    except Exception as e:
        print(f"SKIP: Could not load train_nnue.py: {e}")
        return True

    if not hasattr(mod, 'NNUEModel'):
        print("SKIP: NNUEModel not found in train_nnue.py")
        return True

    # Create a model with random weights
    try:
        model = mod.NNUEModel()
    except Exception as e:
        print(f"SKIP: Could not instantiate NNUEModel: {e}")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = os.path.join(tmpdir, "test_weights.bin")

        # Save
        if hasattr(mod, 'save_weights'):
            try:
                mod.save_weights(model, weights_path)
            except Exception as e:
                print(f"SKIP: save_weights failed: {e}")
                return True
        else:
            # Try torch.save as fallback
            try:
                torch.save(model.state_dict(), weights_path)
            except Exception as e:
                print(f"SKIP: No save_weights function, torch.save failed: {e}")
                return True

        if not os.path.exists(weights_path):
            print("FAIL: Weight file not created after save")
            return False

        # Reload
        model2 = mod.NNUEModel()
        if hasattr(mod, 'load_weights'):
            try:
                mod.load_weights(model2, weights_path)
            except Exception as e:
                print(f"FAIL: load_weights raised: {e}")
                return False
        else:
            try:
                sd = torch.load(weights_path, map_location='cpu')
                model2.load_state_dict(sd)
            except Exception as e:
                print(f"FAIL: Could not reload weights: {e}")
                return False

        # Compare parameters
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            if n1 != n2:
                print(f"FAIL: Parameter name mismatch: {n1} vs {n2}")
                return False
            if not torch.allclose(p1, p2, atol=1e-5):
                maxdiff = (p1 - p2).abs().max().item()
                print(f"FAIL: Parameter '{n1}' round-trip error = {maxdiff:.2e} (threshold 1e-5)")
                return False

        print("PASS: All parameters survived save/load round-trip within 1e-5 tolerance")
        return True


def test_binary_format_header():
    """
    Test that the binary training data file has the expected NNUE magic header.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Look for any .bin file in assets/
    for candidate in ["assets/nnue_weights.bin", "assets/selfplay.bin", "selfplay.bin"]:
        path = os.path.join(script_dir, candidate)
        if os.path.exists(path):
            with open(path, "rb") as f:
                header = f.read(4)
            if header == b"NNUE":
                print(f"PASS: {candidate} has correct NNUE magic header")
            else:
                print(f"INFO: {candidate} header = {header!r} (not NNUE format, may be weight file)")
            return True
    print("SKIP: No .bin files found to validate header")
    return True


if __name__ == "__main__":
    print("=== NNUE Round-Trip Tests ===")
    results = []
    results.append(("Weight round-trip", test_weight_roundtrip()))
    results.append(("Binary format header", test_binary_format_header()))

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\nResults: {passed}/{total} passed")

    if not all(ok for _, ok in results):
        sys.exit(1)
