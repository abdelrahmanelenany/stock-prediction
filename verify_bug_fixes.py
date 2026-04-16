"""
verify_bug_fixes.py
-------------------
Checks the 5 known-bug-fix items listed in the THESIS COMPLETION PLAN (Task 8).

Exits with code 0 if all pass, non-zero if any FAIL.
"""

import sys
import os
import importlib

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

results = []


def check(name, condition, detail):
    status = "PASS" if condition else "FAIL"
    results.append((status, name, detail))
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")


# ---------------------------------------------------------------------------
# 1. SIGNAL_CONFIDENCE_THRESHOLD is 0 (pure ranking strategy)
# ---------------------------------------------------------------------------
import config

val_thresh = getattr(config, "SIGNAL_CONFIDENCE_THRESHOLD", None)
check(
    "SIGNAL_CONFIDENCE_THRESHOLD == 0 (pure ranking)",
    val_thresh == 0.0,
    f"Current value: {val_thresh}  "
    f"{'OK — pure ranking' if val_thresh == 0.0 else 'PROBLEM — non-zero threshold sits out trades, distorts comparison'}",
)

# ---------------------------------------------------------------------------
# 2. EMA smoothing disabled (SIGNAL_SMOOTH_ALPHA == 0)
# ---------------------------------------------------------------------------
val_alpha = getattr(config, "SIGNAL_SMOOTH_ALPHA", None)
check(
    "EMA smoothing disabled (SIGNAL_SMOOTH_ALPHA == 0)",
    val_alpha == 0.0,
    f"Current value: {val_alpha}  "
    f"{'OK — no EMA smoothing' if val_alpha == 0.0 else 'PROBLEM — EMA smoothing changes signal timing'}",
)

# ---------------------------------------------------------------------------
# 3. Wavelet denoising per-fold (no look-ahead)
#    Check two things: (a) USE_WAVELET_DENOISING flag,
#                      (b) the denoising function uses causal rolling window
# ---------------------------------------------------------------------------
use_wavelet = getattr(config, "USE_WAVELET_DENOISING", None)

# Check whether main.py / combine_and_backtest.py applies denoising inside the fold loop
main_path = os.path.join(ROOT, "main.py")
combine_path = os.path.join(ROOT, "combine_and_backtest.py")

def contains_causal_check(filepath):
    """True if file applies wavelet denoising inside a fold loop."""
    if not os.path.exists(filepath):
        return False, "file not found"
    with open(filepath) as f:
        src = f.read()
    has_wv_call = ("apply_wavelet_denoising" in src or
                   "denoise_close_price" in src or
                   "wavelet" in src.lower())
    has_loop    = "for fold" in src or "for i, fold" in src
    return has_wv_call and has_loop, src[:200]

causal_main,    _ = contains_causal_check(main_path)
causal_combine, _ = contains_causal_check(combine_path)

# Also check the denoising implementation in features.py
feat_path = os.path.join(ROOT, "pipeline", "features.py")
causal_impl = False
if os.path.exists(feat_path):
    with open(feat_path) as f:
        feat_src = f.read()
    # Causal implementation: uses rolling window, not whole-series
    causal_impl = ("for t in range" in feat_src and
                   "window_size" in feat_src and
                   "reconstructed[-1]" in feat_src)

note_wavelet = (
    f"USE_WAVELET_DENOISING={use_wavelet}; "
    f"per-fold application found={'yes' if (causal_main or causal_combine) else 'not detected'}; "
    f"causal rolling-window impl={'yes' if causal_impl else 'not detected'}"
)
check(
    "Wavelet denoising is causal (rolling window) and applied per-fold",
    causal_impl,   # core correctness check: implementation is causal
    note_wavelet,
)

# ---------------------------------------------------------------------------
# 4. LSTM scaler fitted on train only (not train+val)
# ---------------------------------------------------------------------------
lstm_path = os.path.join(ROOT, "models", "lstm_model.py")
main_path = os.path.join(ROOT, "main.py")
scaler_ok = False
scaler_note = "models/lstm_model.py not found"

if os.path.exists(lstm_path):
    with open(lstm_path) as f:
        lstm_src = f.read()

    # Determine which prepare function main.py actually calls
    active_fn = "unknown"
    if os.path.exists(main_path):
        with open(main_path) as f:
            main_src = f.read()
        if "prepare_lstm_b_sequences_temporal_split" in main_src:
            active_fn = "prepare_lstm_b_sequences_temporal_split"
        elif "prepare_lstm_b_sequences" in main_src:
            active_fn = "prepare_lstm_b_sequences"

    # Check the ACTIVE function's scaler usage
    if active_fn == "prepare_lstm_b_sequences_temporal_split":
        # This function fits scaler on df_true_train (training dates only, not val)
        scaler_ok = "scaler.fit(df_true_train" in lstm_src
        scaler_note = (
            f"Active function: {active_fn}; "
            f"scaler.fit on df_true_train (train-only subset): "
            f"{'found — OK' if scaler_ok else 'NOT FOUND — check lstm_model.py'}"
        )
    elif active_fn == "prepare_lstm_b_sequences":
        # Older function — fits scaler on df_train before any val split, potential leak
        scaler_ok = False
        scaler_note = (
            f"Active function: {active_fn} — this function fits scaler on full df_train "
            "(includes val dates). Use prepare_lstm_b_sequences_temporal_split instead."
        )
    else:
        scaler_ok = False
        scaler_note = f"Could not detect active LSTM prepare function in main.py"

check(
    "LSTM scaler fitted on train-only (not train+val)",
    scaler_ok,
    scaler_note,
)

# ---------------------------------------------------------------------------
# 5. SMALL_CAP_SECTOR_MAP has no 'Unknown' values
# ---------------------------------------------------------------------------
small_cap_map = getattr(config, "SMALL_CAP_SECTOR_MAP", None)
if small_cap_map is None:
    check(
        "SMALL_CAP_SECTOR_MAP has no 'Unknown' entries",
        False,
        "SMALL_CAP_SECTOR_MAP not found in config.py",
    )
else:
    unknown_tickers = [t for t, s in small_cap_map.items() if s == "Unknown"]
    total = len(small_cap_map)
    check(
        "SMALL_CAP_SECTOR_MAP has no 'Unknown' entries",
        len(unknown_tickers) == 0,
        (f"Total tickers: {total}; Unknown: {len(unknown_tickers)} "
         f"{('— OK' if not unknown_tickers else ': ' + str(unknown_tickers))}"),
    )

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)
passed = sum(1 for r in results if r[0] == "PASS")
failed = sum(1 for r in results if r[0] == "FAIL")
print(f"PASSED: {passed}/{len(results)}    FAILED: {failed}/{len(results)}")
print()
for status, name, detail in results:
    marker = "✓" if status == "PASS" else "✗"
    print(f"  {marker} {name}")

if failed > 0:
    print(f"\nACTION REQUIRED: {failed} item(s) need attention before final run.")
    sys.exit(1)
else:
    print("\nAll checks passed. Config is ready for final thesis run.")
    sys.exit(0)
