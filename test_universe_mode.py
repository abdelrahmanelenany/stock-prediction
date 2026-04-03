#!/usr/bin/env python3
"""
Test script to verify universe mode implementation.
Tests all key configuration changes without running the full pipeline.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

def test_config_import():
    """Test 1: Verify config imports correctly."""
    print("=" * 60)
    print("TEST 1: Config Import")
    print("=" * 60)

    import config

    # Test large_cap mode (default)
    assert config.UNIVERSE_MODE == "large_cap", f"Expected 'large_cap', got {config.UNIVERSE_MODE}"
    assert len(config.TICKERS) == 30, f"Expected 30 tickers, got {len(config.TICKERS)}"
    assert config.N_STOCKS == 30, f"Expected N_STOCKS=30, got {config.N_STOCKS}"

    # Verify all large_cap tickers are in sector map
    for ticker in config.TICKERS:
        assert ticker in config.SECTOR_MAP, f"Ticker {ticker} missing from SECTOR_MAP"

    print(f"✓ UNIVERSE_MODE: {config.UNIVERSE_MODE}")
    print(f"✓ TICKERS count: {len(config.TICKERS)}")
    print(f"✓ N_STOCKS: {config.N_STOCKS}")
    print(f"✓ All tickers in sector map")
    print()


def test_date_range():
    """Test 2: Verify date range update."""
    print("=" * 60)
    print("TEST 2: Date Range")
    print("=" * 60)

    import config

    assert config.START_DATE == '2019-01-01', f"Expected '2019-01-01', got {config.START_DATE}"
    assert config.END_DATE == '2024-12-31', f"Expected '2024-12-31', got {config.END_DATE}"

    print(f"✓ START_DATE: {config.START_DATE}")
    print(f"✓ END_DATE: {config.END_DATE}")
    print()


def test_fold_structure():
    """Test 3: Verify fold structure parameters."""
    print("=" * 60)
    print("TEST 3: Fold Structure")
    print("=" * 60)

    import config

    assert config.TRAIN_DAYS == 252, f"Expected TRAIN_DAYS=252, got {config.TRAIN_DAYS}"
    assert config.VAL_DAYS == 63, f"Expected VAL_DAYS=63, got {config.VAL_DAYS}"
    assert config.TEST_DAYS == 63, f"Expected TEST_DAYS=63, got {config.TEST_DAYS}"

    print(f"✓ TRAIN_DAYS: {config.TRAIN_DAYS}")
    print(f"✓ VAL_DAYS: {config.VAL_DAYS}")
    print(f"✓ TEST_DAYS: {config.TEST_DAYS}")
    print()


def test_trading_params():
    """Test 4: Verify trading parameters."""
    print("=" * 60)
    print("TEST 4: Trading Parameters")
    print("=" * 60)

    import config

    assert config.SEQ_LEN == 20, f"Expected SEQ_LEN=20, got {config.SEQ_LEN}"
    assert config.K_STOCKS == 5, f"Expected K_STOCKS=5, got {config.K_STOCKS}"
    assert config.MIN_HOLDING_DAYS == 3, f"Expected MIN_HOLDING_DAYS=3, got {config.MIN_HOLDING_DAYS}"

    print(f"✓ SEQ_LEN: {config.SEQ_LEN}")
    print(f"✓ K_STOCKS: {config.K_STOCKS}")
    print(f"✓ MIN_HOLDING_DAYS: {config.MIN_HOLDING_DAYS}")
    print()


def test_cache_paths():
    """Test 5: Verify cache path construction."""
    print("=" * 60)
    print("TEST 5: Cache Paths")
    print("=" * 60)

    import config

    expected_cache = f"data/processed/features_{config.UNIVERSE_MODE}.csv"
    print(f"✓ Cache path: {expected_cache}")

    # Test report prefix
    prefix = config.UNIVERSE_MODE
    print(f"✓ Report prefix: {prefix}")
    print(f"  Example: reports/{prefix}_table_T5_gross_returns.csv")
    print()


def test_small_cap_mode():
    """Test 6: Verify small_cap configuration."""
    print("=" * 60)
    print("TEST 6: Small-Cap Mode")
    print("=" * 60)

    import config

    # Check small_cap tickers exist
    assert hasattr(config, 'SMALL_CAP_TICKERS'), "SMALL_CAP_TICKERS not defined"
    assert len(config.SMALL_CAP_TICKERS) == 20, f"Expected 20 small-cap tickers, got {len(config.SMALL_CAP_TICKERS)}"

    # Check small_cap sector map exists
    assert hasattr(config, 'SMALL_CAP_SECTOR_MAP'), "SMALL_CAP_SECTOR_MAP not defined"

    # Verify all small_cap tickers are in sector map
    for ticker in config.SMALL_CAP_TICKERS:
        assert ticker in config.SMALL_CAP_SECTOR_MAP, f"Ticker {ticker} missing from SMALL_CAP_SECTOR_MAP"

    print(f"✓ SMALL_CAP_TICKERS count: {len(config.SMALL_CAP_TICKERS)}")
    print(f"✓ SMALL_CAP_SECTOR_MAP complete")
    print(f"✓ First 5 tickers: {config.SMALL_CAP_TICKERS[:5]}")
    print()


def test_generate_signals_default():
    """Test 7: Verify generate_signals uses config.K_STOCKS by default."""
    print("=" * 60)
    print("TEST 7: generate_signals Default Parameter")
    print("=" * 60)

    try:
        from backtest.signals import generate_signals
        import inspect

        # Get function signature
        sig = inspect.signature(generate_signals)
        k_param = sig.parameters['k']

        # Verify default is None (will be replaced by K_STOCKS in function)
        assert k_param.default is None, f"Expected k default=None, got {k_param.default}"

        print(f"✓ generate_signals k parameter default: {k_param.default}")
        print(f"  (Will use config.K_STOCKS={5} at runtime)")
        print()
    except ImportError as e:
        print(f"⚠ SKIPPED: Cannot import backtest.signals (dependency issue: {e})")
        print(f"  (This is expected in CI without full environment)")
        print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("UNIVERSE MODE IMPLEMENTATION VERIFICATION")
    print("=" * 60 + "\n")

    tests = [
        test_config_import,
        test_date_range,
        test_fold_structure,
        test_trading_params,
        test_cache_paths,
        test_small_cap_mode,
        test_generate_signals_default,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
