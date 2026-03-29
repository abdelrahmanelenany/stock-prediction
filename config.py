# config.py — Single source of truth for all hyperparameters and constants
# Implements Bhandari et al. (2022) extensions from IMPLEMENTATION_EXTENSIONS.md
# Expanded to ~100 S&P 500 components for better diversification and more training data
TICKERS = [
    # Technology (20)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'ADBE', 'CRM', 'INTC', 'CSCO',
    'NFLX', 'ORCL', 'AVGO', 'QCOM', 'TXN', 'AMD', 'IBM', 'NOW', 'INTU', 'AMAT',
    # Finance (15)
    'JPM', 'V', 'MA', 'BRK-B', 'GS', 'BAC', 'WFC', 'MS', 'AXP', 'C',
    'BLK', 'SCHW', 'CME', 'ICE', 'USB',
    # Healthcare (15)
    'JNJ', 'UNH', 'ABT', 'MRK', 'PFE', 'LLY', 'ABBV', 'TMO', 'DHR', 'BMY',
    'AMGN', 'GILD', 'MDT', 'ISRG', 'CVS',
    # Consumer Discretionary (12)
    'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'TGT', 'LOW', 'TJX', 'BKNG', 'MAR',
    'ORLY', 'CMG',
    # Consumer Staples (8)
    'PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'CL',
    # Communication Services (6)
    'DIS', 'CMCSA', 'TMUS', 'VZ', 'T', 'CHTR',
    # Energy (8)
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',
    # Industrials (10)
    'HON', 'CAT', 'UPS', 'GE', 'RTX', 'BA', 'DE', 'LMT', 'UNP', 'FDX',
    # Utilities (4)
    'NEE', 'DUK', 'SO', 'D',
    # Real Estate (4)
    'AMT', 'PLD', 'CCI', 'EQIX',
    # Materials (3)
    'LIN', 'APD', 'SHW',
]

SECTOR_MAP = {
    # Technology
    'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOGL': 'Tech', 'AMZN': 'Tech', 'NVDA': 'Tech',
    'META': 'Tech', 'ADBE': 'Tech', 'CRM': 'Tech', 'INTC': 'Tech', 'CSCO': 'Tech',
    'NFLX': 'Tech', 'ORCL': 'Tech', 'AVGO': 'Tech', 'QCOM': 'Tech', 'TXN': 'Tech',
    'AMD': 'Tech', 'IBM': 'Tech', 'NOW': 'Tech', 'INTU': 'Tech', 'AMAT': 'Tech',
    # Finance
    'JPM': 'Finance', 'V': 'Finance', 'MA': 'Finance', 'BRK-B': 'Finance', 'GS': 'Finance',
    'BAC': 'Finance', 'WFC': 'Finance', 'MS': 'Finance', 'AXP': 'Finance', 'C': 'Finance',
    'BLK': 'Finance', 'SCHW': 'Finance', 'CME': 'Finance', 'ICE': 'Finance', 'USB': 'Finance',
    # Healthcare
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'ABT': 'Healthcare', 'MRK': 'Healthcare',
    'PFE': 'Healthcare', 'LLY': 'Healthcare', 'ABBV': 'Healthcare', 'TMO': 'Healthcare',
    'DHR': 'Healthcare', 'BMY': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare',
    'MDT': 'Healthcare', 'ISRG': 'Healthcare', 'CVS': 'Healthcare',
    # Consumer Discretionary
    'TSLA': 'Consumer', 'HD': 'Consumer', 'NKE': 'Consumer', 'MCD': 'Consumer',
    'SBUX': 'Consumer', 'TGT': 'Consumer', 'LOW': 'Consumer', 'TJX': 'Consumer',
    'BKNG': 'Consumer', 'MAR': 'Consumer', 'ORLY': 'Consumer', 'CMG': 'Consumer',
    # Consumer Staples
    'PG': 'Staples', 'KO': 'Staples', 'PEP': 'Staples', 'COST': 'Staples',
    'WMT': 'Staples', 'PM': 'Staples', 'MO': 'Staples', 'CL': 'Staples',
    # Communication Services
    'DIS': 'Comm', 'CMCSA': 'Comm', 'TMUS': 'Comm', 'VZ': 'Comm', 'T': 'Comm', 'CHTR': 'Comm',
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    'EOG': 'Energy', 'MPC': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy',
    # Industrials
    'HON': 'Industrial', 'CAT': 'Industrial', 'UPS': 'Industrial', 'GE': 'Industrial',
    'RTX': 'Industrial', 'BA': 'Industrial', 'DE': 'Industrial', 'LMT': 'Industrial',
    'UNP': 'Industrial', 'FDX': 'Industrial',
    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
    # Real Estate
    'AMT': 'REIT', 'PLD': 'REIT', 'CCI': 'REIT', 'EQIX': 'REIT',
    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials',
}

START_DATE = '2015-01-01'
END_DATE   = '2024-12-31'

# Walk-forward fold structure
TRAIN_DAYS = 500   # ~2 years
VAL_DAYS   = 125   # ~6 months (hyperparameter tuning)
TEST_DAYS  = 125   # ~6 months (out-of-sample evaluation)

# Walk-forward: stride between folds (None = roll by one test window)
WALK_FORWARD_STRIDE = None  # resolved to TEST_DAYS when None
# "rolling" = fixed-length train window slides; "expanding" = train always from index 0 to val start
TRAIN_WINDOW_MODE = "rolling"

# Train-window experiment grid (~2y / 3y / 5y trading days at 252 d/y)
TRAIN_DAYS_CANDIDATES = [504, 756, 1260]

# Optional train-only quantile clipping applied before scaler (fit quantiles on train rows)
WINSORIZE_ENABLED = False
WINSORIZE_LOWER_Q = 0.005
WINSORIZE_UPPER_Q = 0.995

# Fold-level JSON/CSV artifacts under reports/fold_reports/
SAVE_FOLD_REPORTS = True

# Extra execution cost (same half-turn structure as TC_BPS); 0 = off
SLIPPAGE_BPS = 0.0

# Signal EMA: use alpha (SIGNAL_SMOOTH_ALPHA) or pandas span
SIGNAL_EMA_METHOD = "alpha"  # "alpha" | "span"
SIGNAL_EMA_SPAN = None  # if set and METHOD=span, e.g. 10

# Run raw ranking vs full post-process pipeline; writes reports/signal_ablation_summary.csv
RUN_SIGNAL_ABLATION = False

# LSTM training audit / diagnostics
LSTM_LOG_EVERY_EPOCH = True
LSTM_SAVE_TRAINING_CSV = True
LSTM_AUDIT_GRAD_NORM = False
LSTM_MAX_GRAD_NORM = None  # if float, clip gradients to this norm
LSTM_FLAT_AUC_WARN_epochs = 8
LSTM_FLAT_AUC_EPS = 0.02
LSTM_OVERFIT_LOSS_RATIO = 3.0
LSTM_OVERFIT_WARN_epochs = 6

# Optional LR grid for experiments/lstm_lr_sweep.py (single value = no sweep)
LSTM_LR_GRID = [0.001]
LSTM_LR_SWEEP_MAX_EPOCHS = 40  # capped budget for experiments/lstm_lr_sweep.py

# Market/sector feature horizons (used in pipeline/features.py)
MARKET_RETURN_HORIZONS = (1, 5, 21)
MARKET_VOL_WINDOWS = (20, 60)
BETA_WINDOW = 60
SECTOR_RETURN_EXTRA_HORIZONS = (21,)
SECTOR_VOL_EXTRA_WINDOWS = (60,)
SECTOR_REL_ZSCORE_RETURN_COLS = ("Return_1d",)

# ── Feature config (10 active features including momentum + Context features) ────────────────
SEQ_LEN               = 60

# Context features flags
MARKET_FEATURES_ENABLED = True
SECTOR_FEATURES_ENABLED = True

# Master feature union: all features used by at least one model
ALL_FEATURE_COLS = [
    "Return_1d",        # LSTM-A, LSTM-B, Baselines
    "Return_5d",        # LSTM-B, Baselines (weekly momentum)
    "Return_21d",       # LSTM-B, Baselines (monthly momentum)
    "RSI_14",           # LSTM-A, LSTM-B, Baselines
    "MACD",             # LSTM-A only
    "ATR_14",           # LSTM-A only
    "BB_PctB",          # LSTM-B, Baselines
    "RealVol_20d",      # LSTM-B, Baselines
    "Volume_Ratio",     # LSTM-B, Baselines
    "SectorRelReturn",  # LSTM-B, Baselines
]

if MARKET_FEATURES_ENABLED:
    ALL_FEATURE_COLS.extend([
        "Market_Return_1d",
        "Market_Return_5d",
        "Market_Return_21d",
        "Market_Vol_20d",
        "Market_Vol_60d",
        "RelToMarket_1d",
        "RelToMarket_5d",
        "RelToMarket_21d",
        f"Beta_{BETA_WINDOW}d",
    ])

if SECTOR_FEATURES_ENABLED:
    ALL_FEATURE_COLS.extend([
        "Sector_Return_1d",
        "Sector_Return_5d",
        "Sector_Return_21d",
        "Sector_Vol_20d",
        "Sector_Vol_60d",
        "SectorRelZ_Return_1d",
    ])

N_TOTAL_FEATURES = len(ALL_FEATURE_COLS)  # Dynamically computed

# ── Per-model feature sets (Section 7.1) ────────────────────────────────────
LSTM_A_FEATURE_COLS = [
    "MACD",        # 12/26 EMA difference (Bhandari §4.3)
    "RSI_14",      # 14-day RSI (Bhandari §4.3)
    "ATR_14",      # 14-day ATR (Bhandari §4.3)
    "Return_1d",   # 1-day simple return
    "Return_5d",   # 5-day simple return (weekly momentum)
    "Return_21d",  # 21-day simple return (monthly momentum)
]

LSTM_B_FEATURE_COLS = [
    "Return_1d",
    "Return_5d",        # Weekly momentum
    "Return_21d",       # Monthly momentum
    "RSI_14",
    "BB_PctB",
    "RealVol_20d",
    "Volume_Ratio",
    "SectorRelReturn",
]

if MARKET_FEATURES_ENABLED:
    LSTM_B_FEATURE_COLS.extend([
        "Market_Return_1d",
        "Market_Return_5d",
        "Market_Return_21d",
        "Market_Vol_20d",
        "Market_Vol_60d",
        "RelToMarket_1d",
        "RelToMarket_5d",
        "RelToMarket_21d",
        f"Beta_{BETA_WINDOW}d",
    ])

if SECTOR_FEATURES_ENABLED:
    LSTM_B_FEATURE_COLS.extend([
        "Sector_Return_1d",
        "Sector_Return_5d",
        "Sector_Return_21d",
        "Sector_Vol_20d",
        "Sector_Vol_60d",
        "SectorRelZ_Return_1d",
    ])

# Baselines use LSTM-B features for fair comparison
BASELINE_FEATURE_COLS = LSTM_B_FEATURE_COLS

# Trading
K_STOCKS = 10  # Number of long / short positions per day (top/bottom 10% of 105 stocks)
TC_BPS   = 5   # Transaction cost per half-turn in basis points (0.0005)
SIGNAL_SMOOTH_ALPHA = 0.3  # EMA smoothing factor for probabilities (lower = stickier)
SIGNAL_CONFIDENCE_THRESHOLD = 0.05  # Requires prob to be >= 0.5 + threshold or <= 0.5 - threshold
SIGNAL_USE_ZSCORE = True  # Use cross-sectional z-score for more robust signal generation
MIN_HOLDING_DAYS = 5  # Enforce minimum holding period to reduce turnover

# Execution semantics (see backtest/portfolio.py): features at date t use data through t;
# signals rank at t; portfolio earns Return_NextDay (close t to close t+1).

# ── LSTM-A: Bhandari-inspired technical indicator LSTM (4 features) ─────────
# Architecture is determined by hyperparameter tuning (Section 1 / 7.4)
LSTM_A_FEATURES      = LSTM_A_FEATURE_COLS  # 4 features: MACD, RSI, ATR, Return_1d
LSTM_A_SEQ_LEN       = 60                    # matches LSTM-B for fair comparison
LSTM_A_OPTIMIZER     = 'adam'                # will be tuned
LSTM_A_LR            = 0.001                 # will be tuned
LSTM_A_BATCH         = 128                   # will be tuned
LSTM_A_MAX_EPOCHS    = 200
LSTM_A_PATIENCE      = 15
LSTM_A_VAL_SPLIT     = 0.2

# LSTM-A architecture search grid (Bhandari §3.3 Algorithm 2)
# Architecture is data-driven, not fixed
LSTM_A_ARCH_GRID = {
    "hidden_size": [16, 32, 64],   # small range appropriate for 4-feature input
    "num_layers":  [1, 2],         # Bhandari §5.5 found single-layer often wins
    "dropout":     [0.1, 0.2],
}

# ── LSTM-B: Extended ablation — curated 6-feature set (fixed architecture) ────
LSTM_B_FEATURES      = LSTM_B_FEATURE_COLS
LSTM_B_SEQ_LEN       = 60
LSTM_B_HIDDEN_SIZE   = 64                     # fixed architecture for LSTM-B
LSTM_B_NUM_LAYERS    = 2
LSTM_B_DROPOUT       = 0.2
LSTM_B_HIDDEN        = LSTM_B_HIDDEN_SIZE     # alias for backward compatibility
LSTM_B_LAYERS        = LSTM_B_NUM_LAYERS
LSTM_B_OPTIMIZER     = 'adam'
LSTM_B_LR            = 0.001
LSTM_B_BATCH         = 128
LSTM_B_MAX_EPOCHS    = 200
LSTM_B_PATIENCE      = 15
LSTM_B_LR_PATIENCE   = 7
LSTM_B_LR_FACTOR     = 0.5
LSTM_B_VAL_SPLIT     = 0.2

# ── Shared LSTM settings ──────────────────────────────────────────────────
LSTM_WD              = 1e-5                   # weight decay (both models)

# ── LSTM Hyperparameter Search Grid (Bhandari §3.3) ──────────────────────────
# Shared by both LSTM-A and LSTM-B for the training hyperparameter search
LSTM_HYPERPARAM_GRID = {
    "optimizer":      ["adam", "adagrad", "nadam"],   # paper tests these three
    "learning_rate":  [0.1, 0.01, 0.001],             # paper tests these three
    "batch_size":     [32, 64, 128],                  # scaled up from paper's 4/8/16
}
LSTM_TUNE_REPLICATES = 3      # paper uses 10; 3 is feasible on M4 for a thesis
LSTM_TUNE_PATIENCE   = 5      # early stopping patience during tuning (paper §3.3)
LSTM_TUNE_MAX_EPOCHS = 50     # cap tuning runs; full training uses MAX_EPOCHS

# ── Wavelet Denoising (Bhandari §4.5) ────────────────────────────────────────
USE_WAVELET_DENOISING = False    # Set False to use raw prices (Fixes OOS domain shift)
WAVELET_TYPE          = "haar"  # Paper uses Haar wavelets
WAVELET_LEVEL         = 1       # Decomposition level; 1 is appropriate for daily data
WAVELET_MODE          = "soft"  # Thresholding mode: 'soft' (paper) or 'hard'
WAVELET_WINDOW_SIZE   = 128     # Lookback window for causal denoising (prevents leakage)

# ── Normalization (Bhandari §4.5 uses MinMax; our default is Standard) ───────
SCALER_TYPE = "standard"   # Options: "standard" (default) | "minmax"

# ── Feature Selection (Bhandari §4.4) ────────────────────────────────────────
FEATURE_CORR_THRESHOLD = 0.80   # Drop features with |r| > threshold

# After running analysis/feature_correlation.py, paste the output list here:
# Leave as None to use all ALL_FEATURE_COLS (before selection is run)
FEATURE_COLS_AFTER_SELECTION = ['Return_1d', 'Return_5d', 'Return_21d', 'RSI_14', 'MACD', 'ATR_14', 'RealVol_20d', 'Volume_Ratio', 'SectorRelReturn']

# Random Forest — expanded grid for 50-stock cross-section
RF_PARAM_GRID = {
    'n_estimators':     [300, 500],
    'max_depth':        [5, 10, 15],
    'min_samples_leaf': [30, 50],
}

# XGBoost — grid search over key hyperparameters
XGB_PARAM_GRID = {
    'max_depth':  [3, 4, 5],
    'eta':        [0.01],
    'subsample':  [0.6, 0.7],
}
XGB_COLSAMPLE    = 0.5
XGB_ROUNDS       = 1000
XGB_EARLY_STOP   = 50
XGB_REG_ALPHA    = 0.1    # L1 regularization
XGB_REG_LAMBDA   = 1.0    # L2 regularization

RANDOM_SEED = 42

# ── Model registry (after refactor) ──────────────────────────────────────────
MODELS = ['LR', 'RF', 'XGBoost', 'LSTM-A', 'LSTM-B']
