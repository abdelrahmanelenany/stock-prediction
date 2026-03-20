# config.py — Single source of truth for all hyperparameters and constants
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

# Feature config
SEQ_LEN               = 60
# Remove 200, 220, 240 day lags — these are mostly NaN within 500-day folds
LAGGED_RETURN_PERIODS = list(range(1, 21)) + [40, 60, 80, 100, 120, 140, 160, 180]
N_RETURN_FEATURES     = 28   # was 31 (removed 200d, 220d, 240d)
N_TECH_FEATURES       = 13   # 10 original + RealVol_5d, RealVol_20d, VolAdj_Mom_10d
N_CROSS_FEATURES      = 2    # ReturnDispersion, SectorRelReturn
N_RANK_FEATURES       = 6
N_TOTAL_FEATURES      = N_RETURN_FEATURES + N_TECH_FEATURES + N_CROSS_FEATURES + N_RANK_FEATURES  # 49

# Trading
K_STOCKS = 10  # Number of long / short positions per day (top/bottom 10% of 105 stocks)
TC_BPS   = 5   # Transaction cost per half-turn in basis points (0.0005)
SIGNAL_SMOOTH_ALPHA = 0.3  # EMA smoothing factor for probabilities (lower = stickier)
SIGNAL_CONFIDENCE_THRESHOLD = 0.0  # Pure ranking (was 0.03 — caused signal imbalance with biased probs)
SIGNAL_USE_ZSCORE = True  # Use cross-sectional z-score for more robust signal generation

# ── LSTM-A: Paper-faithful replication (Fischer & Krauss 2017) ─────────────
LSTM_A_FEATURES      = ['Return_1d']          # single feature, as per paper
LSTM_A_SEQ_LEN       = 240                    # ~1 trading year of history
LSTM_A_HIDDEN        = 25                     # matches paper's h=25
LSTM_A_LAYERS        = 1                      # single LSTM layer, as per paper
LSTM_A_DROPOUT       = 0.16                   # matches paper's dropout value
LSTM_A_OPTIMIZER     = 'rmsprop'              # paper explicitly uses RMSprop
LSTM_A_LR            = 0.001
LSTM_A_BATCH         = 512                    # large batch for stable gradients on 1 feature
LSTM_A_MAX_EPOCHS    = 1000                   # paper trains up to 1000 epochs
LSTM_A_PATIENCE      = 10                     # paper uses patience=10
LSTM_A_VAL_SPLIT     = 0.2                    # paper uses 80/20 train/val split

# ── LSTM-B: Extended ablation — curated 6-feature set ────────────────────
LSTM_B_FEATURES      = [
    'Return_1d',        # core price signal
    'RSI_14',           # bounded momentum [0, 100]
    'BB_PctB',          # position within Bollinger band
    'RealVol_20d',      # realized volatility regime
    'Volume_Ratio',     # relative volume anomaly
    'SectorRelReturn',  # cross-sectional sector context
]
LSTM_B_SEQ_LEN       = 60                     # shorter window; justified by wider feature set
LSTM_B_HIDDEN        = 64
LSTM_B_LAYERS        = 2
LSTM_B_DROPOUT       = 0.2
LSTM_B_OPTIMIZER     = 'adam'
LSTM_B_LR            = 0.001
LSTM_B_BATCH         = 128
LSTM_B_MAX_EPOCHS    = 300
LSTM_B_PATIENCE      = 15
LSTM_B_LR_PATIENCE   = 7
LSTM_B_LR_FACTOR     = 0.5
LSTM_B_VAL_SPLIT     = 0.2

# ── Shared LSTM settings ──────────────────────────────────────────────────
LSTM_WD              = 1e-5                   # weight decay (both models)

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
