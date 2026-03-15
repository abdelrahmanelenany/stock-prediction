# config.py — Single source of truth for all hyperparameters and constants
TICKERS = [
    # Tech (11)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'ADBE', 'CRM', 'INTC', 'CSCO', 'NFLX',
    # Finance (4)
    'JPM', 'V', 'MA', 'BRK-B',
    # Healthcare (5)
    'JNJ', 'UNH', 'ABT', 'MRK', 'PFE',
    # Consumer (5)
    'TSLA', 'DIS', 'PG', 'HD', 'NKE',
    # Energy (2)
    'XOM', 'CVX',
    # Industrials / Comm (3)
    'PEP', 'COST', 'CMCSA',
]
START_DATE = '2015-01-01'
END_DATE   = '2024-12-31'

# Walk-forward fold structure
TRAIN_DAYS = 500   # ~2 years
VAL_DAYS   = 125   # ~6 months (hyperparameter tuning)
TEST_DAYS  = 125   # ~6 months (out-of-sample evaluation)

# Feature config
SEQ_LEN               = 60
LAGGED_RETURN_PERIODS = list(range(1, 21)) + [40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
N_RETURN_FEATURES     = 31
N_TECH_FEATURES       = 10
N_RANK_FEATURES       = 6
N_TOTAL_FEATURES      = N_RETURN_FEATURES + N_TECH_FEATURES + N_RANK_FEATURES  # 47

# Trading
K_STOCKS = 3   # Number of long / short positions per day (top/bottom 10% of 30 stocks)
TC_BPS   = 5   # Transaction cost per half-turn in basis points (0.0005)

# LSTM — simplified architecture (~13K sequences/fold with 30 stocks)
LSTM_HIDDEN      = 32
LSTM_LAYERS      = 1
LSTM_DROPOUT     = 0.2
LSTM_LR          = 0.001
LSTM_BATCH       = 128
LSTM_MAX_EPOCHS  = 200
LSTM_PATIENCE    = 10

# Random Forest — regularization for 30-stock cross-section
RF_PARAM_GRID = {
    'n_estimators':     [200, 300],
    'max_depth':        [3, 5],
    'min_samples_leaf': [50, 100],
}

# XGBoost — stronger regularization to prevent overfitting
XGB_MAX_DEPTH    = 3
XGB_ETA          = 0.01
XGB_SUBSAMPLE    = 0.6
XGB_COLSAMPLE    = 0.5
XGB_ROUNDS       = 1000
XGB_EARLY_STOP   = 50
XGB_REG_ALPHA    = 0.1    # L1 regularization
XGB_REG_LAMBDA   = 1.0    # L2 regularization

RANDOM_SEED = 42
