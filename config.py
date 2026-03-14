# config.py — Single source of truth for all hyperparameters and constants
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'V']
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
N_TOTAL_FEATURES      = N_RETURN_FEATURES + N_TECH_FEATURES  # 41

# Trading
K_STOCKS = 2   # Number of long / short positions per day
TC_BPS   = 5   # Transaction cost per half-turn in basis points (0.0005)

# LSTM hyperparameters
LSTM_HIDDEN      = 64
LSTM_LAYERS      = 2
LSTM_DROPOUT     = 0.2
LSTM_LR          = 0.001
LSTM_BATCH       = 128
LSTM_MAX_EPOCHS  = 200   # reduced from 200; MPS is slow for small LSTMs — use CPU
LSTM_PATIENCE    = 15   # reduced from 15

# Random Forest
RF_N_ESTIMATORS = 500   # reduced from 500; 200 trees is sufficient for 10 stocks
RF_MAX_DEPTH    = 20    # reduced from 20; depth 20 overfits and is slow

# XGBoost
XGB_MAX_DEPTH    = 4
XGB_ETA          = 0.05
XGB_SUBSAMPLE    = 0.7
XGB_COLSAMPLE    = 0.5
XGB_ROUNDS       = 500
XGB_EARLY_STOP   = 30

RANDOM_SEED = 42
