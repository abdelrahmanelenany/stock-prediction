import os
import pandas as pd
import numpy as np

import config
from config import LARGE_CAP_CONFIG, SMALL_CAP_CONFIG
from backtest.signals import generate_signals, smooth_probabilities, apply_holding_period_constraint, compute_turnover_and_holding_stats
from backtest.portfolio import compute_portfolio_returns
from backtest.metrics import compute_metrics, evaluate_classification, compute_daily_auc
from evaluation.metrics_utils import compute_subperiod_metrics
from main import save_all_results

# Mirror main.py: respect invert_signals so classification metrics match trading direction.
_UNIVERSE_CFG = LARGE_CAP_CONFIG if config.UNIVERSE_MODE == 'large_cap' else SMALL_CAP_CONFIG
INVERT_SIGNALS: bool = _UNIVERSE_CFG.invert_signals

def run_combined_backtest(baseline_preds_path: str, lstm_preds_path: str, reports_dir: str = 'reports_combined'):
    print(f"Loading Baseline Predictions from: {baseline_preds_path}")
    base_df = pd.read_csv(baseline_preds_path, parse_dates=['Date'])
    
    print(f"Loading LSTM Predictions from: {lstm_preds_path}")
    lstm_df = pd.read_csv(lstm_preds_path, parse_dates=['Date'])
    
    print("Combining predictions...")
    full_preds = base_df.copy()
    full_preds['Prob_LSTM_B'] = lstm_df['Prob_LSTM_B']

    model_cols = {
        'LR': 'Prob_LR',
        'RF': 'Prob_RF',
        'XGBoost': 'Prob_XGB',
        'LSTM-B': 'Prob_LSTM_B',
    }

    port_returns_gross = {}
    port_returns_net_5 = {}
    class_metrics = []
    all_signals = []
    daily_returns_gross = {'Date': None}
    daily_returns_net_5 = {'Date': None}

    print('\n' + '=' * 60)
    print('BACKTESTING ALL MODELS')
    print('=' * 60)

    for model_name, prob_col in model_cols.items():
        valid_preds = full_preds.dropna(subset=[prob_col]).copy()
        
        if len(valid_preds) == 0:
            print(f'  {model_name:<12}  [API: NO PREDICTIONS. SKIPPING]')
            continue

        smoothed_preds = smooth_probabilities(
            valid_preds, prob_col,
            alpha=config.SIGNAL_SMOOTH_ALPHA,
            ema_method=config.SIGNAL_EMA_METHOD if hasattr(config, 'SIGNAL_EMA_METHOD') else 'ewm',
            ema_span=config.SIGNAL_EMA_SPAN if hasattr(config, 'SIGNAL_EMA_SPAN') else 20,
        )
        smoothed_col = f'{prob_col}_Smooth'

        sig_df, sig_diag = generate_signals(
            smoothed_preds, k=config.K_STOCKS, prob_col=smoothed_col,
            return_diagnostics=True,
        )
        sig_df = apply_holding_period_constraint(sig_df, min_hold_days=config.MIN_HOLDING_DAYS)
        hold_st = compute_turnover_and_holding_stats(sig_df, k=config.K_STOCKS)

        sig_df['Model'] = model_name
        all_signals.append(sig_df)

        port_gross = compute_portfolio_returns(
            sig_df, tc_bps=0, k=config.K_STOCKS, slippage_bps=getattr(config, 'SLIPPAGE_BPS', 0),
        )
        port_net_5 = compute_portfolio_returns(
            sig_df, tc_bps=config.TC_BPS, k=config.K_STOCKS, slippage_bps=getattr(config, 'SLIPPAGE_BPS', 0),
        )

        port_returns_gross[model_name] = port_gross
        port_returns_net_5[model_name] = port_net_5

        if daily_returns_gross['Date'] is None:
            daily_returns_gross['Date'] = port_gross.index
            daily_returns_net_5['Date'] = port_net_5.index
        daily_returns_gross[model_name] = port_gross['Gross_Return'].values
        daily_returns_net_5[model_name] = port_net_5['Net_Return'].values

        y_true = valid_preds[config.TARGET_COL].values
        y_prob = valid_preds[prob_col].values
        cm = evaluate_classification(y_true, y_prob, invert_probs=INVERT_SIGNALS)

        daily_auc = compute_daily_auc(
            valid_preds, prob_col, config.TARGET_COL, invert_probs=INVERT_SIGNALS
        )
        cm['Daily AUC (mean)'] = daily_auc['Daily AUC (mean)']
        cm['Daily AUC (std)'] = daily_auc['Daily AUC (std)']
        cm['Signals Inverted'] = INVERT_SIGNALS

        cm['Model'] = model_name
        class_metrics.append(cm)

        m = compute_metrics(port_gross['Gross_Return'])
        print(f'  {model_name:<12}  '
              f'Sharpe={m["Sharpe Ratio"]:>6.3f}  '
              f'Sortino={m["Sortino Ratio"]:>6.3f}  '
              f'Ann.Ret={m["Annualized Return (%)"]:.2f}%  '
              f'MDD={m["Max Drawdown (%)"]:.2f}%')

    results_gross = [compute_metrics(port_returns_gross[m]['Gross_Return']) | {'Model': m} for m in port_returns_gross]
    results_net_5 = [compute_metrics(port_returns_net_5[m]['Net_Return']) | {'Model': m} for m in port_returns_net_5]
    
    subperiod_metrics = None
    if 'LSTM-B' in port_returns_net_5:
        try:
            subperiod_metrics = compute_subperiod_metrics(port_returns_net_5['LSTM-B']['Net_Return'])
        except Exception as e:
            print(f'Warning: Could not compute subperiod metrics: {e}')

    results_dict = {
        'gross': results_gross,
        'net_5': results_net_5,
        'classification': class_metrics,
        'subperiod': subperiod_metrics,
    }

    signals_df = pd.concat(all_signals).reset_index(drop=True)
    full_preds.to_csv(f"{reports_dir}/full_predictions_combined.csv", index=False)

    save_all_results(
        results_dict=results_dict,
        daily_returns_dict={
            'gross': pd.DataFrame(daily_returns_gross),
            'net_5': pd.DataFrame(daily_returns_net_5),
        },
        signals_dict=signals_df,
        reports_dir=reports_dir
    )
    print(f"\nAll models combined and saved into {reports_dir}/!")

if __name__ == "__main__":
    b_path = 'reports/full_predictions_baselines.csv'
    l_path = 'reports/full_predictions_lstms.csv'
    
    if not os.path.exists(b_path) or not os.path.exists(l_path):
        print(f"ERROR: You must rename your saved prediction csvs to:")
        print(f" -> {b_path}")
        print(f" -> {l_path}")
        print("\n(You will get 'full_predictions.csv' inside the 'reports' folder after successfully running main.py. Just rename them and put them here.)")
    else:
        run_combined_backtest(b_path, l_path)