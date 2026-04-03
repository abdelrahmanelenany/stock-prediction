import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add the project root to sys.path if not running from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import BASELINE_FEATURE_COLS, TARGET_COL, TRAIN_DAYS, VAL_DAYS, TEST_DAYS, K_STOCKS, SIGNAL_SMOOTH_ALPHA
except ImportError:
    BASELINE_FEATURE_COLS = ['Return_1d', 'Return_5d', 'Return_21d', 'RSI_14', 'BB_PctB', 'RealVol_20d', 'Volume_Ratio', 'SectorRelReturn']
    TARGET_COL = 'Target'
    TRAIN_DAYS = 500
    VAL_DAYS = 125
    TEST_DAYS = 125
    K_STOCKS = 10
    SIGNAL_SMOOTH_ALPHA = 0.3

try:
    from pipeline.walk_forward import generate_walk_forward_folds
    from pipeline.standardizer import standardize_fold
    from models.baselines import train_logistic, train_random_forest
    from backtest.signals import smooth_probabilities, generate_signals
except ImportError as e:
    print(f"Error importing pipeline components: {e}")
    print("Please ensure you are running this from the project root.")
    sys.exit(1)

def run_diagnostics():
    print("="*60)
    print("           STOCK PREDICTION PIPELINE DIAGNOSTICS")
    print("="*60)
    
    # Load data
    data_path = 'data/processed/features.csv'
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}")
        return
        
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['Date'])
    df = df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    
    summary_results = {
        'CHECK 1 - Class Balance': 'PASS',
        'CHECK 2 - In-sample Overfit': 'FAIL',
        'CHECK 3 - Trivial Baseline': 'PASS',
        'CHECK 4 - Feature Correlation': 'PASS',
        'CHECK 5 - Date Alignment': '(manual review required)',
        'CHECK 6 - Fold Continuity': 'PASS',
        'CHECK 7 - Distribution Shift': 'INFO'
    }
    
    # --- CHECK 1: Target class balance ---
    print("\n" + "-"*40)
    print("CHECK 1: Target class balance")
    print("-"*40)
    target_counts = df[TARGET_COL].value_counts(normalize=True) * 100
    print(f"Global Target split: 1: {target_counts.get(1, 0):.2f}%, 0: {target_counts.get(0, 0):.2f}%")
    
    dates = sorted(df['Date'].unique())
    folds = generate_walk_forward_folds(dates, TRAIN_DAYS, VAL_DAYS, TEST_DAYS)
    
    for i, fold in enumerate(folds[:3]):
        train_idx = (df['Date'] >= dates[fold['train'][0]]) & (df['Date'] < dates[fold['train'][1]])
        val_idx = (df['Date'] >= dates[fold['val'][0]]) & (df['Date'] < dates[fold['val'][1]])
        test_idx = (df['Date'] >= dates[fold['test'][0]]) & (df['Date'] < dates[fold['test'][1]])
        
        train_df = df[train_idx]
        val_df = df[val_idx]
        test_df = df[test_idx]
        
        splits = {'Train': train_df, 'Val': val_df, 'Test': test_df}
        print(f"\nFold {i+1}:")
        for split_name, split_df in splits.items():
            if len(split_df) > 0:
                pct_1 = (split_df[TARGET_COL] == 1).mean() * 100
                flag = "*** FLAG ***" if pct_1 < 45 or pct_1 > 55 else ""
                print(f"  {split_name}: {len(split_df)} rows | Class 1: {pct_1:.2f}% {flag}")
                if pct_1 < 45 or pct_1 > 55:
                    summary_results['CHECK 1 - Class Balance'] = 'FAIL (Imbalance > 55/45)'
            else:
                print(f"  {split_name}: 0 rows")

    # Keep a reference to Fold 1 for subsequent checks
    f1 = folds[0]
    train_idx = (df['Date'] >= dates[f1['train'][0]]) & (df['Date'] < dates[f1['train'][1]])
    val_idx = (df['Date'] >= dates[f1['val'][0]]) & (df['Date'] < dates[f1['val'][1]])
    test_idx = (df['Date'] >= dates[f1['test'][0]]) & (df['Date'] < dates[f1['test'][1]])
    
    train_df = df[train_idx]
    val_df = df[val_idx]
    test_df = df[test_idx]

    # --- CHECK 2: In-sample overfitting test ---
    print("\n" + "-"*40)
    print("CHECK 2: In-sample overfitting test (CRITICAL)")
    print("-"*40)
    
    X_train_raw = train_df[BASELINE_FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_val_raw = val_df[BASELINE_FEATURE_COLS].values
    y_val = val_df[TARGET_COL].values
    X_test_raw = test_df[BASELINE_FEATURE_COLS].values
    y_test = test_df[TARGET_COL].values
    
    # Standardize
    X_train_s, X_val_s, X_test_s, scaler = standardize_fold(X_train_raw, X_val_raw, X_test_raw)
    
    print("Training Logistic Regression on Fold 1 Train...")
    lr_model = train_logistic(X_train_s, y_train)
    lr_train_preds = lr_model.predict(X_train_s)
    lr_train_probs = lr_model.predict_proba(X_train_s)[:, 1]
    
    print("Training Random Forest on Fold 1 Train...")
    rf_model = train_random_forest(X_train_s, y_train, X_val_s, y_val)
    rf_train_preds = rf_model.predict(X_train_s)
    rf_train_probs = rf_model.predict_proba(X_train_s)[:, 1]
    
    models_res = [
        ("Logistic Regression", lr_train_preds, lr_train_probs),
        ("Random Forest", rf_train_preds, rf_train_probs)
    ]
    
    passed_overfit = False
    for name, preds, probs in models_res:
        acc = accuracy_score(y_train, preds)
        auc = roc_auc_score(y_train, probs)
        f1 = f1_score(y_train, preds)
        print(f"\n{name} In-Sample Performance:")
        print(f"  Accuracy: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")
        if acc > 0.58:
            passed_overfit = True
            
    if passed_overfit:
        print("\nVERDICT: PASS (At least one model scored > 58% in-sample accuracy)")
        summary_results['CHECK 2 - In-sample Overfit'] = 'PASS'
    else:
        print("\nVERDICT: FAIL (Models failed to score > 58% in-sample accuracy. Possible feature-target misalignment!)")
        summary_results['CHECK 2 - In-sample Overfit'] = 'FAIL'

    # --- CHECK 3: Trivial baseline comparison ---
    print("\n" + "-"*40)
    print("CHECK 3: Trivial baseline comparison (Fold 1 Test)")
    print("-"*40)
    
    lr_test_preds = lr_model.predict(X_test_s)
    rf_test_preds = rf_model.predict(X_test_s)
    lr_test_acc = accuracy_score(y_test, lr_test_preds)
    rf_test_acc = accuracy_score(y_test, rf_test_preds)
    
    print(f"Trained LR Test Accuracy: {lr_test_acc:.4f}")
    print(f"Trained RF Test Accuracy: {rf_test_acc:.4f}")
    
    ones_pred = np.ones_like(y_test)
    zeros_pred = np.zeros_like(y_test)
    np.random.seed(42)
    rand_pred = np.random.randint(0, 2, size=y_test.shape)
    
    acc_ones = accuracy_score(y_test, ones_pred)
    acc_zeros = accuracy_score(y_test, zeros_pred)
    acc_rand = accuracy_score(y_test, rand_pred)
    
    print(f"\nAll-1s Baseline Accuracy:  {acc_ones:.4f}")
    print(f"All-0s Baseline Accuracy:  {acc_zeros:.4f}")
    print(f"Random 50/50 Accuracy:     {acc_rand:.4f}")
    
    max_baseline_acc = max(acc_ones, acc_zeros, acc_rand)
    if max_baseline_acc >= max(lr_test_acc, rf_test_acc):
        print("\nWARNING: Trivial baselines match or exceed trained models on test set!")
        summary_results['CHECK 3 - Trivial Baseline'] = 'WARNING'
    else:
        print("\nPASS: Trained models beat trivial baselines on test set.")

    # --- CHECK 4: Feature-target correlation sanity check ---
    print("\n" + "-"*40)
    print("CHECK 4: Feature-target correlation (Fold 1 Train)")
    print("-"*40)
    
    corrs = []
    for col in BASELINE_FEATURE_COLS:
        if col in train_df.columns:
            r, p = pearsonr(train_df[col], train_df[TARGET_COL])
            corrs.append((col, r, p))
            
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"{'Feature':<20} | {'Pearson r':<10} | {'p-value':<10} | {'Flag'}")
    print("-" * 55)
    
    significant_features = 0
    for col, r, p in corrs:
        flag = "Potentially predictive" if abs(r) > 0.02 and p < 0.05 else ""
        if p < 0.05:
            significant_features += 1
        print(f"{col:<20} | {r:>10.4f} | {p:>10.4g} | {flag}")
        
    if significant_features == 0:
        print("\nWARNING: No features show statistically significant correlation with target — possible target construction bug!")
        summary_results['CHECK 4 - Feature Correlation'] = 'WARNING (No significant correlations)'
    else:
        print(f"\nPASS: {significant_features} features show statistically significant correlation.")

    # --- CHECK 5: Date alignment verification ---
    print("\n" + "-"*40)
    print("CHECK 5: Date alignment verification")
    print("-"*40)
    np.random.seed(42)
    random_tickers = np.random.choice(df['Ticker'].unique(), 5, replace=False)
    
    cols_to_show = ['Date', 'Ticker', 'Return_1d', TARGET_COL]
    if 'Close' in df.columns:
        cols_to_show.insert(2, 'Close')
        
    for ticker in random_tickers:
        ticker_df = df[df['Ticker'] == ticker].sort_values('Date').head(5)
        print(f"\nTicker: {ticker}")
        print(ticker_df[cols_to_show].to_string(index=False))
        
    print("\n--> PLEASE MANUALLY VERIFY: Return_1d on row t ≈ (Close[t] - Close[t-1])/Close[t-1]")
    print("--> Target on row t refers to outperformance on NEXT DATE (row t+1).")

    # --- CHECK 6: Walk-forward fold date continuity ---
    print("\n" + "-"*40)
    print("CHECK 6: Walk-forward fold date continuity")
    print("-"*40)
    
    print(f"{'Fold':<5} | {'Train Start':<12} | {'Train End':<12} | {'Val Start':<12} | {'Val End':<12} | {'Test Start':<12} | {'Test End':<12}")
    print("-" * 95)
    
    continuity_fail = False
    for i, fold in enumerate(folds):
        tr_s, tr_e = dates[fold['train'][0]].strftime('%Y-%m-%d'), dates[fold['train'][1]-1].strftime('%Y-%m-%d')
        v_s, v_e = dates[fold['val'][0]].strftime('%Y-%m-%d'), dates[fold['val'][1]-1].strftime('%Y-%m-%d')
        ts_s, ts_e = dates[fold['test'][0]].strftime('%Y-%m-%d'), dates[fold['test'][1]-1].strftime('%Y-%m-%d')
        
        print(f"{i+1:<5} | {tr_s:<12} | {tr_e:<12} | {v_s:<12} | {v_e:<12} | {ts_s:<12} | {ts_e:<12}")
        
        # Check overlaps
        if dates[fold['train'][1]-1] >= dates[fold['val'][0]]:
            print(f"  --> OVERLAP DETECTED Train/Val in Fold {i+1}")
            continuity_fail = True
        if dates[fold['val'][1]-1] >= dates[fold['test'][0]]:
            print(f"  --> OVERLAP DETECTED Val/Test in Fold {i+1}")
            continuity_fail = True
            
        # Check gaps between folds (optional depending on rolling logic, but we can verify test end vs next train start)
        if i > 0:
            prev_ts_e = dates[folds[i-1]['test'][1]-1]
            if prev_ts_e > dates[fold['train'][0]]:
                # overlapping folds are fine, this is a rolling window
                pass
                
    if continuity_fail:
        summary_results['CHECK 6 - Fold Continuity'] = 'FAIL (Overlaps detected)'
    else:
        summary_results['CHECK 6 - Fold Continuity'] = 'PASS'

    # --- CHECK 7: Scaler leakage check ---
    print("\n" + "-"*40)
    print("CHECK 7: Scaler distribution shift check (Fold 1)")
    print("-"*40)
    
    std_scaler = StandardScaler()
    std_scaler.fit(X_train_raw)
    
    train_means = std_scaler.mean_
    train_stds = np.sqrt(std_scaler.var_)
    
    test_means = np.nanmean(X_test_raw, axis=0)
    
    shifted_features = 0
    for idx, col in enumerate(BASELINE_FEATURE_COLS):
        # difference in units of train standard deviation
        diff_std = abs(test_means[idx] - train_means[idx]) / (train_stds[idx] + 1e-10)
        if diff_std > 0.5:
            shifted_features += 1
            print(f"  {col}: Test mean deviates by {diff_std:.2f} std from Train mean")
            
    if shifted_features > len(BASELINE_FEATURE_COLS) / 2:
        print(f"\nWARNING: Distribution shift detected in {shifted_features}/{len(BASELINE_FEATURE_COLS)} features (>0.5 std dev).")
        print("This is expected but worth noting, it might explain poor out-of-sample generalization.")
        summary_results['CHECK 7 - Distribution Shift'] = 'WARNING (High Shift)'
    else:
        print("\nINFO: No massive distribution shift detected between Train and Test.")

    # --- SUMMARY ---
    print("\n==============================")
    print("===   DIAGNOSTIC SUMMARY   ===")
    print("==============================")
    for key, val in summary_results.items():
        print(f"{key:<30}: {val}")
    print("==============================\n")


def run_fold_analysis():
    print("Loading data from data/processed/features.csv...")
    df = pd.read_csv('data/processed/features.csv', parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    dates = sorted(df['Date'].unique())
    folds = generate_walk_forward_folds(dates, TRAIN_DAYS, VAL_DAYS, TEST_DAYS)
    
    results = []
    
    print("\nStarting fold analysis...")
    for fold in folds:
        fold_idx = fold['fold']
        start_dt = pd.to_datetime(dates[fold['test'][0]])
        end_dt = pd.to_datetime(dates[fold['test'][1] - 1])
        
        print(f"Processing fold {fold_idx}/{len(folds)} (test: {start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')})...")
        
        train_dates = dates[fold['train'][0]:fold['train'][1]]
        val_dates = dates[fold['val'][0]:fold['val'][1]]
        test_dates = dates[fold['test'][0]:fold['test'][1]]
        
        df_tr = df[df['Date'].isin(train_dates)]
        df_v = df[df['Date'].isin(val_dates)]
        df_te = df[df['Date'].isin(test_dates)]
        
        X_tr = df_tr[BASELINE_FEATURE_COLS].values
        y_tr = df_tr[TARGET_COL].values
        
        X_v = df_v[BASELINE_FEATURE_COLS].values
        y_v = df_v[TARGET_COL].values
        
        X_te = df_te[BASELINE_FEATURE_COLS].values
        y_te = df_te[TARGET_COL].values
        
        X_tr_s, X_v_s, X_te_s, _ = standardize_fold(X_tr, X_v, X_te)
        
        # Train
        lr_model = train_logistic(X_tr_s, y_tr)
        rf_model = train_random_forest(X_tr_s, y_tr, X_v_s, y_v)
        
        # Test eval
        lr_prob = lr_model.predict_proba(X_te_s)[:, 1]
        lr_pred = lr_model.predict(X_te_s)
        
        rf_prob = rf_model.predict_proba(X_te_s)[:, 1]
        rf_pred = rf_model.predict(X_te_s)
        
        lr_acc = accuracy_score(y_te, lr_pred)
        lr_auc = roc_auc_score(y_te, lr_prob)
        
        rf_acc = accuracy_score(y_te, rf_pred)
        rf_auc = roc_auc_score(y_te, rf_prob)
        
        # Flag Note
        note = ""
        def overlaps(s1, e1, s2, e2):
            return (s1 <= e2) and (e1 >= s2)
            
        if overlaps(start_dt, end_dt, pd.to_datetime("2020-02-20"), pd.to_datetime("2020-04-30")):
            note = "← COVID crash"
        elif overlaps(start_dt, end_dt, pd.to_datetime("2022-01-01"), pd.to_datetime("2022-12-31")):
            note = "← 2022 bear"
        elif start_dt >= pd.to_datetime("2023-01-01"):
            note = "← AI rally"
            
        results.append({
            'fold': fold_idx,
            'start': start_dt.strftime('%Y-%m-%d'),
            'end': end_dt.strftime('%Y-%m-%d'),
            'start_dt': start_dt,
            'end_dt': end_dt,
            'lr_acc': lr_acc,
            'lr_auc': lr_auc,
            'rf_acc': rf_acc,
            'rf_auc': rf_auc,
            'note': note
        })
        
    print("\n=== PER-FOLD TEST PERFORMANCE ===")
    print("Fold | Test Period              | LR Acc | LR AUC | RF Acc | RF AUC | Note")
    print("-" * 81)
    for r in results:
        print(f"{r['fold']:<4} | {r['start']} → {r['end']} | {r['lr_acc']:.4f} | {r['lr_auc']:.4f} | {r['rf_acc']:.4f} | {r['rf_auc']:.4f} | {r['note']}")

    lr_aucs = [r['lr_auc'] for r in results]
    rf_aucs = [r['rf_auc'] for r in results]
    
    print(f"\nLR AUC Overall: {np.mean(lr_aucs):.4f} ± {np.std(lr_aucs):.4f}")
    print(f"RF AUC Overall: {np.mean(rf_aucs):.4f} ± {np.std(rf_aucs):.4f}")
    
    best_lr = max(results, key=lambda x: x['lr_auc'])
    worst_lr = min(results, key=lambda x: x['lr_auc'])
    print(f"\nBest LR  Fold: {best_lr['fold']} ({best_lr['start']} → {best_lr['end']}), AUC: {best_lr['lr_auc']:.4f}")
    print(f"Worst LR Fold: {worst_lr['fold']} ({worst_lr['start']} → {worst_lr['end']}), AUC: {worst_lr['lr_auc']:.4f}")
    
    best_rf = max(results, key=lambda x: x['rf_auc'])
    worst_rf = min(results, key=lambda x: x['rf_auc'])
    print(f"Best RF  Fold: {best_rf['fold']} ({best_rf['start']} → {best_rf['end']}), AUC: {best_rf['rf_auc']:.4f}")
    print(f"Worst RF Fold: {worst_rf['fold']} ({worst_rf['start']} → {worst_rf['end']}), AUC: {worst_rf['rf_auc']:.4f}")
    
    lr_meaningful = sum(1 for a in lr_aucs if a > 0.52)
    rf_meaningful = sum(1 for a in rf_aucs if a > 0.52)
    print(f"\nFolds with LR AUC > 0.52: {lr_meaningful}/{len(results)}")
    print(f"Folds with RF AUC > 0.52: {rf_meaningful}/{len(results)}")
    
    def regime_matcher(r):
        st = r['start_dt']
        en = r['end_dt']
        regimes = []
        if st >= pd.to_datetime("2017-01-01") and en <= pd.to_datetime("2020-02-19"):
            regimes.append("Pre-COVID")
        if overlaps(st, en, pd.to_datetime("2020-02-20"), pd.to_datetime("2020-09-14")):
            regimes.append("COVID")
        if st >= pd.to_datetime("2020-09-15") and en <= pd.to_datetime("2021-12-31"):
            regimes.append("Post-COVID")
        if st >= pd.to_datetime("2022-01-01") and en <= pd.to_datetime("2022-12-31"):
            regimes.append("Bear/Rate")
        if st >= pd.to_datetime("2023-01-01") and en <= pd.to_datetime("2024-12-31"):
            regimes.append("AI Rally")
        return regimes

    regime_results = {"Pre-COVID": [], "COVID": [], "Post-COVID": [], "Bear/Rate": [], "AI Rally": []}
    for r in results:
        for reg in regime_matcher(r):
            regime_results[reg].append(r)
            
    print("\n=== REGIME SUMMARY ===")
    print("Regime        | Folds | LR AUC (mean) | RF AUC (mean)")
    print("-" * 55)
    for reg in ["Pre-COVID", "COVID", "Post-COVID", "Bear/Rate", "AI Rally"]:
        folds_in_reg = regime_results[reg]
        num_folds = len(folds_in_reg)
        if num_folds > 0:
            mean_lr = np.mean([x['lr_auc'] for x in folds_in_reg])
            mean_rf = np.mean([x['rf_auc'] for x in folds_in_reg])
            print(f"{reg:<13} | {num_folds:^5} |    {mean_lr:.4f}     |    {mean_rf:.4f}")
        else:
            print(f"{reg:<13} | {num_folds:^5} |    N/A        |    N/A")


def run_signal_direction_check():
    print("\n" + "="*60)
    print("           SIGNAL DIRECTION CHECK")
    print("="*60)
    
    # Load data
    data_path = 'data/processed/features.csv'
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}")
        return
        
    df = pd.read_csv(data_path, parse_dates=['Date'])
    df = df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    
    # Get fold 1
    dates = sorted(df['Date'].unique())
    folds = generate_walk_forward_folds(dates, TRAIN_DAYS, VAL_DAYS, TEST_DAYS)
    f1 = folds[0]
    
    train_idx = (df['Date'] >= dates[f1['train'][0]]) & (df['Date'] < dates[f1['train'][1]])
    val_idx = (df['Date'] >= dates[f1['val'][0]]) & (df['Date'] < dates[f1['val'][1]])
    test_idx = (df['Date'] >= dates[f1['test'][0]]) & (df['Date'] < dates[f1['test'][1]])
    
    train_df = df[train_idx].copy()
    val_df = df[val_idx].copy()
    test_df = df[test_idx].copy()
    
    # --- PART A ---
    print("\n=== FEATURE DIRECTION CHECK (Raw Data, Fold 1 Train) ===")
    print(f"{'Feature':<15} | {'Top-10 Avg Target':<17} | {'Bot-10 Avg Target':<17} | {'Direction'}")
    print("-" * 75)
    
    mean_rev_features = set()
    momentum_features = set()
    
    for feature in BASELINE_FEATURE_COLS:
        top10_targets = []
        bot10_targets = []
        for date, group in train_df.groupby('Date'):
            if len(group) < K_STOCKS * 2: continue
            g_sorted = group.sort_values(feature, ascending=False)
            top10_targets.extend(g_sorted.head(K_STOCKS)[TARGET_COL].dropna().tolist())
            bot10_targets.extend(g_sorted.tail(K_STOCKS)[TARGET_COL].dropna().tolist())
            
        top_avg = np.mean(top10_targets) if top10_targets else np.nan
        bot_avg = np.mean(bot10_targets) if bot10_targets else np.nan
        
        diff = bot_avg - top_avg
        if diff > 0.005:
            flag = "Bot > Top ← MEAN REVERSION"
            mean_rev_features.add(feature)
        elif diff < -0.005:
            flag = "Top > Bot ← MOMENTUM"
            momentum_features.add(feature)
        else:
            flag = "No clear direction"
            
        print(f"{feature:<15} | {top_avg:^17.4f} | {bot_avg:^17.4f} | {flag}")

    # --- PART B ---
    X_train_raw = train_df[BASELINE_FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_val_raw = val_df[BASELINE_FEATURE_COLS].values
    y_val = val_df[TARGET_COL].values
    X_test_raw = test_df[BASELINE_FEATURE_COLS].values
    y_test = test_df[TARGET_COL].values
    
    X_train_s, X_val_s, X_test_s, scaler = standardize_fold(X_train_raw, X_val_raw, X_test_raw)
    
    print("\nTraining LR and RF on Fold 1...")
    lr_model = train_logistic(X_train_s, y_train)
    rf_model = train_random_forest(X_train_s, y_train, X_val_s, y_val)
    
    test_df['Prob_LR'] = lr_model.predict_proba(X_test_s)[:, 1]
    test_df['Prob_RF'] = rf_model.predict_proba(X_test_s)[:, 1]
    
    for model_name, prob_col in [('LR', 'Prob_LR'), ('RF', 'Prob_RF')]:
        print(f"\n=== PORTFOLIO FEATURE PROFILE (Fold 1 Test, {model_name}) ===")
        print(f"{'Feature':<15} | {'Long Avg':<15} | {'Short Avg':<15} | {'Ratio L/S':<10} | {'Status'}")
        print("-" * 80)
        
        long_features = {f: [] for f in BASELINE_FEATURE_COLS}
        short_features = {f: [] for f in BASELINE_FEATURE_COLS}
        
        for date, group in test_df.groupby('Date'):
            if len(group) < K_STOCKS * 2: continue
            g_sorted = group.sort_values(prob_col, ascending=False)
            long_st = g_sorted.head(K_STOCKS)
            short_st = g_sorted.tail(K_STOCKS)
            
            for f in BASELINE_FEATURE_COLS:
                long_features[f].extend(long_st[f].tolist())
                short_features[f].extend(short_st[f].tolist())
                
        for feature in BASELINE_FEATURE_COLS:
            l_avg = np.mean(long_features[feature])
            s_avg = np.mean(short_features[feature])
            ratio = l_avg / s_avg if s_avg != 0 else np.nan
            
            if feature in mean_rev_features:
                flag = "← CORRECT (mean rev)" if l_avg < s_avg else "← INVERTED"
            elif feature in momentum_features:
                flag = "← CORRECT (momentum)" if l_avg > s_avg else "← INVERTED"
            else:
                flag = ""
                
            print(f"{feature:<15} | {l_avg:>14.4f}  | {s_avg:>14.4f}  | {ratio:>9.3f}  | {flag}")

    # --- PART C: Smoothed ---
    test_df_lr_sm = smooth_probabilities(test_df[['Date', 'Ticker', 'Prob_LR']], 'Prob_LR', alpha=SIGNAL_SMOOTH_ALPHA).sort_values(['Date', 'Ticker']).reset_index(drop=True)
    test_df_rf_sm = smooth_probabilities(test_df[['Date', 'Ticker', 'Prob_RF']], 'Prob_RF', alpha=SIGNAL_SMOOTH_ALPHA).sort_values(['Date', 'Ticker']).reset_index(drop=True)
    
    test_df['Prob_LR_Smooth'] = test_df_lr_sm['Prob_LR_Smooth']
    test_df['Prob_RF_Smooth'] = test_df_rf_sm['Prob_RF_Smooth']
    
    sig_lr_sm = generate_signals(test_df, k=K_STOCKS, prob_col='Prob_LR_Smooth', use_cross_sectional_z=True, confidence_threshold=0.0)
    sig_rf_sm = generate_signals(test_df, k=K_STOCKS, prob_col='Prob_RF_Smooth', use_cross_sectional_z=True, confidence_threshold=0.0)
    
    for model_name, sig_df in [('LR', sig_lr_sm), ('RF', sig_rf_sm)]:
        print(f"\n=== AFTER SMOOTHING + Z-SCORING ({model_name}) ===")
        print(f"{'Feature':<15} | {'Long Avg':<15} | {'Short Avg':<15} | {'Ratio L/S':<10} | {'Status'}")
        print("-" * 80)
        
        long_features = {f: [] for f in BASELINE_FEATURE_COLS}
        short_features = {f: [] for f in BASELINE_FEATURE_COLS}
        
        for date, group in sig_df.groupby('Date'):
            long_st = group[group['Signal'] == 'Long']
            short_st = group[group['Signal'] == 'Short']
            if len(long_st) == 0 or len(short_st) == 0: continue
            
            for f in BASELINE_FEATURE_COLS:
                long_features[f].extend(long_st[f].tolist())
                short_features[f].extend(short_st[f].tolist())
                
        for feature in BASELINE_FEATURE_COLS:
            l_avg = np.mean(long_features[feature])
            s_avg = np.mean(short_features[feature])
            ratio = l_avg / s_avg if s_avg != 0 else np.nan
            
            if feature in mean_rev_features:
                flag = "← CORRECT (mean rev)" if l_avg < s_avg else "← INVERTED"
            elif feature in momentum_features:
                flag = "← CORRECT (momentum)" if l_avg > s_avg else "← INVERTED"
            else:
                flag = ""
                
            print(f"{feature:<15} | {l_avg:>14.4f}  | {s_avg:>14.4f}  | {ratio:>9.3f}  | {flag}")

    # --- PART D ---
    print("\n=== PORTFOLIO RETURN SUMMARY (Fold 1 Test) ===")
    print(f"{'Model':<5} | {'Signal Type':<16} | {'Mean Daily Return':<17} | {'Ann. Return':<11} | {'Win Rate'}")
    print("-" * 75)
    
    sig_lr_raw = generate_signals(test_df, k=K_STOCKS, prob_col='Prob_LR', use_cross_sectional_z=False, confidence_threshold=0.0)
    sig_rf_raw = generate_signals(test_df, k=K_STOCKS, prob_col='Prob_RF', use_cross_sectional_z=False, confidence_threshold=0.0)
    
    results_to_print = []
    
    def calc_returns(signals):
        daily = []
        for date, group in signals.groupby('Date'):
            l_ret = group[group['Signal'] == 'Long']['Return_NextDay'].mean()
            s_ret = group[group['Signal'] == 'Short']['Return_NextDay'].mean()
            if pd.notna(l_ret) and pd.notna(s_ret):
                daily.append(l_ret - s_ret)
        if not daily: return 0, 0, 0
        mean_r = np.mean(daily)
        ann_r = (1 + mean_r)**252 - 1
        win_rt = np.mean(np.array(daily) > 0) * 100
        return mean_r, ann_r, win_rt

    for name, s_type, sigs in [
        ('LR', 'Raw probs', sig_lr_raw),
        ('LR', 'Smoothed+ZScore', sig_lr_sm),
        ('RF', 'Raw probs', sig_rf_raw),
        ('RF', 'Smoothed+ZScore', sig_rf_sm)
    ]:
        m, a, w = calc_returns(sigs)
        print(f"{name:<5} | {s_type:<16} | {m:>16.5f}  | {a*100:>9.1f}% | {w:>6.1f}%")
        results_to_print.append({'model': name, 'type': s_type, 'mean_ret': m})
        
    lr_raw_ret = results_to_print[0]['mean_ret']
    lr_sm_ret  = results_to_print[1]['mean_ret']
    rf_raw_ret = results_to_print[2]['mean_ret']
    rf_sm_ret  = results_to_print[3]['mean_ret']
    
    if lr_sm_ret < lr_raw_ret or rf_sm_ret < rf_raw_ret:
        print("\nWARNING: Smoothing/z-scoring is degrading signal quality.")
        print("The EMA or z-score step may be attenuating the mean-reversion signal.")
        print("Consider reducing alpha or removing z-scoring.")
        
    if lr_raw_ret < 0 or rf_raw_ret < 0:
        print("\nWARNING: Model predictions are inverting the mean-reversion signal.")
        print("Long portfolio has HIGHER feature values than short portfolio despite")
        print("negative feature-target correlations. Check probability calibration.")


if __name__ == '__main__':
    # run_diagnostics()
     run_fold_analysis()
    #run_signal_direction_check()


