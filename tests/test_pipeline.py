import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backtest.signals import smooth_probabilities, apply_holding_period_constraint, generate_signals
from pipeline.features import compute_sector_context_features, compute_market_context_features

class TestExecutionLayer(unittest.TestCase):
    
    def setUp(self):
        dates = pd.date_range('2023-01-01', periods=10)
        self.df = pd.DataFrame({
            'Date': dates.repeat(2),
            'Ticker': ['A', 'B'] * 10,
            'Prob_ENS': np.random.rand(20)
        })
        
    def test_ema_smoothing(self):
        # 1. EMA Smoothing Correctness
        df = self.df.copy()
        df.loc[(df['Date'] == '2023-01-01') & (df['Ticker'] == 'A'), 'Prob_ENS'] = 0.8
        df.loc[(df['Date'] == '2023-01-02') & (df['Ticker'] == 'A'), 'Prob_ENS'] = 0.5
        
        smoothed = smooth_probabilities(df, 'Prob_ENS', alpha=0.3)
        smooth_A_day1 = smoothed.loc[(smoothed['Date'] == '2023-01-01') & (smoothed['Ticker'] == 'A'), 'Prob_ENS_Smooth'].values[0]
        smooth_A_day2 = smoothed.loc[(smoothed['Date'] == '2023-01-02') & (smoothed['Ticker'] == 'A'), 'Prob_ENS_Smooth'].values[0]
        
        self.assertAlmostEqual(smooth_A_day1, 0.8)
        self.assertAlmostEqual(smooth_A_day2, 0.3 * 0.5 + 0.7 * 0.8)

    def test_min_hold_behavior(self):
        # 2. Min Hold Logic
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=5),
            'Ticker': ['A']*5,
            'Signal': ['Long', 'Hold', 'Short', 'Hold', 'Short']
        })
        res = apply_holding_period_constraint(df, min_hold_days=3)
        signals = res['Signal'].tolist()
        
        # Day 1: Long (entry)
        # Day 2: Expected Hold, but min hold is 3, so stays Long
        # Day 3: Expected Short, but min hold is 3, so stays Long (entry at day 0 + 2 < 3)
        # Day 4: It's Day 3 index (idx 3 >= 3). We try to Hold -> Hold it is allowed. Wait, we exited to Hold.
        
        expected = ['Long', 'Long', 'Long', 'Hold', 'Short']
        self.assertEqual(signals, expected)

    def test_confidence_threshold_behavior(self):
        # 3. Confidence Threshold
        df = pd.DataFrame({
            'Date': [pd.Timestamp('2023-01-01')] * 6,
            'Ticker': ['A', 'B', 'C', 'D', 'E', 'F'],
            'Prob_LR': [0.9, 0.53, 0.51, 0.49, 0.47, 0.8], # Doesn't matter because prob_col is used or average
            'Prob_RF': [0.9, 0.53, 0.51, 0.49, 0.47, 0.8],
            'Prob_XGB': [0.9, 0.53, 0.51, 0.49, 0.47, 0.8],
            'Prob_LSTM_A': [0.9, 0.53, 0.51, 0.49, 0.47, 0.8],
            'Prob_LSTM_B': [0.9, 0.53, 0.51, 0.49, 0.47, 0.1],
            'Return_NextDay': [0] * 6,
            'Target': [0] * 6
        })
        
        # average Prob_ENS for F is ~ (0.8*4 + 0.1)/5 = 0.66
        # A: 0.9, B: 0.53, C: 0.51, D: 0.49, E: 0.47, F: 0.66
        
        res = generate_signals(df, k=3, confidence_threshold=0.05, use_cross_sectional_z=False)
        # A is 0.9 >= 0.55 -> Long
        # F is 0.66 >= 0.55 -> Long
        # B is 0.53 < 0.55 -> Hold
        
        # E is 0.47 <= 0.45? False.
        # D is 0.49 <= 0.45? False.
        
        # Since use_cross_sectional_z is False, we check raw probs.
        sigs = dict(zip(res['Ticker'], res['Signal']))
        self.assertEqual(sigs['A'], 'Long')
        self.assertEqual(sigs['F'], 'Long')
        self.assertEqual(sigs['B'], 'Hold')
        self.assertEqual(sigs['C'], 'Hold')
        self.assertEqual(sigs['D'], 'Hold')
        self.assertEqual(sigs['E'], 'Hold')

class TestFeatureEngineering(unittest.TestCase):
    def test_sector_leave_one_out(self):
        # 4. Sector Leave-One-Out
        df = pd.DataFrame({
            'Date': [pd.Timestamp('2023-01-01')] * 3,
            'Ticker': ['A', 'B', 'C'],
            'Return_1d': [0.01, 0.03, 0.05],
            'Return_5d': [0.02, 0.04, 0.06]
        })
        sector_map = {'A': 'Tech', 'B': 'Tech', 'C': 'Tech'}
        
        res = compute_sector_context_features(df, sector_map)
        
        # A's sector mean should be mean of B & C = (0.03 + 0.05)/2 = 0.04
        # B's sector mean should be mean of A & C = 0.03
        # C's sector mean should be mean of A & B = 0.02
        
        vals = dict(zip(res['Ticker'], res['Sector_Return_1d']))
        self.assertAlmostEqual(vals['A'], 0.04)
        self.assertAlmostEqual(vals['B'], 0.03)
        self.assertAlmostEqual(vals['C'], 0.02)
        
    def test_feature_causality(self):
        # 5. Feature Causality - No shift backward
        dates = pd.date_range('2023-01-01', periods=100)
        df_market = pd.DataFrame({
            'Date': dates,
            'Ticker': ['SPY'] * 100,
            'Return_1d': np.random.randn(100) * 0.01,
            'Return_5d': np.random.randn(100) * 0.01
        })
        
        res = compute_market_context_features(df_market)
        # Test causality of rolling std
        # Volatility at day 20 should only depend on day 1..20
        # If we change day 21, day 20 vol shouldn't change
        vol_20 = res.loc[20, 'Market_Vol_20d']
        
        df_market_shifted = df_market.copy()
        df_market_shifted.loc[21:, 'Return_1d'] += 0.5
        
        res_shifted = compute_market_context_features(df_market_shifted)
        vol_20_shifted = res_shifted.loc[20, 'Market_Vol_20d']
        
        self.assertEqual(vol_20, vol_20_shifted)

if __name__ == '__main__':
    unittest.main()
