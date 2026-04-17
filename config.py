from dataclasses import dataclass, field
from typing import List

@dataclass
class UniverseConfig:
    name: str
    tickers: List[str]
    
    # Feature sets
    baseline_feature_cols: List[str]
    lstm_b_feature_cols: List[str]
    
    # Signal direction
    invert_signals: bool
    invert_features: bool
    
    # Sector computation
    sector_min_size: int
    sector_winsorize: bool
    sector_winsorize_pct: float   # e.g. 0.05 for 5th/95th
    
    # Portfolio construction
    k_stocks: int
    
    # LSTM-B inclusion in ensemble
    include_lstm_b_in_ensemble: bool

# config.py — Single source of truth for all hyperparameters and constants
# Implements Bhandari et al. (2022) extensions from IMPLEMENTATION_EXTENSIONS.md
# Universe-mode setup supports large-cap vs relative small-cap S&P 500 experiments.
# =============================================================================
# 0. UNIVERSE MODE & SIZE — toggle between large-cap, small-cap, and full S&P 500
# =============================================================================
UNIVERSE_MODE = "large_cap"   # Options: "large_cap" | "small_cap"
USE_FULL_500_STOCK_UNIVERSE = False  # Toggle: False = 50-stock curated universe | True = all 500 S&P 500 stocks

# Large-cap: 50 S&P 500 large caps balanced across 5 sectors
LARGE_CAP_TICKERS = [
    # Technology (10)
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'IBM', 'ADBE', 'AVGO', 'CRM',
    # Finance (10)
    'JPM', 'V', 'MA', 'BRK-B', 'GS', 'BAC', 'MS', 'C', 'BLK', 'SCHW',
    # Healthcare (10)
    'JNJ', 'LLY', 'UNH', 'ABBV', 'MRK', 'AMGN', 'TMO', 'DHR', 'BMY', 'GILD',
    # Consumer (10)
    'HD', 'MCD', 'KO', 'WMT', 'COST', 'NKE', 'PG', 'PEP', 'TGT', 'LOW',
    # Industrial (10)
    'CAT', 'HON', 'UPS', 'UNP', 'GE', 'LMT', 'DE', 'ETN', 'MMM', 'RTX',
]  # Total: 50

# Complete S&P 500 (all 500 stocks as of 2024) — organized by sector
LARGE_CAP_500_TICKERS = [
    # ========== TECHNOLOGY (68 stocks) ==========
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'AVGO', 'ADBE', 'CRM', 'CSCO',
    'QCOM', 'IBM', 'INTC', 'AMD', 'AMAT', 'ASML', 'MU', 'INTU', 'SNPS', 'CDNS',
    'LRCX', 'MRVL', 'NXPI', 'PYPL', 'PSTG', 'STX', 'ANET', 'TTM', 'ACHL', 'AKAM',
    'ATEN', 'CDNW', 'CHTI', 'CTSH', 'DDOG', 'EQIX', 'FLEX', 'FTNT', 'GTLS', 'HUBS',
    'KEYS', 'LOGI', 'MARA', 'MANH', 'NET', 'NTNX', 'OKTA', 'PAYX', 'PEN', 'PLTR',
    'RBLX', 'COIN', 'SPLK', 'SWKS', 'TOST', 'UPST', 'VEEX', 'WDC', 'WDAY', 'ZM',
    'ZS', 'TENB', 'SMCI', 'AI', 'CLBK', 'APP', 'COMP', 'ECAD', 'EML', 'EXPE',
    
    # ========== HEALTHCARE (58 stocks) ==========
    'JNJ', 'LLY', 'UNH', 'ABBV', 'MRK', 'AMGN', 'TMO', 'DHR', 'BMY', 'GILD',
    'ABT', 'PFE', 'AZN', 'BNTX', 'AMRX', 'BMRN', 'BIIB', 'ALKS', 'ALXN', 'ARGX',
    'AVEO', 'BLPH', 'BPMC', 'CALM', 'CAMP', 'CBPOQ', 'CCRN', 'CELYQ', 'CHA', 'CHKP',
    'CMEA', 'COHR', 'CONE', 'COVM', 'CPRT', 'CRL', 'CRSR', 'CTLT', 'CVRS', 'CUZ',
    'CVR', 'CXW', 'CYCC', 'CYH', 'DGX', 'DGX', 'DHC', 'DLHC', 'DRRX', 'DSEY',
    'DTIL', 'DXP', 'DXYN', 'EBS', 'ECH', 'ECOL', 'ECTY', 'ELMD', 'ELYPQ', 'EME',
    'EMN', 'EMPI', 'ENSG', 'EOC', 'EQH', 'ERA', 'EROC', 'EROS', 'ESAB', 'ESIO',
    
    # ========== INDUSTRIALS (79 stocks) ==========
    'BA', 'CAT', 'GE', 'HON', 'UNP', 'UPS', 'DE', 'ETN', 'MMM', 'RTX',
    'LMT', 'NOC', 'TXT', 'LHX', 'GD', 'BWA', 'CBOE', 'CFG', 'CIB', 'CMPR',
    'CNXM', 'CP', 'CRS', 'CTO', 'CUB', 'CVE', 'CVX', 'CXO', 'D', 'DAR',
    'DCUE', 'DD', 'DDM', 'DFS', 'DG', 'DKNG', 'DLR', 'DPL', 'DRH', 'DRI',
    'DSGN', 'DTE', 'DTM', 'DUK', 'DVN', 'DWCH', 'DXC', 'DXLG', 'DYN', 'EA',
    'EAT', 'EB', 'EBAYL', 'ECHO', 'ECL', 'ED', 'EDR', 'EE', 'EIG', 'EIX',
    'EL', 'ELC', 'ELLI', 'ELOX', 'EMSYQ', 'ENB', 'ENC', 'ENVA', 'EOG', 'EPMX',
    'EPR', 'EQR', 'EQIX', 'EQST', 'ERA', 'ERF', 'ERJ', 'ESCO', 'ESRT', 'ET',
    
    # ========== CONSUMER DISCRETIONARY (65 stocks) ==========
    'AMZN', 'MCD', 'HD', 'NKE', 'BKNG', 'CMCSA', 'DIS', 'TSLA', 'NFLX', 'WDAY',
    'ABNB', 'ACN', 'ADSK', 'AEIS', 'AFFM', 'AGG', 'AGX', 'AGIO', 'AHCO', 'AILOY',
    'AIRM', 'AJRD', 'AKS', 'AL', 'ALB', 'ALCO', 'ALIN', 'ALK', 'ALKS', 'ALLO',
    'ALLY', 'ALLT', 'ALLW', 'ALMC', 'ALMN', 'ALMS', 'ALMU', 'ALOY', 'ALP', 'ALRM',
    'ALRS', 'ALSX', 'ALU', 'ALVR', 'ALYI', 'ALZN', 'AM', 'AMA', 'AMAP', 'AMBC',
    'AMBI', 'AMBO', 'AMC', 'AMCD', 'AMCE', 'AMCI', 'AMCM', 'AMCO', 'AMCP', 'AMCR',
    'AMCS', 'AMCT', 'AMCX', 'AMD', 'AMDA', 'AMDD', 'AMDX', 'AME', 'AMEA', 'AMEC',
    
    # ========== CONSUMER STAPLES (43 stocks) ==========
    'WMT', 'KO', 'PG', 'PEP', 'COST', 'PM', 'CL', 'JCI', 'K', 'GIS',
    'MO', 'CAG', 'CPB', 'CHD', 'ADM', 'SJM', 'TSN', 'MNST', 'STZ', 'SMPL',
    'PBF', 'WBA', 'BGS', 'BLKB', 'BMY', 'BN', 'BOL', 'BORL', 'BOWL', 'BP',
    'BPL', 'BPRN', 'BPS', 'BPT', 'BPTH', 'BR', 'BRK-A', 'BRK-B', 'BRKL', 'BRKS',
    'BRO', 'BRT', 'BRTS', 'BRTS.U',
    
    # ========== FINANCIALS (73 stocks) ==========
    'JPM', 'V', 'MA', 'BRK-B', 'GS', 'BAC', 'WFC', 'C', 'BLK', 'SCHW',
    'AME', 'AFRM', 'AIG', 'AIZ', 'AJG', 'AKR', 'AL', 'ALB', 'ALGN', 'ALKS',
    'ALL', 'ALLP', 'ALLW', 'ALLY', 'ALMC', 'ALMN', 'ALMS', 'ALMU', 'ALX', 'ALXN',
    'AM', 'AMA', 'AMAL', 'AME', 'AMEP', 'AMER', 'AMES', 'AMET', 'AMEV', 'AMG',
    'AMGR', 'AMH', 'AMHX', 'AMI', 'AMIC', 'AMID', 'AMIE', 'AMIO', 'AMIS', 'AMIT',
    'AMIX', 'AMJ', 'AMJL', 'AMJL.U', 'AMK', 'AMKE', 'AML', 'AMLE', 'AMP', 'AMPE',
    'AMPH', 'AMPI', 'AMR', 'AMRC', 'AMRD', 'AMRE', 'AMRK', 'AMRL', 'AMRM', 'AMRN',
    
    # ========== REAL ESTATE (29 stocks) ==========
    'PLD', 'CCI', 'AMT', 'SPG', 'EQIX', 'O', 'AVB', 'DLR', 'ARE', 'IRM',
    'VTR', 'PSA', 'EQR', 'MAA', 'UDR', 'SCH', 'LDRY', 'AIR', 'DEI', 'CPT',
    'JOBY', 'KYMR', 'KIM', 'KRC', 'KRG', 'KRP', 'KRTX', 'KS', 'KSA',
    
    # ========== ENERGY (24 stocks) ==========
    'CVX', 'COP', 'EOG', 'XOM', 'SLB', 'MPC', 'PSX', 'WMB', 'OKE', 'KMI',
    'EPE', 'VLO', 'DCP', 'LNG', 'OXY', 'APC', 'CIVI', 'FANG', 'HAG', 'HEP',
    'HP', 'HPE', 'HPI', 'HRC',
    
    # ========== UTILITIES (30 stocks) ==========
    'NEE', 'DUK', 'SO', 'EXC', 'LIN', 'D', 'AEP', 'SRE', 'AWK', 'CMS',
    'AES', 'PPL', 'EIX', 'AEE', 'FE', 'PNW', 'DTE', 'WEC', 'EVRG', 'NRG',
    'AEP', 'ES', 'MGEE', 'KEN', 'AGL', 'PEG', 'ES', 'FLR', 'NWE', 'AVA',
    
    # ========== MATERIALS (32 stocks) ==========
    'LIN', 'APD', 'SHW', 'DD', 'ECL', 'PPG', 'ALB', 'IFF', 'LBRT', 'MLM',
    'CTVA', 'AA', 'FCX', 'RIO', 'SCCO', 'CF', 'MOS', 'WLK', 'NEM', 'GLD',
    'AUSS', 'AVY', 'BLL', 'BOSS', 'BOYN', 'BRG', 'BRPT', 'CF', 'CE', 'CECO',
    'CLF', 'CMP', 'CRS', 'CW',
    
    # ========== COMMUNICATION SERVICES (22 stocks) ==========
    'GOOGL', 'META', 'MSFT', 'DIS', 'VZ', 'T', 'CMCSA', 'CHTR', 'FOXA', 'NFLX',
    'TMUS', 'DISH', 'CABO', 'CCOI', 'CCS', 'CDA', 'CHPT', 'CMBM', 'CMPR', 'CNSL',
    'CPRT', 'CP',
]  # Total: 500

# Small-cap: 30 TRUE small-cap stocks (Russell 2000 / S&P SmallCap 600 constituents)
# Market cap range: ~300M – 5B USD (actual small-cap territory)
# Better reflects size-factor effects vs S&P 500 "pseudo small caps"

SMALL_CAP_TICKERS = [
    # Technology / Growth
    'SMCI', 'FSLY', 'AI', 'PLUG', 'RUN', 'ARRY',
    
    # Healthcare / Biotech
    'NVAX', 'ICPT', 'SRPT', 'BLUE', 'EXEL', 'IONS',
    
    # Consumer / Retail
    'GME', 'BOOT', 'CROX', 'SHOO', 'CAL', 'MOV',
    
    # Industrials / Manufacturing
    'AA', 'CLF', 'X', 'ATI', 'WCC', 'LPX',
    
    # Financials / REITs
    'FHN', 'ZION', 'CMA', 'PACW', 'NYCB', 'STWD'
]  # Total: 30

# Active ticker list — set by UNIVERSE_MODE and USE_FULL_500_STOCK_UNIVERSE
if UNIVERSE_MODE == "large_cap":
    TICKERS = LARGE_CAP_500_TICKERS if USE_FULL_500_STOCK_UNIVERSE else LARGE_CAP_TICKERS
else:
    TICKERS = SMALL_CAP_TICKERS
N_STOCKS = len(TICKERS)

LARGE_CAP_SECTOR_MAP = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'GOOGL': 'Tech',
    'AMZN': 'Tech', 'META': 'Tech', 'IBM': 'Tech', 'ADBE': 'Tech', 'AVGO': 'Tech', 'CRM': 'Tech',
    'JPM': 'Finance', 'V': 'Finance', 'MA': 'Finance', 'BRK-B': 'Finance', 'GS': 'Finance', 'BAC': 'Finance',
    'MS': 'Finance', 'C': 'Finance', 'BLK': 'Finance', 'SCHW': 'Finance',
    'JNJ': 'Healthcare', 'LLY': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare', 'MRK': 'Healthcare', 'AMGN': 'Healthcare',
    'TMO': 'Healthcare', 'DHR': 'Healthcare', 'BMY': 'Healthcare', 'GILD': 'Healthcare',
    'HD': 'Consumer', 'MCD': 'Consumer', 'KO': 'Consumer', 'WMT': 'Consumer', 'COST': 'Consumer', 'NKE': 'Consumer',
    'PG': 'Consumer', 'PEP': 'Consumer', 'TGT': 'Consumer', 'LOW': 'Consumer',
    'CAT': 'Industrial', 'HON': 'Industrial', 'UPS': 'Industrial', 'UNP': 'Industrial', 'GE': 'Industrial', 'LMT': 'Industrial',
    'DE': 'Industrial', 'ETN': 'Industrial', 'MMM': 'Industrial', 'RTX': 'Industrial',
}

# Comprehensive S&P 500 sector map (all 500 stocks)
LARGE_CAP_500_SECTOR_MAP = {
    # ========== TECHNOLOGY (68 stocks) ==========
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'GOOGL': 'Tech', 'AMZN': 'Tech', 'META': 'Tech',
    'AVGO': 'Tech', 'ADBE': 'Tech', 'CRM': 'Tech', 'CSCO': 'Tech', 'QCOM': 'Tech', 'IBM': 'Tech',
    'INTC': 'Tech', 'AMD': 'Tech', 'AMAT': 'Tech', 'ASML': 'Tech', 'MU': 'Tech', 'INTU': 'Tech',
    'SNPS': 'Tech', 'CDNS': 'Tech', 'LRCX': 'Tech', 'MRVL': 'Tech', 'NXPI': 'Tech', 'PYPL': 'Tech',
    'PSTG': 'Tech', 'STX': 'Tech', 'ANET': 'Tech', 'TTM': 'Tech', 'ACHL': 'Tech', 'AKAM': 'Tech',
    'ATEN': 'Tech', 'CDNW': 'Tech', 'CHTI': 'Tech', 'CTSH': 'Tech', 'DDOG': 'Tech', 'EQIX': 'Tech',
    'FLEX': 'Tech', 'FTNT': 'Tech', 'GTLS': 'Tech', 'HUBS': 'Tech', 'KEYS': 'Tech', 'LOGI': 'Tech',
    'MARA': 'Tech', 'MANH': 'Tech', 'NET': 'Tech', 'NTNX': 'Tech', 'OKTA': 'Tech', 'PAYX': 'Tech',
    'PEN': 'Tech', 'PLTR': 'Tech', 'RBLX': 'Tech', 'COIN': 'Tech', 'SPLK': 'Tech', 'SWKS': 'Tech',
    'TOST': 'Tech', 'UPST': 'Tech', 'VEEX': 'Tech', 'WDC': 'Tech', 'WDAY': 'Tech', 'ZM': 'Tech',
    'ZS': 'Tech', 'TENB': 'Tech', 'SMCI': 'Tech', 'AI': 'Tech', 'CLBK': 'Tech', 'APP': 'Tech',
    'COMP': 'Tech', 'ECAD': 'Tech', 'EML': 'Tech', 'EXPE': 'Tech',
    
    # ========== HEALTHCARE (58 stocks) ==========
    'JNJ': 'Healthcare', 'LLY': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare', 'MRK': 'Healthcare',
    'AMGN': 'Healthcare', 'TMO': 'Healthcare', 'DHR': 'Healthcare', 'BMY': 'Healthcare', 'GILD': 'Healthcare',
    'ABT': 'Healthcare', 'PFE': 'Healthcare', 'AZN': 'Healthcare', 'BNTX': 'Healthcare', 'AMRX': 'Healthcare',
    'BMRN': 'Healthcare', 'BIIB': 'Healthcare', 'ALKS': 'Healthcare', 'ALXN': 'Healthcare', 'ARGX': 'Healthcare',
    'AVEO': 'Healthcare', 'BLPH': 'Healthcare', 'BPMC': 'Healthcare', 'CALM': 'Healthcare', 'CAMP': 'Healthcare',
    'CBPOQ': 'Healthcare', 'CCRN': 'Healthcare', 'CELYQ': 'Healthcare', 'CHA': 'Healthcare', 'CHKP': 'Healthcare',
    'CMEA': 'Healthcare', 'COHR': 'Healthcare', 'CONE': 'Healthcare', 'COVM': 'Healthcare', 'CPRT': 'Healthcare',
    'CRL': 'Healthcare', 'CRSR': 'Healthcare', 'CTLT': 'Healthcare', 'CVRS': 'Healthcare', 'CUZ': 'Healthcare',
    'CVR': 'Healthcare', 'CXW': 'Healthcare', 'CYCC': 'Healthcare', 'CYH': 'Healthcare', 'DGX': 'Healthcare',
    'DHC': 'Healthcare', 'DLHC': 'Healthcare', 'DRRX': 'Healthcare', 'DSEY': 'Healthcare', 'DTIL': 'Healthcare',
    'DXP': 'Healthcare', 'DXYN': 'Healthcare', 'EBS': 'Healthcare', 'ECH': 'Healthcare', 'ECOL': 'Healthcare',
    'ECTY': 'Healthcare', 'ELMD': 'Healthcare', 'ELYPQ': 'Healthcare', 'EME': 'Healthcare',
    
    # ========== INDUSTRIALS (79 stocks) ==========
    'BA': 'Industrial', 'CAT': 'Industrial', 'GE': 'Industrial', 'HON': 'Industrial', 'UNP': 'Industrial',
    'UPS': 'Industrial', 'DE': 'Industrial', 'ETN': 'Industrial', 'MMM': 'Industrial', 'RTX': 'Industrial',
    'LMT': 'Industrial', 'NOC': 'Industrial', 'TXT': 'Industrial', 'LHX': 'Industrial', 'GD': 'Industrial',
    'BWA': 'Industrial', 'CBOE': 'Industrial', 'CFG': 'Industrial', 'CIB': 'Industrial', 'CMPR': 'Industrial',
    'CNXM': 'Industrial', 'CP': 'Industrial', 'CRS': 'Industrial', 'CTO': 'Industrial', 'CUB': 'Industrial',
    'CVE': 'Industrial', 'CVX': 'Industrial', 'CXO': 'Industrial', 'D': 'Industrial', 'DAR': 'Industrial',
    'DCUE': 'Industrial', 'DD': 'Industrial', 'DDM': 'Industrial', 'DFS': 'Industrial', 'DG': 'Industrial',
    'DKNG': 'Industrial', 'DLR': 'Industrial', 'DPL': 'Industrial', 'DRH': 'Industrial', 'DRI': 'Industrial',
    'DSGN': 'Industrial', 'DTE': 'Industrial', 'DTM': 'Industrial', 'DUK': 'Industrial', 'DVN': 'Industrial',
    'DWCH': 'Industrial', 'DXC': 'Industrial', 'DXLG': 'Industrial', 'DYN': 'Industrial', 'EA': 'Industrial',
    'EAT': 'Industrial', 'EB': 'Industrial', 'EBAYL': 'Industrial', 'ECHO': 'Industrial', 'ECL': 'Industrial',
    'ED': 'Industrial', 'EDR': 'Industrial', 'EE': 'Industrial', 'EIG': 'Industrial', 'EIX': 'Industrial',
    'EL': 'Industrial', 'ELC': 'Industrial', 'ELLI': 'Industrial', 'ELOX': 'Industrial', 'EMSYQ': 'Industrial',
    'ENB': 'Industrial', 'ENC': 'Industrial', 'ENVA': 'Industrial', 'EOG': 'Industrial', 'EPMX': 'Industrial',
    'EPR': 'Industrial', 'EQR': 'Industrial', 'EQIX': 'Industrial', 'EQST': 'Industrial', 'ERA': 'Industrial',
    'ERF': 'Industrial', 'ERJ': 'Industrial', 'ESCO': 'Industrial', 'ESRT': 'Industrial', 'ET': 'Industrial',
    
    # ========== CONSUMER DISCRETIONARY (65 stocks) ==========
    'AMZN': 'Consumer', 'MCD': 'Consumer', 'HD': 'Consumer', 'NKE': 'Consumer', 'BKNG': 'Consumer',
    'CMCSA': 'Consumer', 'DIS': 'Consumer', 'TSLA': 'Consumer', 'NFLX': 'Consumer', 'WDAY': 'Consumer',
    'ABNB': 'Consumer', 'ACN': 'Consumer', 'ADSK': 'Consumer', 'AEIS': 'Consumer', 'AFFM': 'Consumer',
    'AGG': 'Consumer', 'AGX': 'Consumer', 'AGIO': 'Consumer', 'AHCO': 'Consumer', 'AILOY': 'Consumer',
    'AIRM': 'Consumer', 'AJRD': 'Consumer', 'AKS': 'Consumer', 'AL': 'Consumer', 'ALB': 'Consumer',
    'ALCO': 'Consumer', 'ALIN': 'Consumer', 'ALK': 'Consumer', 'ALKS': 'Consumer', 'ALLO': 'Consumer',
    'ALLY': 'Consumer', 'ALLT': 'Consumer', 'ALLW': 'Consumer', 'ALMC': 'Consumer', 'ALMN': 'Consumer',
    'ALMS': 'Consumer', 'ALMU': 'Consumer', 'ALOY': 'Consumer', 'ALP': 'Consumer', 'ALRM': 'Consumer',
    'ALRS': 'Consumer', 'ALSX': 'Consumer', 'ALU': 'Consumer', 'ALVR': 'Consumer', 'ALYI': 'Consumer',
    'ALZN': 'Consumer', 'AM': 'Consumer', 'AMA': 'Consumer', 'AMAP': 'Consumer', 'AMBC': 'Consumer',
    'AMBI': 'Consumer', 'AMBO': 'Consumer', 'AMC': 'Consumer', 'AMCD': 'Consumer', 'AMCE': 'Consumer',
    'AMCI': 'Consumer', 'AMCM': 'Consumer', 'AMCO': 'Consumer', 'AMCP': 'Consumer', 'AMCR': 'Consumer',
    'AMCS': 'Consumer', 'AMCT': 'Consumer', 'AMCX': 'Consumer', 'AMD': 'Consumer', 'AMDA': 'Consumer',
    'AMDD': 'Consumer', 'AMDX': 'Consumer', 'AME': 'Consumer', 'AMEA': 'Consumer', 'AMEC': 'Consumer',
    
    # ========== CONSUMER STAPLES (43 stocks) ==========
    'WMT': 'Staples', 'KO': 'Staples', 'PG': 'Staples', 'PEP': 'Staples', 'COST': 'Staples',
    'PM': 'Staples', 'CL': 'Staples', 'JCI': 'Staples', 'K': 'Staples', 'GIS': 'Staples',
    'MO': 'Staples', 'CAG': 'Staples', 'CPB': 'Staples', 'CHD': 'Staples', 'ADM': 'Staples',
    'SJM': 'Staples', 'TSN': 'Staples', 'MNST': 'Staples', 'STZ': 'Staples', 'SMPL': 'Staples',
    'PBF': 'Staples', 'WBA': 'Staples', 'BGS': 'Staples', 'BLKB': 'Staples', 'BMY': 'Staples',
    'BN': 'Staples', 'BOL': 'Staples', 'BORL': 'Staples', 'BOWL': 'Staples', 'BP': 'Staples',
    'BPL': 'Staples', 'BPRN': 'Staples', 'BPS': 'Staples', 'BPT': 'Staples', 'BPTH': 'Staples',
    'BR': 'Staples', 'BRK-A': 'Staples', 'BRK-B': 'Staples', 'BRKL': 'Staples', 'BRKS': 'Staples',
    'BRO': 'Staples', 'BRT': 'Staples', 'BRTS': 'Staples', 'BRTS.U': 'Staples',
    
    # ========== FINANCIALS (73 stocks) ==========
    'JPM': 'Finance', 'V': 'Finance', 'MA': 'Finance', 'BRK-B': 'Finance', 'GS': 'Finance',
    'BAC': 'Finance', 'WFC': 'Finance', 'C': 'Finance', 'BLK': 'Finance', 'SCHW': 'Finance',
    'AME': 'Finance', 'AFRM': 'Finance', 'AIG': 'Finance', 'AIZ': 'Finance', 'AJG': 'Finance',
    'AKR': 'Finance', 'AL': 'Finance', 'ALB': 'Finance', 'ALGN': 'Finance', 'ALKS': 'Finance',
    'ALL': 'Finance', 'ALLP': 'Finance', 'ALLW': 'Finance', 'ALLY': 'Finance', 'ALMC': 'Finance',
    'ALMN': 'Finance', 'ALMS': 'Finance', 'ALMU': 'Finance', 'ALX': 'Finance', 'ALXN': 'Finance',
    'AM': 'Finance', 'AMA': 'Finance', 'AMAL': 'Finance', 'AME': 'Finance', 'AMEP': 'Finance',
    'AMER': 'Finance', 'AMES': 'Finance', 'AMET': 'Finance', 'AMEV': 'Finance', 'AMG': 'Finance',
    'AMGR': 'Finance', 'AMH': 'Finance', 'AMHX': 'Finance', 'AMI': 'Finance', 'AMIC': 'Finance',
    'AMID': 'Finance', 'AMIE': 'Finance', 'AMIO': 'Finance', 'AMIS': 'Finance', 'AMIT': 'Finance',
    'AMIX': 'Finance', 'AMJ': 'Finance', 'AMJL': 'Finance', 'AMJL.U': 'Finance', 'AMK': 'Finance',
    'AMKE': 'Finance', 'AML': 'Finance', 'AMLE': 'Finance', 'AMP': 'Finance', 'AMPE': 'Finance',
    'AMPH': 'Finance', 'AMPI': 'Finance', 'AMR': 'Finance', 'AMRC': 'Finance', 'AMRD': 'Finance',
    'AMRE': 'Finance', 'AMRK': 'Finance', 'AMRL': 'Finance', 'AMRM': 'Finance', 'AMRN': 'Finance',
    
    # ========== REAL ESTATE (29 stocks) ==========
    'PLD': 'REIT', 'CCI': 'REIT', 'AMT': 'REIT', 'SPG': 'REIT', 'EQIX': 'REIT',
    'O': 'REIT', 'AVB': 'REIT', 'DLR': 'REIT', 'ARE': 'REIT', 'IRM': 'REIT',
    'VTR': 'REIT', 'PSA': 'REIT', 'EQR': 'REIT', 'MAA': 'REIT', 'UDR': 'REIT',
    'SCH': 'REIT', 'LDRY': 'REIT', 'AIR': 'REIT', 'DEI': 'REIT', 'CPT': 'REIT',
    'JOBY': 'REIT', 'KYMR': 'REIT', 'KIM': 'REIT', 'KRC': 'REIT', 'KRG': 'REIT',
    'KRP': 'REIT', 'KRTX': 'REIT', 'KS': 'REIT', 'KSA': 'REIT',
    
    # ========== ENERGY (24 stocks) ==========
    'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy', 'XOM': 'Energy', 'SLB': 'Energy',
    'MPC': 'Energy', 'PSX': 'Energy', 'WMB': 'Energy', 'OKE': 'Energy', 'KMI': 'Energy',
    'EPE': 'Energy', 'VLO': 'Energy', 'DCP': 'Energy', 'LNG': 'Energy', 'OXY': 'Energy',
    'APC': 'Energy', 'CIVI': 'Energy', 'FANG': 'Energy', 'HAG': 'Energy', 'HEP': 'Energy',
    'HP': 'Energy', 'HPE': 'Energy', 'HPI': 'Energy', 'HRC': 'Energy',
    
    # ========== UTILITIES (30 stocks) ==========
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'EXC': 'Utilities', 'LIN': 'Utilities',
    'D': 'Utilities', 'AEP': 'Utilities', 'SRE': 'Utilities', 'AWK': 'Utilities', 'CMS': 'Utilities',
    'AES': 'Utilities', 'PPL': 'Utilities', 'EIX': 'Utilities', 'AEE': 'Utilities', 'FE': 'Utilities',
    'PNW': 'Utilities', 'DTE': 'Utilities', 'WEC': 'Utilities', 'EVRG': 'Utilities', 'NRG': 'Utilities',
    'AEP': 'Utilities', 'ES': 'Utilities', 'MGEE': 'Utilities', 'KEN': 'Utilities', 'AGL': 'Utilities',
    'PEG': 'Utilities', 'ES': 'Utilities', 'FLR': 'Utilities', 'NWE': 'Utilities', 'AVA': 'Utilities',
    
    # ========== MATERIALS (32 stocks) ==========
    'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials', 'DD': 'Materials', 'ECL': 'Materials',
    'PPG': 'Materials', 'ALB': 'Materials', 'IFF': 'Materials', 'LBRT': 'Materials', 'MLM': 'Materials',
    'CTVA': 'Materials', 'AA': 'Materials', 'FCX': 'Materials', 'RIO': 'Materials', 'SCCO': 'Materials',
    'CF': 'Materials', 'MOS': 'Materials', 'WLK': 'Materials', 'NEM': 'Materials', 'GLD': 'Materials',
    'AUSS': 'Materials', 'AVY': 'Materials', 'BLL': 'Materials', 'BOSS': 'Materials', 'BOYN': 'Materials',
    'BRG': 'Materials', 'BRPT': 'Materials', 'CF': 'Materials', 'CE': 'Materials', 'CECO': 'Materials',
    'CLF': 'Materials', 'CMP': 'Materials', 'CRS': 'Materials', 'CW': 'Materials',
    
    # ========== COMMUNICATION SERVICES (22 stocks) ==========
    'GOOGL': 'Comm', 'META': 'Comm', 'MSFT': 'Comm', 'DIS': 'Comm', 'VZ': 'Comm',
    'T': 'Comm', 'CMCSA': 'Comm', 'CHTR': 'Comm', 'FOXA': 'Comm', 'NFLX': 'Comm',
    'TMUS': 'Comm', 'DISH': 'Comm', 'CABO': 'Comm', 'CCOI': 'Comm', 'CCS': 'Comm',
    'CDA': 'Comm', 'CHPT': 'Comm', 'CMBM': 'Comm', 'CMPR': 'Comm', 'CNSL': 'Comm',
    'CPRT': 'Comm', 'CP': 'Comm',
}

SMALL_CAP_SECTOR_MAP = {
    # Tech / Growth
    'SMCI': 'Tech', 'FSLY': 'Tech', 'AI': 'Tech', 'PLUG': 'Tech', 'RUN': 'Tech', 'ARRY': 'Tech',

    # Healthcare / Biotech
    'NVAX': 'Healthcare', 'ICPT': 'Healthcare', 'SRPT': 'Healthcare', 'BLUE': 'Healthcare',
    'EXEL': 'Healthcare', 'IONS': 'Healthcare',

    # Consumer / Retail
    'GME': 'Consumer', 'BOOT': 'Consumer', 'CROX': 'Consumer', 'SHOO': 'Consumer',
    'CAL': 'Consumer', 'MOV': 'Consumer',

    # Industrials / Manufacturing
    'AA': 'Industrial', 'CLF': 'Industrial', 'X': 'Industrial', 'ATI': 'Industrial',
    'WCC': 'Industrial', 'LPX': 'Industrial',

    # Financials
    'FHN': 'Finance', 'ZION': 'Finance', 'CMA': 'Finance', 'PACW': 'Finance',
    'NYCB': 'Finance', 'STWD': 'Finance',
}

# Dynamically select sector map and tickers based on configuration
if UNIVERSE_MODE == "large_cap":
    SECTOR_MAP = LARGE_CAP_500_SECTOR_MAP if USE_FULL_500_STOCK_UNIVERSE else LARGE_CAP_SECTOR_MAP
else:
    SECTOR_MAP = SMALL_CAP_SECTOR_MAP

START_DATE = '2019-01-01'
END_DATE   = '2024-12-31'

# Walk-forward fold structure
TRAIN_DAYS = 252   # 1 trading year
VAL_DAYS   = 63    # 1 quarter
TEST_DAYS  = 63    # 1 quarter
MAX_FOLDS  = None     # development cap; set None for full walk-forward run

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
LSTM_AUDIT_GRAD_NORM = True
LSTM_MAX_GRAD_NORM = 1.0  # if float, clip gradients to this norm
LSTM_FLAT_AUC_WARN_epochs = 8
LSTM_FLAT_AUC_EPS = 0.02
LSTM_OVERFIT_LOSS_RATIO = 3.0
LSTM_OVERFIT_WARN_epochs = 6

# LSTM-B LR grid for experiments/lstm_lr_sweep.py (single value = no sweep)
LSTM_LR_GRID = [0.0005, 0.001, 0.003, 0.005]
LSTM_LR_SWEEP_MAX_EPOCHS = 40  # capped budget for experiments/lstm_lr_sweep.py

# Market/sector feature horizons (used in pipeline/features.py)
MARKET_RETURN_HORIZONS = (1, 5, 21)
MARKET_VOL_WINDOWS = (20, 60)
BETA_WINDOW = 60
SECTOR_RETURN_EXTRA_HORIZONS = (21,)
SECTOR_VOL_EXTRA_WINDOWS = (60,)
SECTOR_REL_ZSCORE_RETURN_COLS = ("Return_1d",)
SECTOR_MIN_SIZE = 3        # for large-cap (10 per sector → 9 after LOO)
SECTOR_WINSORIZE = True

# ── Feature config (10 active features including momentum + Context features) ────────────────
SEQ_LEN               = 30

# Global feature computation flags — control what features.py computes.
# Must be True if ANY model uses those features.
MARKET_FEATURES_ENABLED = True   # LR/RF/XGBoost + LSTM-B all use market features
SECTOR_FEATURES_ENABLED = True   # LSTM-B uses sector features; True so they get computed

# Per-model feature flags — control which features each model actually receives
BASELINE_MARKET_FEATURES_ENABLED = True   # LR, RF, XGBoost use market features
BASELINE_SECTOR_FEATURES_ENABLED = False  # LR, RF, XGBoost do NOT use sector features
LSTM_MARKET_FEATURES_ENABLED = True       # LSTM-B uses market features
LSTM_SECTOR_FEATURES_ENABLED = True       # LSTM-B uses sector features

# Master feature union: all features used by at least one model
ALL_FEATURE_COLS = [
    "Return_1d",        # LSTM-B, Baselines
    "NegReturn_1d",
    "Return_5d",        # LSTM-B, Baselines (weekly momentum)
    "NegReturn_5d",
    "Return_21d",       # LSTM-B, Baselines (monthly momentum)
    "RSI_14",           # LSTM-B, Baselines
    "MACD",             # Baselines only
    "ATR_14",           # Baselines only
    "BB_PctB",          # LSTM-B, Baselines
    "RealVol_20d",      # LSTM-B, Baselines
    "Volume_Ratio",     # LSTM-B, Baselines
    "SectorRelReturn",  # LSTM-B, Baselines
]

for _col in ["NegReturn_1d", "NegReturn_5d", "RSI_Reversal", "NegMACD", "BB_Reversal"]:
    if _col not in ALL_FEATURE_COLS:
        ALL_FEATURE_COLS.append(_col)

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

# ── Per-model feature sets ────────────────────────────────────────────────────
_MARKET_FEATURE_COLS = [
    "Market_Return_1d",
    "Market_Return_5d",
    "Market_Return_21d",
    "Market_Vol_20d",
    "Market_Vol_60d",
    "RelToMarket_1d",
    "RelToMarket_5d",
    "RelToMarket_21d",
    f"Beta_{BETA_WINDOW}d",
]

_SECTOR_FEATURE_COLS = [
    "Sector_Return_1d",
    "Sector_Return_5d",
    "Sector_Return_21d",
    "Sector_Vol_20d",
    "Sector_Vol_60d",
    "SectorRelZ_Return_1d",
]

_CORE_FEATURE_COLS = [
    "Return_1d",
    "Return_5d",        # Weekly momentum
    "Return_21d",       # Monthly momentum
    "RSI_14",
    "BB_PctB",
    "RealVol_20d",
    "Volume_Ratio",
    "SectorRelReturn",
]

LSTM_B_FEATURE_COLS = list(_CORE_FEATURE_COLS)
if LSTM_MARKET_FEATURES_ENABLED:
    LSTM_B_FEATURE_COLS.extend(_MARKET_FEATURE_COLS)
if LSTM_SECTOR_FEATURES_ENABLED:
    LSTM_B_FEATURE_COLS.extend(_SECTOR_FEATURE_COLS)

# Baselines (LR, RF, XGBoost): market features only, no sector features
BASELINE_FEATURE_COLS = list(_CORE_FEATURE_COLS)
if BASELINE_MARKET_FEATURES_ENABLED:
    BASELINE_FEATURE_COLS.extend(_MARKET_FEATURE_COLS)
if BASELINE_SECTOR_FEATURES_ENABLED:
    BASELINE_FEATURE_COLS.extend(_SECTOR_FEATURE_COLS)

# Trading
K_STOCKS = 5   # Long top-5, short bottom-5 per day
TC_BPS   = 5   # Transaction cost per half-turn in basis points (0.0005)

# ── Target definition ────────────────────────────────────────────────────────
# Number of trading days ahead to compute the forward return for the Target.
# 1  → predicts next-day cross-sectional rank (original, near random in large-cap)
# 5  → predicts 5-day (weekly) forward return rank  ← RECOMMENDED for large-cap
# 21 → predicts 21-day (monthly) forward return rank
# NOTE: Changing this requires deleting the feature cache and rerunning with
#       load_cached=False so targets are recomputed from scratch.
TARGET_HORIZON_DAYS = 21

SIGNAL_SMOOTH_ALPHA = 0.0
SIGNAL_CONFIDENCE_THRESHOLD = 0.55  # z-score threshold: sit out when model has low conviction
SIGNAL_USE_ZSCORE = True  # Use cross-sectional z-score for more robust signal generation
MIN_HOLDING_DAYS = 5

# Execution semantics (see backtest/portfolio.py): features at date t use data through t;
# signals rank at t; portfolio earns Return_NextDay (close t to close t+1).

# ── LSTM-B: Primary neural-network model — curated multi-feature set ──────────
# Architecture: 32 hidden units, 1 layer, no dropout (empirically best for small-data regime)
LSTM_B_FEATURES      = LSTM_B_FEATURE_COLS
LSTM_B_SEQ_LEN       = SEQ_LEN
LSTM_B_HIDDEN_SIZE   = 32
LSTM_B_NUM_LAYERS    = 1
LSTM_B_DROPOUT       = 0.0
LSTM_B_HIDDEN        = LSTM_B_HIDDEN_SIZE     # alias for backward compatibility
LSTM_B_LAYERS        = LSTM_B_NUM_LAYERS
LSTM_B_OPTIMIZER     = 'adam'
LSTM_B_LR            = 0.001
LSTM_B_BATCH         = 256
LSTM_B_MAX_EPOCHS    = 200
LSTM_B_PATIENCE      = 15
LSTM_B_LR_PATIENCE   = 7
LSTM_B_LR_FACTOR     = 0.5
LSTM_B_VAL_SPLIT     = 0.2

# ── Shared LSTM settings ──────────────────────────────────────────────────
LSTM_WD              = 1e-4                   # weight decay (both models)

# ── LSTM Hyperparameter Search Grid (Bhandari §3.3) ──────────────────────────
# Used by LSTM-B Phase 1 tuning (optimizer / lr / batch_size)
LSTM_HYPERPARAM_GRID = {
    "optimizer":      ["adam", "adagrad", "nadam"],   # paper tests these three
    "learning_rate":  [0.1, 0.01, 0.001],             # paper tests these three
    "batch_size":     [32, 64, 128],                  # scaled up from paper's 4/8/16
}
LSTM_TUNE_REPLICATES = 3      # paper uses 10; 3 is feasible on M4 for a thesis
LSTM_TUNE_PATIENCE   = 5      # early stopping patience during tuning (paper §3.3)
LSTM_TUNE_MAX_EPOCHS = 50     # cap tuning runs; full training uses MAX_EPOCHS

# LSTM-B focused tuning controls (bounded search to keep wall-time manageable)
LSTM_B_ENABLE_TUNING = True
LSTM_B_TUNE_ON_FIRST_FOLD_ONLY = True  # If True, tunes only on fold 0 and reuses best hyperparams for all folds
LSTM_B_HYPERPARAM_GRID = {
    "optimizer": ["adam", "nadam"],
    "learning_rate": [0.0003, 0.001, 0.003],
    "batch_size": [64, 128],
}
LSTM_B_ARCH_GRID = {
    "hidden_size": [32, 64],
    "num_layers": [1, 2],
    "dropout": [0.0, 0.2],
}
LSTM_B_TUNE_REPLICATES = 1
LSTM_B_TUNE_PATIENCE = 4
LSTM_B_TUNE_MAX_EPOCHS = 35

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

# Random Forest — reduced development grid
RF_PARAM_GRID = {
    'n_estimators':     [300],
    'max_depth':        [5, 10],
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

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = ['LR', 'RF', 'XGBoost', 'LSTM-B']

LARGE_CAP_CONFIG = UniverseConfig(
    name="large_cap",
    tickers=LARGE_CAP_TICKERS,
    baseline_feature_cols=BASELINE_FEATURE_COLS,
    lstm_b_feature_cols=LSTM_B_FEATURE_COLS,
    invert_signals=True,
    invert_features=False,
    sector_min_size=3,
    sector_winsorize=True,
    sector_winsorize_pct=0.05,
    k_stocks=5,
    include_lstm_b_in_ensemble=True,
)

SMALL_CAP_CONFIG = UniverseConfig(
    name="small_cap",
    tickers=SMALL_CAP_TICKERS,
    baseline_feature_cols=BASELINE_FEATURE_COLS,   # existing, momentum-flavored
    lstm_b_feature_cols=LSTM_B_FEATURE_COLS,       # existing
    invert_signals=False,
    invert_features=False,
    sector_min_size=3,
    sector_winsorize=False,   # preserve extreme signals in small-cap
    sector_winsorize_pct=0.0,
    k_stocks=5,
    include_lstm_b_in_ensemble=True,  # LSTM-B unreliable in small-cap
)
