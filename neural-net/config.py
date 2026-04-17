"""
Shared configuration for the 3-model temperature prediction ensemble.
"""
import sys, os

# Allow imports from parent kalshi-weather directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date

# ── Data splits ──────────────────────────────────────────────────────
TRAIN_START = date(2022, 1, 1)
TRAIN_END = date(2024, 12, 31)
VAL_START = date(2025, 1, 1)
VAL_END = date(2025, 12, 31)
TEST_START = date(2026, 1, 1)
TEST_END = date(2026, 4, 16)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

# ── Cities ───────────────────────────────────────────────────────────
# Mirrors parent config.CITIES: ticker -> (name, tz, lat, lon)
CITIES = {
    "KXHIGHNY":     ("New York",        "US/Eastern",   40.71, -74.01),
    "KXHIGHCHI":    ("Chicago",         "US/Central",   41.88, -87.63),
    "KXHIGHMIA":    ("Miami",           "US/Eastern",   25.76, -80.19),
    "KXHIGHTBOS":   ("Boston",          "US/Eastern",   42.36, -71.06),
    "KXHIGHLAX":    ("Los Angeles",     "US/Pacific",   34.05, -118.24),
    "KXHIGHAUS":    ("Austin",          "US/Central",   30.27, -97.74),
    "KXHIGHTSFO":   ("San Francisco",   "US/Pacific",   37.77, -122.42),
    "KXHIGHTDAL":   ("Dallas",          "US/Central",   32.78, -96.80),
    "KXHIGHPHIL":   ("Philadelphia",    "US/Eastern",   39.95, -75.17),
    "KXHIGHTPHX":   ("Phoenix",         "US/Arizona",   33.45, -112.07),
    "KXHIGHTOKC":   ("Oklahoma City",   "US/Central",   35.47, -97.52),
    "KXHIGHDEN":    ("Denver",          "US/Mountain",  39.74, -104.98),
    "KXHIGHTDC":    ("Washington DC",   "US/Eastern",   38.91, -77.04),
    "KXHIGHTSATX":  ("San Antonio",     "US/Central",   29.42, -98.49),
    "KXHIGHTHOU":   ("Houston",         "US/Central",   29.76, -95.37),
    "KXHIGHTMIN":   ("Minneapolis",     "US/Central",   44.98, -93.27),
    "KXHIGHTATL":   ("Atlanta",         "US/Eastern",   33.75, -84.39),
    "KXHIGHTSEA":   ("Seattle",         "US/Pacific",   47.61, -122.33),
    "KXHIGHTLV":    ("Las Vegas",       "US/Pacific",   36.17, -115.14),
    "KXHIGHTNOLA":  ("New Orleans",     "US/Central",   29.95, -90.07),
}

# ── NWS station IDs (ACIS/FAA codes) ────────────────────────────────
# These are the official NWS stations whose daily recorded high is the
# settlement source for Kalshi weather markets.
NWS_STATIONS = {
    "KXHIGHNY":    "NYC",   # Central Park
    "KXHIGHCHI":   "ORD",   # O'Hare
    "KXHIGHMIA":   "MIA",   # Miami Intl
    "KXHIGHTBOS":  "BOS",   # Logan
    "KXHIGHLAX":   "LAX",   # LAX
    "KXHIGHAUS":   "AUS",   # Austin-Bergstrom
    "KXHIGHTSFO":  "SFO",   # SFO
    "KXHIGHTDAL":  "DFW",   # DFW
    "KXHIGHPHIL":  "PHL",   # PHL
    "KXHIGHTPHX":  "PHX",   # Sky Harbor
    "KXHIGHTOKC":  "OKC",   # Will Rogers
    "KXHIGHDEN":   "DEN",   # DIA
    "KXHIGHTDC":   "DCA",   # Reagan National
    "KXHIGHTSATX": "SAT",   # SAT
    "KXHIGHTHOU":  "IAH",   # George Bush Intercontinental
    "KXHIGHTMIN":  "MSP",   # MSP
    "KXHIGHTATL":  "ATL",   # Hartsfield-Jackson
    "KXHIGHTSEA":  "SEA",   # Sea-Tac
    "KXHIGHTLV":   "LAS",   # Harry Reid
    "KXHIGHTNOLA": "MSY",   # Louis Armstrong
}

CITY_TICKERS = list(CITIES.keys())
TICKER_TO_IDX = {t: i for i, t in enumerate(CITY_TICKERS)}
N_CITIES = len(CITY_TICKERS)

# ── Neighbor mapping (user-specified) ────────────────────────────────
NEIGHBORS = {
    "KXHIGHNY":    ["KXHIGHTBOS", "KXHIGHPHIL", "KXHIGHTDC", "KXHIGHCHI"],
    "KXHIGHCHI":   ["KXHIGHTMIN", "KXHIGHDEN", "KXHIGHTOKC", "KXHIGHTDAL"],
    "KXHIGHMIA":   ["KXHIGHTATL", "KXHIGHTNOLA", "KXHIGHTHOU"],
    "KXHIGHTBOS":  ["KXHIGHNY", "KXHIGHPHIL", "KXHIGHTDC"],
    "KXHIGHLAX":   ["KXHIGHTSFO", "KXHIGHTLV", "KXHIGHTPHX"],
    "KXHIGHAUS":   ["KXHIGHTDAL", "KXHIGHTSATX", "KXHIGHTHOU", "KXHIGHTOKC"],
    "KXHIGHTSFO":  ["KXHIGHLAX", "KXHIGHTSEA", "KXHIGHTLV"],
    "KXHIGHTDAL":  ["KXHIGHAUS", "KXHIGHTOKC", "KXHIGHTHOU", "KXHIGHDEN"],
    "KXHIGHPHIL":  ["KXHIGHNY", "KXHIGHTDC", "KXHIGHTBOS", "KXHIGHCHI"],
    "KXHIGHTPHX":  ["KXHIGHTLV", "KXHIGHLAX", "KXHIGHDEN"],
    "KXHIGHTOKC":  ["KXHIGHTDAL", "KXHIGHDEN", "KXHIGHCHI"],
    "KXHIGHDEN":   ["KXHIGHTPHX", "KXHIGHTLV", "KXHIGHTDAL", "KXHIGHTMIN"],
    "KXHIGHTDC":   ["KXHIGHPHIL", "KXHIGHTATL", "KXHIGHNY"],
    "KXHIGHTSATX": ["KXHIGHAUS", "KXHIGHTHOU", "KXHIGHTDAL"],
    "KXHIGHTHOU":  ["KXHIGHAUS", "KXHIGHTSATX", "KXHIGHTNOLA", "KXHIGHTDAL"],
    "KXHIGHTMIN":  ["KXHIGHCHI", "KXHIGHDEN", "KXHIGHTOKC"],
    "KXHIGHTATL":  ["KXHIGHMIA", "KXHIGHTNOLA", "KXHIGHTDC", "KXHIGHTHOU"],
    "KXHIGHTSEA":  ["KXHIGHTSFO", "KXHIGHDEN"],
    "KXHIGHTLV":   ["KXHIGHLAX", "KXHIGHTPHX", "KXHIGHTSFO", "KXHIGHDEN"],
    "KXHIGHTNOLA": ["KXHIGHTHOU", "KXHIGHTATL", "KXHIGHMIA", "KXHIGHTDAL"],
}

# ── City static metadata ─────────────────────────────────────────────
# elevation_ft, coastal (bool), desert (bool), continentality (0-1)
CITY_META = {
    "KXHIGHNY":    {"elevation_ft": 33,   "coastal": True,  "desert": False, "continentality": 0.40},
    "KXHIGHCHI":   {"elevation_ft": 594,  "coastal": False, "desert": False, "continentality": 0.70},
    "KXHIGHMIA":   {"elevation_ft": 6,    "coastal": True,  "desert": False, "continentality": 0.10},
    "KXHIGHTBOS":  {"elevation_ft": 20,   "coastal": True,  "desert": False, "continentality": 0.35},
    "KXHIGHLAX":   {"elevation_ft": 285,  "coastal": True,  "desert": False, "continentality": 0.15},
    "KXHIGHAUS":   {"elevation_ft": 489,  "coastal": False, "desert": False, "continentality": 0.55},
    "KXHIGHTSFO":  {"elevation_ft": 52,   "coastal": True,  "desert": False, "continentality": 0.10},
    "KXHIGHTDAL":  {"elevation_ft": 430,  "coastal": False, "desert": False, "continentality": 0.65},
    "KXHIGHPHIL":  {"elevation_ft": 39,   "coastal": False, "desert": False, "continentality": 0.45},
    "KXHIGHTPHX":  {"elevation_ft": 1086, "coastal": False, "desert": True,  "continentality": 0.90},
    "KXHIGHTOKC":  {"elevation_ft": 1201, "coastal": False, "desert": False, "continentality": 0.75},
    "KXHIGHDEN":   {"elevation_ft": 5280, "coastal": False, "desert": False, "continentality": 0.80},
    "KXHIGHTDC":   {"elevation_ft": 25,   "coastal": False, "desert": False, "continentality": 0.40},
    "KXHIGHTSATX": {"elevation_ft": 650,  "coastal": False, "desert": False, "continentality": 0.60},
    "KXHIGHTHOU":  {"elevation_ft": 80,   "coastal": True,  "desert": False, "continentality": 0.30},
    "KXHIGHTMIN":  {"elevation_ft": 830,  "coastal": False, "desert": False, "continentality": 0.85},
    "KXHIGHTATL":  {"elevation_ft": 1050, "coastal": False, "desert": False, "continentality": 0.50},
    "KXHIGHTSEA":  {"elevation_ft": 175,  "coastal": True,  "desert": False, "continentality": 0.20},
    "KXHIGHTLV":   {"elevation_ft": 2001, "coastal": False, "desert": True,  "continentality": 0.85},
    "KXHIGHTNOLA": {"elevation_ft": 3,    "coastal": True,  "desert": False, "continentality": 0.20},
}

# ── Model hyperparameters ────────────────────────────────────────────
MODEL1_HP = {
    "hidden_dims": [128, 64, 32],
    "dropout": [0.2, 0.15, 0.0],
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 200,
    "batch_size": 256,
    "patience": 20,
    "city_embed_dim": 8,
}

MODEL2_HP = {
    "hidden_dims": [256, 128, 64],
    "dropout": [0.25, 0.2, 0.1],
    "lr": 5e-4,
    "weight_decay": 5e-4,
    "epochs": 300,
    "batch_size": 256,
    "patience": 20,
    "city_embed_dim": 8,
}

MODEL3_HP = {
    "hidden_dims": [192, 96, 48],
    "dropout": [0.3, 0.2, 0.0],
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 200,
    "batch_size": 256,
    "patience": 20,
    "city_embed_dim": 8,
}

ENSEMBLE_HP = {
    "context_dim": 8,
    "hidden_dim": 16,
    "lr": 5e-4,
    "epochs": 100,
    "batch_size": 256,
    "patience": 15,
}

# Kalshi fee model
TAKER_FEE_RATE = 0.07
