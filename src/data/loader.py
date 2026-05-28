"""
Data loader for QOIN. Reads all CSVs from DATA_DIR, standardizes schemas,
applies stationarity transforms, and returns daily-aligned DataFrames.

Per-factor transform decisions:
  gold_return      : log-return of Gold Close              (non-stationary level → return)
  usd_inr_return   : log-return of USD/INR spot            (non-stationary level → return)
  us_10y_diff      : 1st diff of US 10Y yield              (rate level → change in rate)
  us_5y_diff       : 1st diff of US 5Y yield               (rate level → change in rate)
  india_10y_diff   : 1st diff of India 10Y yield           (rate level → change in rate)
  india_5y_diff    : 1st diff of India 5Y yield            (rate level → change in rate)
  cpi_diff         : 1st diff of CPI index (monthly→daily) (price level → inflation change)
  wpi_diff         : 1st diff of WPI index (monthly→daily) (price level → inflation change)
  gdp_yoy          : Real GDP YoY growth % (quarterly→daily, approx stationary growth rate)
  trade_balance    : Trade balance USD bn (monthly→daily)  (spread, approx stationary)
  repo_rate_diff   : 1st diff of Repo Rate (monthly→daily) (rate level → policy change)
  fii_net          : FII net flows ₹ Cr (daily, from 2018) (flow, stationary-ish)
  dii_net          : DII net flows ₹ Cr (daily, from 2018) (flow, stationary-ish)
  icici_nii_qoq    : ICICI NII QoQ change ₹ Cr (quarterly, release_date = period_end + 45d)
  icici_pat_qoq    : ICICI PAT QoQ change ₹ Cr (quarterly, release_date = period_end + 45d)

Quarterly fundamentals release-lag assumption:
  No release date in the CSV. Approximated as fiscal_quarter_end + 45 calendar days.
  This is documented here and shown in the UI diagnostics.
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(os.environ.get(
    "QOIN_DATA_DIR",
    "/Users/surisettivamsikrishna/Desktop/Qoin/Cleaned data"
))

FACTOR_TRANSFORMS = {
    "gold_return":     "log-return of Gold Close",
    "usd_inr_return":  "log-return of USD/INR spot",
    "us_10y_diff":     "1st diff of US 10Y yield",
    "us_5y_diff":      "1st diff of US 5Y yield",
    "india_10y_diff":  "1st diff of India 10Y yield",
    "india_5y_diff":   "1st diff of India 5Y yield",
    "cpi_diff":        "1st diff of CPI index (monthly fwd-filled)",
    "wpi_diff":        "1st diff of WPI index (monthly fwd-filled)",
    "gdp_yoy":         "Real GDP YoY growth % (quarterly fwd-filled)",
    "trade_balance":   "Trade balance USD bn (monthly fwd-filled, spread)",
    "repo_rate_diff":  "1st diff of RBI Repo Rate (monthly fwd-filled)",
    "fii_net":         "FII net flows ₹ Cr (daily, starts 2018)",
    "dii_net":         "DII net flows ₹ Cr (daily, starts 2018)",
    "icici_nii_qoq":   "ICICI NII QoQ Δ ₹ Cr (quarterly, lag = period_end + 45d)",
    "icici_pat_qoq":   "ICICI PAT QoQ Δ ₹ Cr (quarterly, lag = period_end + 45d)",
}

_MONTH_END = {
    "jan": (1, 31), "feb": (2, 28), "mar": (3, 31), "apr": (4, 30),
    "may": (5, 31), "jun": (6, 30), "jul": (7, 31), "aug": (8, 31),
    "sep": (9, 30), "oct": (10, 31), "nov": (11, 30), "dec": (12, 31),
}


def _require(filename: str) -> Path:
    p = DATA_DIR / filename
    if not p.exists():
        raise FileNotFoundError(f"Required file missing: {p}")
    return p


def _to_daily_ffill(series: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    """Reindex a low-frequency series onto trading-day index using step-function forward-fill."""
    combined = series.reindex(series.index.union(daily_index)).sort_index().ffill()
    return combined.reindex(daily_index)


def _load_ohlcv(filename: str, name: str) -> pd.Series:
    df = pd.read_csv(_require(filename), parse_dates=["Date"], index_col="Date")
    df = df.sort_index()
    return df["Close"].rename(name)


def _load_india_bond_price(filename: str) -> pd.Series:
    """Return raw yield level series (daily, native calendar)."""
    df = pd.read_csv(_require(filename), encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"].str.strip('"'), format="%m/%d/%Y")
    df = df.set_index("Date").sort_index()
    return pd.to_numeric(df["Price"], errors="coerce")


def _parse_period_end(period_str: str) -> pd.Timestamp:
    """'Apr–Jun 2009' → Timestamp(2009-06-30). Returns NaT on failure."""
    matches = re.findall(r"([A-Za-z]{3})\s+(\d{4})", period_str)
    if not matches:
        return pd.NaT
    end_month_abbr, year_str = matches[-1]
    key = end_month_abbr.lower()[:3]
    if key not in _MONTH_END:
        return pd.NaT
    mon, day = _MONTH_END[key]
    return pd.Timestamp(int(year_str), mon, day)


def load_all_factors() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Returns:
        factors_df : daily DatetimeIndex, all factor columns (returns / diffs / levels)
        stocks_df  : daily DatetimeIndex, log-return columns (nifty50, bank_nifty, icici_bank)
        meta       : diagnostics dict
    """
    dropped: dict[str, str] = {}

    # ── Stocks (log-returns) ──────────────────────────────────────────────────
    bank_nifty_close = _load_ohlcv("bank_nifty_daily.csv", "bank_nifty")
    nifty50_close    = _load_ohlcv("nifty50_daily.csv",    "nifty50")
    icici_close      = _load_ohlcv("icici_bank_daily.csv", "icici_bank")

    stocks_close = pd.concat([nifty50_close, bank_nifty_close, icici_close], axis=1)
    stocks_df    = np.log(stocks_close).diff()

    # Master trading-day index (union of all three stock calendars)
    daily_index = pd.DatetimeIndex(
        sorted(nifty50_close.index.union(bank_nifty_close.index).union(icici_close.index).unique())
    )
    stocks_close = stocks_close.reindex(daily_index)
    # Reindex log-returns; fill any isolated NaN (data gaps / holiday mismatches) with 0
    stocks_df    = stocks_df.reindex(daily_index).fillna(0.0)

    # ── Daily macro factors ──────────────────────────────────────────────────
    # Strategy: reindex each price/level series to daily_index + ffill BEFORE
    # computing log-returns or diffs. On days when the source market was closed
    # (e.g. US holiday but India open), the return is 0 (carried price = no change).
    # This ensures no NaN gaps within the series that would break rolling z-scores.

    gold_close_raw = _load_ohlcv("gold_daily.csv", "gold")
    gold_px_daily  = _to_daily_ffill(gold_close_raw, daily_index)
    gold_return    = np.log(gold_px_daily).diff().rename("gold_return")

    usd_df         = pd.read_csv(_require("usd_inr_daily.csv"), parse_dates=["Date"], index_col="Date").sort_index()
    usd_px_daily   = _to_daily_ffill(usd_df["USD_INR"], daily_index)
    usd_inr_ret    = np.log(usd_px_daily).diff().rename("usd_inr_return")

    us_df          = pd.read_csv(_require("us_bond_yields.csv"), parse_dates=["Date"], index_col="Date").sort_index()
    us_10y_daily   = _to_daily_ffill(us_df["US_10Y"], daily_index)
    us_5y_daily    = _to_daily_ffill(us_df["US_5Y"],  daily_index)
    us_10y_diff    = us_10y_daily.diff().rename("us_10y_diff")
    us_5y_diff     = us_5y_daily.diff().rename("us_5y_diff")

    india_10y_raw  = _load_india_bond_price("India 10-Year Bond Yield Historical Data.csv")
    india_5y_raw   = _load_india_bond_price("India 5-Year Bond Yield Historical Data.csv")
    india_10y_diff = _to_daily_ffill(india_10y_raw, daily_index).diff().rename("india_10y_diff")
    india_5y_diff  = _to_daily_ffill(india_5y_raw,  daily_index).diff().rename("india_5y_diff")

    # ── Monthly macro (forward-fill to daily) ────────────────────────────────
    cpi_df   = pd.read_csv(_require("india_cpi_wpi_monthly.csv"), parse_dates=["Date"], index_col="Date").sort_index()
    cpi_diff = _to_daily_ffill(cpi_df["CPI_Index"].diff().rename("cpi_diff"), daily_index)
    wpi_diff = _to_daily_ffill(cpi_df["WPI_Index"].diff().rename("wpi_diff"), daily_index)

    gdp_df   = pd.read_csv(_require("india_gdp_quarterly.csv"), parse_dates=["Date"], index_col="Date").sort_index()
    yoy_col  = "Real_GDP_YoY_Growth_%"
    if yoy_col in gdp_df.columns and gdp_df[yoy_col].notna().any():
        gdp_raw = gdp_df[yoy_col]
    else:
        gdp_raw = gdp_df["Real_GDP_INR_Billion"].pct_change(4) * 100
    gdp_yoy  = _to_daily_ffill(gdp_raw.rename("gdp_yoy"), daily_index)

    trade_df      = pd.read_csv(_require("india_trade_data_monthly.csv"), parse_dates=["Date"], index_col="Date").sort_index()
    trade_balance = _to_daily_ffill(trade_df["Trade_Balance_Billion_USD"].rename("trade_balance"), daily_index)

    rbi_df          = pd.read_csv(_require("rbi_policy_rates.csv"), parse_dates=["Date"], index_col="Date").sort_index()
    repo_rate_diff  = _to_daily_ffill(rbi_df["Repo_Rate_%"].diff().rename("repo_rate_diff"), daily_index)

    # ── Flows (from 2018) ────────────────────────────────────────────────────
    fii_df  = pd.read_csv(_require("fii_dii_2018_to_2026.csv"), parse_dates=["Date"], index_col="Date").sort_index()
    fii_net = fii_df["FII_Net"].rename("fii_net")
    dii_net = fii_df["DII_Net"].rename("dii_net")

    # ── ICICI quarterly fundamentals (release-lag = period_end + 45d) ────────
    icici_fund_path = DATA_DIR / "ICICI_Bank_Quarterly_Results_FY2010_FY2026 copy.csv"
    icici_nii_qoq   = pd.Series(dtype=float, name="icici_nii_qoq")
    icici_pat_qoq   = pd.Series(dtype=float, name="icici_pat_qoq")

    if icici_fund_path.exists():
        idf = pd.read_csv(icici_fund_path, encoding="utf-8")
        if "Period" in idf.columns:
            release_dates = [
                _parse_period_end(str(p)) + pd.Timedelta(days=45)
                if pd.notna(_parse_period_end(str(p))) else pd.NaT
                for p in idf["Period"]
            ]
            idf["release_date"] = release_dates
            idf = idf.dropna(subset=["release_date"]).set_index("release_date").sort_index()
            nii_col = "NII QoQ Change (₹ Cr)"
            pat_col = "PAT QoQ Change (₹ Cr)"
            if nii_col in idf.columns:
                icici_nii_qoq = _to_daily_ffill(
                    pd.to_numeric(idf[nii_col], errors="coerce").rename("icici_nii_qoq"),
                    daily_index,
                )
            if pat_col in idf.columns:
                icici_pat_qoq = _to_daily_ffill(
                    pd.to_numeric(idf[pat_col], errors="coerce").rename("icici_pat_qoq"),
                    daily_index,
                )

    # ── Assemble factor DataFrame ─────────────────────────────────────────────
    raw_factors = [
        gold_return, usd_inr_ret,
        us_10y_diff, us_5y_diff,
        india_10y_diff, india_5y_diff,
        cpi_diff, wpi_diff, gdp_yoy,
        trade_balance, repo_rate_diff,
        fii_net, dii_net,
        icici_nii_qoq, icici_pat_qoq,
    ]
    factors_df = pd.concat(raw_factors, axis=1).reindex(daily_index)

    # Drop factors that are >80% NaN across the full history
    for col in list(factors_df.columns):
        pct_nan = factors_df[col].isna().mean()
        if pct_nan > 0.80:
            dropped[col] = f"{pct_nan:.1%} NaN across full history"
            factors_df = factors_df.drop(columns=[col])

    # Expose raw close prices for mean-reversion price chart
    meta = {
        "dropped_factors":    dropped,
        "factor_transforms":  FACTOR_TRANSFORMS,
        "data_dir":           str(DATA_DIR),
        "stocks_close":       stocks_close,
        "stocks_ohlcv":       {
            "nifty50":    pd.read_csv(_require("nifty50_daily.csv"),    parse_dates=["Date"], index_col="Date").sort_index().reindex(daily_index),
            "bank_nifty": pd.read_csv(_require("bank_nifty_daily.csv"), parse_dates=["Date"], index_col="Date").sort_index().reindex(daily_index),
            "icici_bank": pd.read_csv(_require("icici_bank_daily.csv"), parse_dates=["Date"], index_col="Date").sort_index().reindex(daily_index),
        },
    }

    return factors_df, stocks_df, meta
