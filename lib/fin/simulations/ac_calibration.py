"""
Ticker-driven calibration helpers for Almgren-Chriss environments.

This module calibrates the unaffected price dynamics (mu, sigma) used by
single-agent and multi-agent AC environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..calibration.data.yfinance_fetcher import YFinanceFetcher
from ..calibration.physical import GBMCalibrator


SUPPORTED_AC_CALIBRATION_MODELS = ("gbm", "sabr", "heston")


@dataclass
class ACTickerCalibrationResult:
    """Container for ticker calibration outputs mapped to AC env units."""

    ticker: str
    model: str
    mu_annual: float
    sigma_annual: float
    mu_env: float
    sigma_env: float
    spot_price: float
    history_start: date
    history_end: date
    n_observations: int
    periods_per_year: int
    used_options: bool
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a compact, serializable summary."""
        return {
            "ticker": self.ticker,
            "model": self.model,
            "mu_annual": self.mu_annual,
            "sigma_annual": self.sigma_annual,
            "mu_env": self.mu_env,
            "sigma_env": self.sigma_env,
            "spot_price": self.spot_price,
            "history_start": self.history_start.isoformat(),
            "history_end": self.history_end.isoformat(),
            "n_observations": self.n_observations,
            "periods_per_year": self.periods_per_year,
            "used_options": self.used_options,
        }


def _annual_to_env_params(
    mu_annual: float,
    sigma_annual: float,
    periods_per_year: int,
    env_time_scale: str,
) -> tuple[float, float]:
    """Convert annualized calibration outputs to env time units."""
    if env_time_scale == "annual":
        return float(mu_annual), float(sigma_annual)
    if env_time_scale == "daily":
        return (
            float(mu_annual) / periods_per_year,
            float(sigma_annual) / np.sqrt(periods_per_year),
        )
    raise ValueError("env_time_scale must be either 'daily' or 'annual'")


def _extract_history_dates(df: pd.DataFrame, default_start: date, default_end: date) -> tuple[date, date]:
    if "Date" not in df.columns or len(df) == 0:
        return default_start, default_end
    start = pd.to_datetime(df["Date"].iloc[0]).date()
    end = pd.to_datetime(df["Date"].iloc[-1]).date()
    return start, end


def calibrate_ac_dynamics_from_ticker(
    ticker: str,
    model: str = "gbm",
    history_days: int = 252,
    periods_per_year: int = 252,
    env_time_scale: str = "daily",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
    history_interval: str = "1d",
    sabr_beta: Optional[float] = 0.5,
    heston_method: str = "L-BFGS-B",
    heston_maxiter: int = 250,
    heston_tol: float = 1e-6,
    fetcher: Optional[YFinanceFetcher] = None,
) -> ACTickerCalibrationResult:
    """
    Calibrate AC price dynamics from ticker data.

    Models:
    - `gbm`: historical close-to-close calibration only.
    - `sabr`: historical drift from GBM + option-implied sigma from SABR nu.
    - `heston`: historical drift from GBM + option-implied sigma from sqrt(theta).
    """
    model_name = model.lower().strip()
    if model_name not in SUPPORTED_AC_CALIBRATION_MODELS:
        supported = ", ".join(SUPPORTED_AC_CALIBRATION_MODELS)
        raise ValueError(f"Unsupported model '{model}'. Supported: {supported}")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    if end_date is None:
        end_date = date.today()
    if start_date is None:
        # Add a small buffer because non-trading days reduce realized sample size.
        start_date = end_date - timedelta(days=history_days + 30)
    if start_date >= end_date:
        raise ValueError("start_date must be before end_date")

    data_fetcher = fetcher or YFinanceFetcher(
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    history_df = data_fetcher.get_history(
        ticker=ticker,
        start=start_date,
        end=end_date,
        interval=history_interval,
    )
    if "Close" not in history_df.columns:
        raise ValueError("Fetched history is missing 'Close' column")

    close_prices = history_df["Close"].to_numpy(dtype=float)
    close_prices = close_prices[np.isfinite(close_prices)]
    if close_prices.size < 30:
        raise ValueError("Need at least 30 close observations for stable calibration")

    # Baseline physical-measure calibration from history.
    gbm_result = GBMCalibrator().fit(
        close_prices,
        dt=1.0 / periods_per_year,
        periods_per_year=periods_per_year,
    )
    mu_annual = float(gbm_result.mu)
    sigma_annual = float(max(gbm_result.sigma, 1e-12))

    diagnostics: Dict[str, Any] = {"gbm": gbm_result}
    used_options = False

    # Optional risk-neutral volatility calibration from options.
    if model_name == "sabr":
        from ..calibration.risk_neutral import SABRCalibrator

        chain = data_fetcher.get_option_chain(ticker)
        sabr_result = SABRCalibrator(beta=sabr_beta).calibrate(chain)
        sigma_annual = float(max(sabr_result.nu, 1e-12))
        diagnostics["sabr"] = sabr_result
        used_options = True
    elif model_name == "heston":
        from ..calibration.risk_neutral import HestonCalibrator

        chain = data_fetcher.get_option_chain(ticker)
        heston_result = HestonCalibrator().calibrate(
            chain,
            method=heston_method,
            maxiter=heston_maxiter,
            tol=heston_tol,
        )
        sigma_annual = float(np.sqrt(max(heston_result.theta, 1e-12)))
        diagnostics["heston"] = heston_result
        used_options = True

    mu_env, sigma_env = _annual_to_env_params(
        mu_annual=mu_annual,
        sigma_annual=sigma_annual,
        periods_per_year=periods_per_year,
        env_time_scale=env_time_scale,
    )
    history_start, history_end = _extract_history_dates(history_df, start_date, end_date)

    return ACTickerCalibrationResult(
        ticker=ticker,
        model=model_name,
        mu_annual=mu_annual,
        sigma_annual=sigma_annual,
        mu_env=mu_env,
        sigma_env=sigma_env,
        spot_price=float(close_prices[-1]),
        history_start=history_start,
        history_end=history_end,
        n_observations=int(close_prices.size),
        periods_per_year=periods_per_year,
        used_options=used_options,
        diagnostics=diagnostics,
    )
