"""
Tests for ticker-driven calibration hooks in AC environments.
"""

import os
import sys
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.fin.calibration.data.data_provider import OptionChain
from lib.fin.calibration import risk_neutral
from lib.fin.simulations import ac_calibration
from lib.fin.simulations.almgren_chriss_env import make_almgren_chriss_env_with_ticker
from lib.fin.simulations.multi_agent_ac_env import make_multi_agent_ac_env_with_ticker


class FakeFetcher:
    """Deterministic market data provider for offline tests."""

    def __init__(self, risk_free_rate: float = 0.05, dividend_yield: float = 0.0):
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield

    def get_history(self, ticker, start, end, interval="1d", **kwargs):
        n = 220
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        log_trend = np.linspace(0.0, 0.08, n)
        close = 100.0 * np.exp(log_trend)

        return pd.DataFrame(
            {
                "Date": dates,
                "Open": close,
                "High": close * 1.001,
                "Low": close * 0.999,
                "Close": close,
                "Volume": np.full(n, 1_000_000, dtype=int),
            }
        )

    def get_option_chain(self, ticker, reference_date=None):
        return OptionChain(
            underlying=ticker,
            spot_price=108.0,
            reference_date=reference_date or date(2025, 1, 2),
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            options=[],
        )


def test_make_single_agent_env_with_ticker_gbm():
    env, calibration = make_almgren_chriss_env_with_ticker(
        ticker="SPY",
        model="gbm",
        n_steps=5,
        fetcher=FakeFetcher(),
        return_calibration=True,
    )

    assert calibration.model == "gbm"
    assert np.isclose(env.params.mu, calibration.mu_env)
    assert np.isclose(env.params.sigma, calibration.sigma_env)
    assert np.isclose(env.params.S_0, calibration.spot_price)

    obs, _ = env.reset(seed=7)
    assert obs.shape == (3,)
    env.close()


def test_make_multi_agent_env_with_ticker_gbm():
    env, calibration = make_multi_agent_ac_env_with_ticker(
        ticker="SPY",
        model="gbm",
        n_agents=2,
        n_steps=5,
        fetcher=FakeFetcher(),
        return_calibration=True,
    )

    assert calibration.model == "gbm"
    assert np.isclose(env.params.mu, calibration.mu_env)
    assert np.isclose(env.params.sigma, calibration.sigma_env)
    assert np.isclose(env.params.S_0, calibration.spot_price)

    observations, _ = env.reset(seed=7)
    assert set(observations.keys()) == {"trader_0", "trader_1"}
    env.close()


def test_sabr_sigma_mapping(monkeypatch):
    class FakeSABRCalibrator:
        def __init__(self, beta=None, weighting="vega"):
            self.beta = beta

        def calibrate(self, chain):
            return SimpleNamespace(nu=0.36)

    monkeypatch.setattr(risk_neutral, "SABRCalibrator", FakeSABRCalibrator)

    result = ac_calibration.calibrate_ac_dynamics_from_ticker(
        ticker="SPY",
        model="sabr",
        fetcher=FakeFetcher(),
    )

    assert result.used_options
    assert np.isclose(result.sigma_annual, 0.36)


def test_heston_sigma_mapping(monkeypatch):
    class FakeHestonCalibrator:
        def __init__(self, *args, **kwargs):
            pass

        def calibrate(self, chain, **kwargs):
            return SimpleNamespace(theta=0.04)

    monkeypatch.setattr(risk_neutral, "HestonCalibrator", FakeHestonCalibrator)

    result = ac_calibration.calibrate_ac_dynamics_from_ticker(
        ticker="SPY",
        model="heston",
        fetcher=FakeFetcher(),
    )

    assert result.used_options
    assert np.isclose(result.sigma_annual, 0.2)


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unsupported model"):
        ac_calibration.calibrate_ac_dynamics_from_ticker(
            ticker="SPY",
            model="not_a_model",
            fetcher=FakeFetcher(),
        )
