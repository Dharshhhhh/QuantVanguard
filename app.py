"""
QuantVanguard — Portfolio Optimization MVP

Features:
- Downloads adjusted market prices from Yahoo Finance when available.
- Uses a transparent momentum-based expected-return estimate (NOT a trained LSTM).
- Enumerates all K-asset subsets for a small universe, then optimizes weights
  with SciPy under long-only, fully-invested, per-holding-cap constraints.
- Reports exact feasibility checks and clearly labels the classical solver.
- Uses synthetic prices only as an explicitly disclosed fallback.
"""

from __future__ import annotations

import itertools
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.optimize import minimize

st.set_page_config(
    page_title="QuantVanguard | Portfolio Optimization",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #090614 0%, #120b24 50%, #040208 100%);
        color: #F8FAFC;
        font-family: Inter, sans-serif;
    }
    h1, h2, h3 { letter-spacing: -0.025em; }
    div[data-testid="stMetric"], div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 12px;
        padding: 16px !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
        color: #C084FC !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED 0%, #2563EB 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.7rem 1.2rem;
    }
    section[data-testid="stSidebar"] {
        background-color: rgba(10,6,20,0.96);
        border-right: 1px solid rgba(255,255,255,0.07);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

KNOWN_SECTORS = {
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "GOOGL": "Communication Services",
    "GOOG": "Communication Services",
    "NVDA": "Information Technology",
    "AMZN": "Consumer Discretionary",
    "META": "Communication Services",
    "TSLA": "Consumer Discretionary",
    "JPM": "Financials",
    "V": "Financials",
    "UNH": "Health Care",
    "XOM": "Energy",
    "AVGO": "Information Technology",
}


def parse_tickers(raw: str) -> list[str]:
    tickers: list[str] = []
    for part in raw.replace(";", ",").split(","):
        ticker = part.strip().upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def synthetic_prices(tickers: list[str], days: int = 504) -> pd.DataFrame:
    """Deterministic demo data. Never represented as live market data."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    values: dict[str, np.ndarray] = {}
    for i, ticker in enumerate(tickers):
        daily_mu = (0.08 + 0.02 * np.sin(i)) / 252
        daily_sigma = (0.20 + 0.025 * (i % 3)) / np.sqrt(252)
        shocks = rng.normal(daily_mu, daily_sigma, days)
        values[ticker] = 100 * np.exp(np.cumsum(shocks))
    return pd.DataFrame(values, index=dates)


def download_prices(tickers: list[str]) -> tuple[pd.DataFrame, str]:
    try:
        import yfinance as yf

        raw = yf.download(
            tickers=tickers,
            period="2y",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw is None or raw.empty:
            raise RuntimeError("Yahoo Finance returned no data.")

        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" not in raw.columns.get_level_values(0):
                raise RuntimeError("Adjusted close prices were not found.")
            closes = raw["Close"].copy()
        else:
            if "Close" not in raw.columns:
                raise RuntimeError("Close prices were not found.")
            closes = raw[["Close"]].copy()
            if len(tickers) == 1:
                closes.columns = tickers

        closes = closes.dropna(how="all").ffill()
        available = [ticker for ticker in tickers if ticker in closes.columns]
        if not available:
            raise RuntimeError("None of the requested tickers returned data.")

        closes = closes[available].dropna()
        if len(closes) < 60:
            raise RuntimeError("Fewer than 60 overlapping daily observations.")
        return closes, "Yahoo Finance adjusted prices (2 years)"
    except Exception as exc:
        return synthetic_prices(tickers), (
            f"SYNTHETIC DEMO DATA — live download failed: "
            f"{type(exc).__name__}: {exc}"
        )


def estimate_expected_returns(prices: pd.DataFrame) -> pd.Series:
    """
    Transparent heuristic, not a trained ML model.
    Combines short-, medium-, and longer-term historical momentum.
    """
    returns = prices.pct_change().dropna()
    if returns.empty:
        return pd.Series(0.08, index=prices.columns)

    short = returns.tail(5).mean() * 252
    medium = returns.tail(21).mean() * 252
    long = returns.tail(63).mean() * 252 if len(returns) >= 63 else medium

    # Shrink volatile, short-term estimates toward the medium/long-term estimate.
    vol = returns.tail(63).std() * np.sqrt(252)
    vol = vol.replace(0, 0.20).fillna(0.20)
    momentum = 0.25 * short + 0.35 * medium + 0.40 * long
    shrink = (1.0 / (1.0 + vol)).clip(0.25, 0.80)
    estimate = momentum * shrink
    return estimate.clip(-0.50, 0.50).fillna(0.0)


def resolve_sectors(tickers: list[str]) -> dict[str, str]:
    sectors = {ticker: KNOWN_SECTORS[ticker] for ticker in tickers if ticker in KNOWN_SECTORS}
    missing = [ticker for ticker in tickers if ticker not in sectors]
    for ticker in missing:
        sectors[ticker] = "Unclassified"
    return sectors


def portfolio_metrics(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    expected_return = float(weights @ mu)
    variance = max(float(weights @ cov @ weights), 0.0)
    return expected_return, float(np.sqrt(variance))


def optimize_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float,
    max_weight: float,
) -> tuple[np.ndarray, bool, str]:
    """Maximize expected return minus risk penalty with long-only weight constraints."""
    n = len(mu)
    if n * max_weight < 1.0 - 1e-9:
        return np.full(n, np.nan), False, (
            f"Infeasible: {n} selected assets × {max_weight:.0%} maximum "
            "weight cannot sum to 100%."
        )

    initial = np.full(n, 1.0 / n)
    initial = np.minimum(initial, max_weight)
    initial = initial / initial.sum()

    def objective(weights: np.ndarray) -> float:
        return float(-(mu @ weights - risk_aversion * (weights @ cov @ weights)))

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[
            {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        ],
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    if not result.success:
        return initial, False, f"SciPy optimizer did not converge: {result.message}"

    weights = np.clip(result.x, 0.0, max_weight)
    if weights.sum() <= 0:
        return initial, False, "Optimizer returned an invalid zero-weight portfolio."
    weights /= weights.sum()

    # Normalizing can marginally violate the cap; check explicitly.
    if weights.max() > max_weight + 1e-6:
        return weights, False, "Returned weights violate the maximum holding cap."
    return weights, True, "Optimization converged."


def select_and_allocate(
    tickers: list[str],
    mu: np.ndarray,
    cov: np.ndarray,
    k: int,
    risk_aversion: float,
    max_weight: float,
    sectors: dict[str, str],
    sector_cap: float | None,
) -> dict[str, Any]:
    """
    For a small universe, examine every K-asset subset and solve its continuous
    allocation. This is an exact subset search with a numerical weight optimizer,
    not QAOA and not a quantum-computing result.
    """
    started = time.perf_counter()
    best: dict[str, Any] | None = None
    n = len(tickers)

    for subset in itertools.combinations(range(n), k):
        subset_mu = mu[list(subset)]
        subset_cov = cov[np.ix_(subset, subset)]
        subset_tickers = [tickers[i] for i in subset]

        weights, success, note = optimize_weights(
            subset_mu, subset_cov, risk_aversion, max_weight
        )
        if not success:
            continue

        sector_weights: dict[str, float] = {}
        for ticker, weight in zip(subset_tickers, weights):
            sector = sectors.get(ticker, "Unclassified")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + float(weight)

        if sector_cap is not None and any(
            value > sector_cap + 1e-6 for value in sector_weights.values()
        ):
            continue

        ret, volatility = portfolio_metrics(weights, subset_mu, subset_cov)
        objective = float(ret - risk_aversion * volatility**2)
        if best is None or objective > best["objective"]:
            best = {
                "indices": list(subset),
                "tickers": subset_tickers,
                "weights": weights,
                "return": ret,
                "volatility": volatility,
                "objective": objective,
                "sector_weights": sector_weights,
                "optimizer_note": note,
            }

    elapsed_ms = (time.perf_counter() - started) * 1000
    if best is None:
        raise ValueError(
            "No feasible portfolio found. Try increasing K, raising the holding "
            "cap, or disabling the sector cap."
        )
    best["elapsed_ms"] = elapsed_ms
    return best


def run_pipeline(
    requested_tickers: list[str],
    k: int,
    risk_aversion: float,
    max_weight: float,
    enforce_sector_cap: bool,
    sector_cap: float,
) -> dict[str, Any]:
    prices, source = download_prices(requested_tickers)
    tickers = list(prices.columns)
    if len(tickers) < 1:
        raise ValueError("No usable ticker data was found.")
    k = min(max(int(k), 1), len(tickers))

    daily_returns = prices.pct_change().dropna()
    cov = daily_returns.cov().to_numpy(dtype=float) * 252.0
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = 0.5 * (cov + cov.T) + 1e-8 * np.eye(len(tickers))

    mu_series = estimate_expected_returns(prices).reindex(tickers).fillna(0.0)
    mu = mu_series.to_numpy(dtype=float)
    sectors = resolve_sectors(tickers)

    chosen_sector_cap = sector_cap if enforce_sector_cap else None
    solution = select_and_allocate(
        tickers=tickers,
        mu=mu,
        cov=cov,
        k=k,
        risk_aversion=risk_aversion,
        max_weight=max_weight,
        sectors=sectors,
        sector_cap=chosen_sector_cap,
    )

    selected_indices = solution["indices"]
    selected_mu = mu[selected_indices]
    selected_cov = cov[np.ix_(selected_indices, selected_indices)]
    weights = solution["weights"]

    weights_table = pd.DataFrame(
        {
            "Ticker": solution["tickers"],
            "Sector": [sectors.get(t, "Unclassified") for t in solution["tickers"]],
            "Weight (%)": weights * 100,
            "Expected Return Estimate (%)": selected_mu * 100,
        }
    )

    sector_table = (
        weights_table.groupby("Sector", as_index=False)["Weight (%)"]
        .sum()
        .sort_values("Weight (%)", ascending=False)
    )

    return {
        "prices": prices,
        "source": source,
        "tickers": tickers,
        "mu": mu,
        "cov": cov,
        "sectors": sectors,
        "k": k,
        "risk_aversion": risk_aversion,
        "max_weight": max_weight,
        "sector_cap": chosen_sector_cap,
        "solution": solution,
        "weights_table": weights_table,
        "sector_table": sector_table,
        "weights": weights,
        "selected_tickers": solution["tickers"],
        "expected_return": solution["return"],
        "volatility": solution["volatility"],
        "objective": solution["objective"],
        "elapsed_ms": solution["elapsed_ms"],
        "all_weights_valid": bool(
            np.isclose(weights.sum(), 1.0, atol=1e-5)
            and np.min(weights) >= -1e-6
            and np.max(weights) <= max_weight + 1e-5
        ),
    }


# ----------------------------- UI ---------------------------------

st.markdown("### ⚛️ QUANTVANGUARD | CONTROL ROOM")
st.title("Hybrid Quantum-AI Portfolio Optimization")
st.markdown(
    "**Portfolio allocation MVP:** historical market prices → transparent "
    "momentum-based return estimates → covariance-based risk → constrained "
    "portfolio search. This version uses classical exact subset enumeration "
    "and SciPy weight optimization; it does not claim to execute QAOA or a trained LSTM."
)
st.markdown("---")

with st.sidebar:
    st.header("Pipeline Config")
    ticker_input = st.text_input(
        "Asset Universe Tickers",
        "AAPL, MSFT, GOOGL, NVDA, AMZN",
    )
    requested_tickers = parse_tickers(ticker_input)
    if not requested_tickers:
        requested_tickers = ["AAPL"]

    max_universe = min(10, len(requested_tickers))
    target_k = st.slider(
        "Target Asset Count (K)",
        min_value=1,
        max_value=max_universe,
        value=min(4, max_universe),
    )
    risk_profile = st.selectbox(
        "Risk Aversion Profile",
        ["Conservative (q=2.0)", "Balanced (q=1.25)", "Aggressive (q=0.5)"],
        index=1,
    )
    risk_aversion = (
        2.0 if risk_profile.startswith("Conservative")
        else 1.25 if risk_profile.startswith("Balanced")
        else 0.5
    )
    max_weight_pct = st.slider(
        "Maximum individual asset weight (%)",
        min_value=20,
        max_value=100,
        value=40,
        step=5,
    )
    max_weight = max_weight_pct / 100.0
    enforce_sector_cap = st.toggle("Enforce sector exposure cap", value=False)
    sector_cap_pct = st.slider(
        "Maximum sector exposure (%)",
        min_value=25,
        max_value=100,
        value=60,
        step=5,
        disabled=not enforce_sector_cap,
    )
    sector_cap = sector_cap_pct / 100.0
    st.caption(
        "The universe is limited to 10 tickers because this version enumerates "
        "asset subsets. A hard 25% individual holding cap requires at least 4 assets."
    )
    run_btn = st.button("🚀 Run Portfolio Optimizer", use_container_width=True)

if run_btn or "results" not in st.session_state:
    if len(requested_tickers) > 10:
        st.error("Please enter 10 tickers or fewer for this exact subset-search MVP.")
        st.stop()
    if target_k * max_weight < 1.0 - 1e-9:
        st.error(
            f"Infeasible constraints: {target_k} selected assets at a maximum "
            f"weight of {max_weight:.0%} can allocate only {target_k * max_weight:.0%}. "
            "Increase K or increase the maximum individual weight."
        )
        st.stop()
    with st.spinner("Downloading prices and optimizing the constrained portfolio..."):
        try:
            st.session_state["results"] = run_pipeline(
                requested_tickers,
                target_k,
                risk_aversion,
                max_weight,
                enforce_sector_cap,
                sector_cap,
            )
            st.session_state["run_error"] = None
        except Exception as exc:
            st.session_state["results"] = None
            st.session_state["run_error"] = f"{type(exc).__name__}: {exc}"

if st.session_state.get("run_error"):
    st.error(f"Pipeline failed: {st.session_state['run_error']}")
    st.stop()

res = st.session_state.get("results")
if not res:
    st.info("Set your configuration in the sidebar and run the optimizer.")
    st.stop()

if not res["source"].startswith("Yahoo Finance"):
    st.warning(res["source"])

if res["k"] != target_k or res["tickers"] != requested_tickers:
    st.info("The displayed results are from the last successful run. Click Run Portfolio Optimizer to apply current sidebar settings.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Selected Assets", f"{len(res['selected_tickers'])} / {res['k']}")
col2.metric("Estimated Annual Return", f"{res['expected_return'] * 100:.2f}%")
col3.metric("Estimated Annual Volatility", f"{res['volatility'] * 100:.2f}%")
col4.metric("Optimization Runtime", f"{res['elapsed_ms']:.1f} ms")
st.caption(
    f"Solver: classical exact subset enumeration + SciPy SLSQP · "
    f"Data: {res['source']} · risk-aversion q = {res['risk_aversion']}"
)
st.caption(
    "Return estimates are heuristic and not investment advice. Past market data does not guarantee future performance."
)

st.markdown("---")
left, right = st.columns(2)

with left:
    st.subheader("📊 Optimized Portfolio Allocation")
    fig = px.pie(
        res["weights_table"],
        names="Ticker",
        values="Weight (%)",
        hole=0.55,
        color_discrete_sequence=px.colors.sequential.Purp_r,
    )
    fig.update_traces(textinfo="label+percent")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#F8FAFC",
        margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("📉 Historical Adjusted Prices")
    st.line_chart(res["prices"], use_container_width=True)

st.subheader("⚖️ Asset Weights and Return Estimates")
st.dataframe(
    res["weights_table"].style.format(
        {"Weight (%)": "{:.2f}", "Expected Return Estimate (%)": "{:.2f}"}
    ),
    use_container_width=True,
    hide_index=True,
)

st.subheader("🏭 Sector Exposure")
st.dataframe(
    res["sector_table"].style.format({"Weight (%)": "{:.2f}"}),
    use_container_width=True,
    hide_index=True,
)

st.subheader("🛡️ Constraint Audit")
weights = res["weights"]
cardinality_ok = len(res["selected_tickers"]) == res["k"]
fully_invested_ok = bool(np.isclose(weights.sum(), 1.0, atol=1e-5))
holding_cap_ok = bool(np.max(weights) <= res["max_weight"] + 1e-5)
if res["sector_cap"] is None:
    sector_cap_status = "Not enforced"
    sector_cap_ok: bool | None = None
else:
    sector_cap_ok = bool(
        res["sector_table"]["Weight (%)"].max() / 100
        <= res["sector_cap"] + 1e-5
    )
    sector_cap_status = "Passed" if sector_cap_ok else "Failed"

audit = pd.DataFrame(
    [
        {
            "Check": "Exact asset count",
            "Rule": f"Select exactly {res['k']} assets",
            "Measured": f"{len(res['selected_tickers'])} assets",
            "Status": "PASS" if cardinality_ok else "FAIL",
        },
        {
            "Check": "Fully invested",
            "Rule": "Weights sum to 100%",
            "Measured": f"{weights.sum() * 100:.2f}%",
            "Status": "PASS" if fully_invested_ok else "FAIL",
        },
        {
            "Check": "Maximum individual holding",
            "Rule": f"Each weight ≤ {res['max_weight'] * 100:.1f}%",
            "Measured": f"{weights.max() * 100:.2f}% maximum",
            "Status": "PASS" if holding_cap_ok else "FAIL",
        },
        {
            "Check": "Maximum sector exposure",
            "Rule": (
                f"Each sector ≤ {res['sector_cap'] * 100:.1f}%"
                if res["sector_cap"] is not None
                else "Not enforced"
            ),
            "Measured": sector_cap_status,
            "Status": (
                sector_cap_status
                if sector_cap_status == "Not enforced"
                else ("PASS" if sector_cap_ok else "FAIL")
            ),
        },
    ]
)
st.dataframe(audit, use_container_width=True, hide_index=True)

with st.expander("🔍 Inspect model assumptions and optimization details"):
    st.markdown("#### Expected return estimates")
    all_forecasts = pd.DataFrame(
        {
            "Ticker": res["tickers"],
            "Sector": [res["sectors"].get(t, "Unclassified") for t in res["tickers"]],
            "Estimated Annual Return (%)": res["mu"] * 100,
        }
    )
    st.dataframe(
        all_forecasts.style.format({"Estimated Annual Return (%)": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("#### Optimization details")
    st.write(f"Selected assets: {', '.join(res['selected_tickers'])}")
    st.write(f"Objective value (return − q × variance): {res['objective']:.6f}")
    st.write(f"Weight optimizer: {res['solution']['optimizer_note']}")
    st.write(f"Maximum holding limit: {res['max_weight']:.0%}")
    st.write(
        "Return estimator: short/medium/long historical momentum with volatility shrinkage. "
        "It is not a trained LSTM."
    )
    st.write(
        "Selection method: all K-asset subsets are examined for this small universe. "
        "Each subset's weights are optimized with SciPy SLSQP. No QAOA circuit is executed."
    )
    st.markdown("#### Annualized covariance matrix")
    st.dataframe(
        pd.DataFrame(res["cov"], index=res["tickers"], columns=res["tickers"]).style.format("{:.4f}"),
        use_container_width=True,
    )
