# QuantVanguard

Hybrid Quantum-AI portfolio optimizer. Streamlit dashboard that forecasts expected returns from recent momentum (LSTM-style gates), formulates a cardinality-constrained QUBO, and solves it with QAOA (NumPy eigensolver fallback) against a SciPy classical baseline.

## Prerequisites

- Python 3.10 or newer
- Internet access (Yahoo Finance via `yfinance`; the app falls back to synthetic prices if the download fails)

## Setup and run

### Windows (PowerShell)

```powershell
cd QuantVanguard
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If script execution is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### macOS / Linux

```bash
cd QuantVanguard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The browser should open at [http://localhost:8501](http://localhost:8501). Use the sidebar to set tickers, cardinality \(K\), risk profile, and the 25% sector cap, then click **Run Hybrid Optimizer (LSTM + QAOA)**.

## Notes

- QAOA is attempted first. If the live Sampler/QAOA stack is unavailable, the app uses Qiskit’s `NumPyMinimumEigenOptimizer` (exact for small universes) so demos stay fast and stable.
- A SciPy SLSQP continuous quadratic program is solved in parallel and discretized to the top-\(K\) names for benchmarking.
