# Futures Pair Trading Backtester

![Status](https://img.shields.io/badge/Status-Active-green)
![Python](https://img.shields.io/badge/Python-3.11-blue)

## Portfolio Positioning

This is not a "high Sharpe demo."  
It is a **research-to-production style stat-arb framework** for futures, built to prioritize:

1. Honest backtests (no hidden look-ahead leakage).
2. Reproducibility (auditable data snapshots and metadata).
3. Operational robustness (degrade gracefully under real-world data failures).

Most pair-trading repos focus on signal formulas.  
This project focuses on the engineering details that usually break in live deployment.

## What Makes This Project Different

1. **Look-ahead prevention is enforced, not just claimed**
- Spread for trading is computed from `beta_lag` (T-1 hedge ratio), not same-bar beta.
- Stability checks also use lagged beta jumps, preventing subtle "future-aware" gating mistakes.
- Model outputs include `valid` flags so warmup and unstable periods are explicitly blocked.

2. **Execution realism is embedded in the signal workflow**
- The logic is designed for **T signal -> T+1 action**, instead of same-bar fill assumptions.
- This intentionally sacrifices inflated paper performance in exchange for realistic behavior.

3. **Non-obvious robustness: Soft Invalid -> Hold**
- When indicators are invalid (`NaN`/`Inf`), the strategy enters a `Hold` state instead of forced flat.
- This avoids unnecessary churn/slippage caused by transient data glitches, a failure mode many demos ignore.

4. **Audit-first data lifecycle**
- Snapshots are persisted with per-file hashes and `manifest.json` metadata.
- Runtime versions and data scope are recorded to support exact reruns and integrity checks.
- Hash mismatch on reload raises a hard failure, guarding against silent data tampering/corruption.

5. **Live-data fragility is handled by design**
- TuShare adapter includes retry, pacing, and local caching.
- If remote requests fail, contract specs can fall back to local config while preserving source/version tags.
- This keeps research and integration tests running when external APIs are unstable.

## Technical Highlights (Implemented)

1. **Model Layer (`src/strategy/models.py`)**
- Rolling OLS and adaptive Kalman implementations.
- Rolling ADF checks for mean-reversion regime validity.
- Defensive numerical handling (`std=0`, non-finite values, warmup validity).

2. **Signal Layer (`src/strategy/signals.py`)**
- Stateful position engine with entry/exit/cooldown/holding timers.
- Risk gates: stop-loss, max-holding-days, volatility filter, beta-jump gate, ADF gate.
- Volatility-target scaling via `vol_target`.

3. **Data Layer (`src/data/`)**
- Standardized adapter interface for decoupling strategy from provider implementation.
- TuShare integration with resilient query behavior.
- Snapshot manager with hash-based reproducibility and audit metadata.

4. **Verification Layer (`scripts/`, `tests/`)**
- `run_smoke.py` for offline integrity checks.
- `run_live.py` for end-to-end online integration checks.
- Unit tests for model outputs, signal transitions, and snapshot lifecycle.

## Repository Layout

```text
main.py
scripts/
  run_smoke.py          # Offline smoke tests (no network)
  run_live.py           # TuShare integration test (network)
src/
  data/
    base.py             # Data adapter interface
    tushare_adapter.py  # TuShare implementation + fallback/cache/retry
    storage.py          # Snapshot save/load + hash audit manifest
  strategy/
    base.py             # Model/signal abstract interfaces
    models.py           # RollingOLS + SimpleKalman
    signals.py          # Signal state machine + risk gates
    rules.py            # StrategyConfig (all thresholds and controls)
tests/
  test_models.py
  test_signals.py
  test_data.py
config/
  instruments.json
  settings.yaml
  universe.json
```

## Quick Start

1. Create environment with Conda:

```bash
conda env create -f environment.yml
conda activate futures-lowfreq-arb
```

2. Set `TUSHARE_TOKEN` in `.env` (required for live integration).

3. Run one of the entry points:

```bash
python main.py
# or
python scripts/run_smoke.py
python scripts/run_live.py
pytest -v
```
