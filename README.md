# FA-RLWA

**Forecast-Aware Deep Reinforcement Learning for Multi-Indicator User Association in SDN-Based Wi-Fi Networks**

FA-RLWA is a research prototype built from the repository's SRS and Manzoor et al.'s 2023 TiWA paper. It simulates a centrally controlled Wi-Fi network, measures the TiWA indicators, provides baseline association policies and forecast models, and exposes an association-control task as a Gymnasium environment.

> This repository contains an implementation scaffold and a runnable abstract simulator. It does **not** claim that the SRS acceptance criteria have already been met: no forecaster or RL checkpoint is included, paper calibration is not yet complete, and Mininet-WiFi/ns-3 validation requires a separate Linux setup. See [`docs/limitations.md`](docs/limitations.md) and [`docs/paper-notes.md`](docs/paper-notes.md).

## What is implemented

- Seedable one-second Python simulator with log-distance path loss, correlated shadowing, RSSI-based PHY-rate lookup, collisions, airtime sharing, traffic profiles, static/random-waypoint/scripted mobility, flash-crowd arrivals, channel changes, and handover costs.
- Per-AP RSSI, failure/loss, throughput, channel occupancy, client count, and own-airtime reports, with moving-average support.
- Policies: client-driven RSSI, original TiWA (with the SRS's logged interpretations), corrected normalized TiWA, random, and TiWA + an EWMA/other saved forecaster.
- Forecasters: EWMA, 10-report moving average, ARIMA, random forest, XGBoost, and two-layer LSTM. ARIMA/ML models are optional dependencies.
- Gymnasium `WifiAssocEnv` with the SRS observation lengths (64 for five APs with forecasts; 44 without), discrete stay/move actions, action masks, dwell-time enforcement, and a configurable fairness/throughput/handover/retransmission reward.
- Experiment CSV logs, plots, paired Wilcoxon tests, Holm correction, and bootstrap 95% confidence intervals.
- DQN and PPO training entry points; training is intentionally optional and can be computationally expensive.
- Optional Mininet-WiFi online policy loop for static AP/STA topologies, with guarded association actions and compatible trace output; it requires a separate Linux/root setup and has not been exercised in this sandbox.

The default output is CSV rather than Parquet. CSV is one of the formats allowed by MON-03 and avoids requiring PyArrow for the baseline simulator. All generated data, model checkpoints, and results are git-ignored.

## Quick start

Python 3.10 or 3.11 is recommended. The abstract simulator, experiment CLI, tests, and Streamlit UI are intended to work on Windows as well as Linux/macOS. The real Wi-Fi scan preflight and Mininet-WiFi backend are Linux-only.

### Windows 10/11 (PowerShell)

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

If PowerShell blocks environment activation, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for that terminal, or call `.venv\Scripts\python.exe` directly.

Run a simulation from PowerShell with a single command line:

```powershell
python -m fa_rlwa.experiments.runner --scenario configs/scenarios/sc1_steady.yaml --policy client_driven --seed 3 --duration 120 --out results/demo
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
pytest -q
```

### Optional Streamlit UI

Install the UI extra and launch the local dashboard:

```bash
python -m pip install -e ".[ui]"
python -m streamlit run src/fa_rlwa/ui/streamlit_app.py
```

The page lets you choose a scenario and policy, optionally check a Linux Wi-Fi adapter, launch a simulation, and view/download its metrics and plots. It runs the abstract simulator only. A separate Mininet-WiFi CLI backend is available for emulated AP/client control; neither the dashboard nor the preflight directly controls physical APs or client devices.

Run a short, reproducible baseline experiment from the CLI:

```bash
python -m fa_rlwa.experiments.runner \
  --scenario configs/scenarios/sc1_steady.yaml \
  --policy client_driven --seed 3 --duration 120 --out results/demo
```

Run another baseline or forecast-aware heuristic:

```bash
python -m fa_rlwa.experiments.runner \
  --scenario configs/scenarios/sc4_flash_crowd.yaml \
  --policy tiwa_original --seed 3 --duration 600 --out results/tiwa-demo

python -m fa_rlwa.experiments.runner \
  --scenario configs/scenarios/sc4_flash_crowd.yaml \
  --policy tiwa_forecast --forecaster ewma --seed 3 --duration 600 --out results/forecast-demo
```

A run directory contains `ap_indicators.csv`, `sta_log.csv`, `handover_log.csv`, `invalid_actions.csv`, `metrics_summary.csv`, `config_resolved.yaml`, and `figures/` with throughput, AP occupancy, and fairness plots. For the SRS-compatible default duration, omit `--duration` (1,200 seconds; first 60 seconds are excluded from summary metrics).

## Project structure

```text
configs/                       Scenario, policy, forecast, RL, and matrix YAML
src/fa_rlwa/
  sim/                          Propagation, mobility, traffic, MAC, network
  monitor/                      Indicators, normalization, history store
  policies/                     Client, TiWA, corrected TiWA, random, RL adapter
  forecast/                     EWMA/MA, ARIMA, RF/XGBoost, LSTM, dataset, train/eval
  env/                          Gymnasium environment, observations, reward
  controller/                   SDN dispatch and handover analysis
  experiments/                  Runner, metrics, statistics, plots, reports
  rl/                           DQN/PPO training
  ui/                           Optional Streamlit dashboard
 tests/                         Unit and integration tests
docs/                           Paper notes, design decisions, limitations
```

## Forecast workflow

Install the heavier training dependencies only when needed:

```bash
python -m pip install -r requirements-ml.txt
```

The reference configuration separates data by scenario and seed. The full SRS dataset-generation workflow is intentionally large; use a smaller YAML split or shorter scenario duration while developing.

```bash
python -m fa_rlwa.forecast.dataset \
  --config configs/forecast/dataset.yaml --out data/forecast/v1
python -m fa_rlwa.forecast.train \
  --model ewma --config configs/forecast/ewma.yaml \
  --train-data data/forecast/v1/train.npz \
  --val-data data/forecast/v1/val.npz --out models/forecast/ewma_v1
python -m fa_rlwa.forecast.evaluate \
  --model models/forecast/ewma_v1 \
  --test-data data/forecast/v1/test_id.npz \
  --out results/forecast/ewma_test.csv
```

Swap `ewma` for `arima`, `random_forest`, `xgboost`/`xgb`, or `lstm` to train a different forecaster. The 10-second forecast used by association policies comes from the saved model's `(5, 10, 30)` horizon layout.

## RL workflow

Train a DQN with and without forecasts (one seed shown; the SRS requests five):

```bash
python -m fa_rlwa.rl.train_dqn --config configs/rl/dqn_forecast.yaml --seed 0
python -m fa_rlwa.rl.train_dqn --config configs/rl/dqn_noforecast.yaml --seed 0
```

Use `--timesteps 20000` for a smoke run. Checkpoints and TensorBoard logs are saved under ignored `models/` and `results/` directories. PPO is available as an alternative:

```bash
python -m fa_rlwa.rl.train_ppo --config configs/rl/ppo.yaml --seed 0
```

Evaluate a saved model through the standard runner:

```bash
python -m fa_rlwa.experiments.runner \
  --scenario configs/scenarios/sc1_steady.yaml \
  --policy rl_forecast --model models/rl/dqn_forecast_s0 \
  --seed 200 --out results/rl-demo
```

An independent K_max=10 config is provided for the scaled scenario because its observation/action spaces differ from K_max=5.

## Experiments and analysis

Run one scenario/policy/seed combination with a YAML policy configuration:

```bash
python -m fa_rlwa.experiments.runner \
  --scenario configs/scenarios/sc1_steady.yaml \
  --policy tiwa_original \
  --policy-config configs/policies/tiwa_original.yaml \
  --seed 0 --out results/tiwa_s0
```

Run a matrix (20 seeds: 200–219) and aggregate its outputs:

```bash
python -m fa_rlwa.experiments.runner \
  --matrix configs/experiments/main.yaml --jobs 4
python -m fa_rlwa.experiments.report \
  --results results/main --out results/report
```

`configs/experiments/rl_main.yaml` is the separate RL comparison matrix. Matrix runs can be parallelized with `--jobs N`; model training is kept separate from evaluation. Reports include mean, standard deviation, percentile-bootstrap confidence intervals, paired Wilcoxon p-values, and Holm-adjusted p-values when at least two matched seeds are available.

## Optional Wi-Fi adapter preflight

For a Linux machine connected to a real Wi-Fi adapter, check that the selected interface accepts a scan before starting an experiment:

```bash
fa-rlwa wifi-check --interface wlan0
fa-rlwa run --scenario configs/scenarios/sc1_steady.yaml \
  --policy client_driven --wifi-adapter wlan0 --seed 3 --out results/demo
```

`iw` must be installed and the interface must already be enabled. The check runs an active scan but does not change the interface state or print SSIDs/BSSIDs; a successful scan with zero visible beacons still passes. Scan permissions, driver support, and radio state can cause a preflight failure. Set `FA_RLWA_WIFI_ADAPTER=wlan0` to make the runner check that adapter automatically. This is an optional real-hardware preflight: the abstract simulator does not otherwise require a Wi-Fi card.

## Mininet-WiFi closed-loop validation

A separate backend runs the same registered policy loop against a Mininet-WiFi topology instead of the abstract simulator. It creates the configured APs and static STAs, starts one UDP iperf3 flow per STA by default, samples `iw` station/survey counters once per control cycle, applies guarded association moves, and writes the usual CSV trace/summary files. The first supported target is the static SC1 scenario; random-waypoint/scripted mobility and flash-crowd arrivals are rejected rather than silently approximated.

Install Mininet-WiFi and its Linux prerequisites following the [upstream instructions](https://github.com/intrig-unicamp/mininet-wifi). Run the command as root from a Python environment where both FA-RLWA and `mn_wifi` are importable:

```bash
sudo "$(command -v fa-rlwa)" validate-mininet \
  --scenario configs/scenarios/sc1_steady.yaml \
  --policy tiwa_fixed --seed 0 --duration 600 \
  --out results/mininet/sc1_tiwa_fixed_s0
```

Use `--no-traffic` only for topology/control smoke tests when iperf3 is not available. For a continuously running controller loop, add `--forever`; stop it with Ctrl-C (the generated iperf3 traffic runs for up to 24 hours). Channel changes are disabled by default because they can interrupt every client on an AP; enable them explicitly with `--allow-channel-changes`. Ctrl-C stops the loop and the Mininet network is torn down in `finally` cleanup. This backend uses Mininet's propagation model for candidate-AP RSSI, `iw` for serving-link and radio-counter telemetry, and survey busy/active time where available (otherwise it estimates own airtime). It does not modify the host's physical Wi-Fi interfaces or connect to a production SDN controller.

The emulator path is implemented and covered with fake-backend integration tests, but Mininet-WiFi, `iw`, and a Linux wireless namespace are not installed in the current sandbox. Therefore no real Mininet run or AC-07 result is claimed; run the command on a supported Linux host and review the resolved configuration, traces, and decision latency before treating it as validation evidence.

## Reproducibility and limits

- Pass explicit integer seeds; the simulator owns a `numpy.random.Generator` and does not use global random state.
- Scenario dimensions, traffic, random variation, and policy choices are synthetic. No real user data is collected.
- The simulated collision/airtime model is a fast approximation, not a packet-level IEEE 802.11 implementation.
- The original paper reports a real 5G-EmPOWER/LEDE testbed with LVAP handovers. A Mininet-WiFi backend now exercises the policy loop on an emulator, but it has not yet been run in this sandbox; physical-radio or production-controller integration remains future work.
- Original TiWA is ambiguous about thresholds, units, and its identical Algorithm 1 branches. The implementation follows the explicit interpretations in SRS Section 10.3; see the decision log.
- The provided AP coordinates and mobile path are approximate digitizations from the paper's floor-plan figures and are clearly marked in the scenario files.

## Source documents

- Requirements: [`SRS Forecast-Aware Deep RL for Multi-Indicator User Association in SDN-Based Wi-Fi Networks.md`](SRS%20Forecast-Aware%20Deep%20RL%20for%20Multi-Indicator%20User%20Association%20in%20SDN-Based%20Wi-Fi%20Networks.md)
- Base paper: [`TiWA_Achieving_Tetra_Indicator_Wi-Fi_Associations_.pdf`](TiWA_Achieving_Tetra_Indicator_Wi-Fi_Associations_.pdf)
- Reading notes: [`docs/paper-notes.md`](docs/paper-notes.md)
