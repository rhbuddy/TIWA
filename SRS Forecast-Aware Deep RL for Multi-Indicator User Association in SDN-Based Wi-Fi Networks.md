# Software Requirements Specification

## Forecast-Aware Deep Reinforcement Learning for Multi-Indicator User Association in SDN-Based Wi-Fi Networks

**Document version:** 1.0 (Draft) **Date:** 16 October 2026 **Project type:** Final Year Project (FYP) **Based on:** Manzoor et al., "TiWA: Achieving Tetra Indicator Wi-Fi Associations in Software Defined Wi-Fi Networks," IEEE Access, vol. 11, 2023 **Format:** Adapted from IEEE 830 / ISO/IEC/IEEE 29148

---

## 1. Introduction

### 1.1 Purpose

This document specifies the requirements for a software system that improves Wi-Fi user association and load balancing in software-defined Wi-Fi networks. The system extends the TiWA scheme by (a) correcting its algorithmic weaknesses, (b) adding a time-series forecaster that predicts AP-level load indicators, and (c) replacing hand-tuned heuristics with a reinforcement learning (RL) policy that uses both current and predicted indicators. The document is intended for the project team, the supervisor, and evaluators.

### 1.2 Scope

The system, referred to as **FA-RLWA** (Forecast-Aware RL Wi-Fi Association), will:

- Collect four indicators per AP (average RSSI, packet loss rate, throughput, channel occupancy) from a simulated SDN-controlled Wi-Fi network.
- Forecast those indicators a few seconds ahead.
- Decide, using a trained RL agent, whether to keep a client on its AP or move it to another AP.
- Execute decisions through an SDN controller abstraction with seamless handover.
- Compare itself against baseline schemes using a repeatable experimental framework.

**In scope:** uplink association, single-band operation (2.4 GHz initially), simulation-based training and evaluation, offline forecaster training, baseline implementations (client-driven, original TiWA).

**Out of scope (stretch goals or future work):** deployment on physical hardware, 5 GHz / Wi-Fi 6 features, downlink-aware association via 802.11k/v, RL-based channel assignment, multi-controller deployments.

### 1.3 Definitions, Acronyms, Abbreviations

| Term | Meaning |
| --- | --- |
| AP | Access Point |
| STA | Wireless station (client device) |
| SDN | Software Defined Networking |
| RSSI | Received Signal Strength Indicator |
| LVAP | Light Virtual Access Point (per-client virtual AP enabling seamless handover) |
| TiWA | Tetra Indicator Wi-Fi Association (baseline paper) |
| RL | Reinforcement Learning |
| DQN | Deep Q-Network |
| PPO | Proximal Policy Optimization |
| LSTM | Long Short-Term Memory network |
| MAE / RMSE | Mean Absolute Error / Root Mean Squared Error |
| JFI | Jain's Fairness Index |
| MDP | Markov Decision Process |
| FYP | Final Year Project |

### 1.4 References

1. S. Manzoor et al., "TiWA: Achieving Tetra Indicator Wi-Fi Associations in Software Defined Wi-Fi Networks," IEEE Access, vol. 11, 2023.
2. S. Manzoor et al., "WiMA: Towards a multi-criterion association in software defined Wi-Fi networks," Comput. Mater. Continua, vol. 75, no. 2, 2023.
3. E. Coronado et al., "Wi-Balance: Channel-aware user association in software-defined Wi-Fi networks," IEEE/IFIP NOMS, 2018.
4. B. Gomez et al., "WiMCA: Multi-indicator client association in software-defined Wi-Fi networks," Wireless Networks, vol. 27, 2021.
5. IEEE 830-1998 and ISO/IEC/IEEE 29148:2018, Requirements Engineering.
6. R. Jain et al., "A quantitative measure of fairness and discrimination for resource allocation in shared computer systems," DEC-TR-301, 1984.

### 1.5 Document Overview

Section 2 gives an overall description. Section 3 lists specific functional requirements. Section 4 covers external interfaces. Section 5 covers non-functional requirements. Section 6 defines the experimental and verification requirements. Section 7 contains risks, milestones, and deliverables. Sections 8 to 16 give the implementation specification: development environment, simulator model, indicators and baselines, data and forecaster, RL environment and agent, module interfaces, experiment protocol and tests, work breakdown, and the decision log. Section 17 is the traceability matrix and Section 18 the formula appendix.

---

## 2. Overall Description

### 2.1 Product Perspective

The product is a research software framework, not a commercial product. It sits in the application plane of an SDN architecture above a controller and below the experiment manager, and interacts with a simulated data plane.

```
+--------------------------------------------------------------+
| Experiment Manager (scenarios, seeds, metrics, reports)      |
+--------------------------------------------------------------+
| Application Plane: FA-RLWA                                   |
|   Forecaster | State Builder | RL Agent | Baseline Policies  |
+--------------------------------------------------------------+
| Control Plane: SDN Controller Abstraction                    |
|   Monitoring | Handover (LVAP) | Policy Dispatch             |
+--------------------------------------------------------------+
| Data Plane: Simulated APs and STAs                           |
|   (Abstract Python simulator, validated in Mininet-WiFi/ns-3)|
+--------------------------------------------------------------+
```

### 2.2 Problem Statement

IEEE 802.11 does not define AP selection, so clients typically pick the strongest RSSI, causing uneven load. TiWA improves this with four indicators, but it is reactive, combines unnormalized quantities in one formula (with identical if/else branches in Algorithm 1), relies on heuristic median thresholds, and was evaluated on only three APs in a single 2.4 GHz testbed. FA-RLWA addresses these limitations through forecasting and learned decision-making.

### 2.3 Product Functions (Summary)

- Network simulation with configurable topology, traffic, and mobility
- Indicator collection and storage
- Indicator forecasting
- RL-based association decision making
- Baseline policies (client-driven, TiWA reproduction, TiWA + forecaster)
- Handover execution with cost modelling
- Metrics computation, statistical analysis, and reporting

### 2.4 User Classes and Characteristics

| User class | Description | Technical level |
| --- | --- | --- |
| Researcher / Student | Configures scenarios, trains models, runs experiments | High (Python, networking, ML) |
| Supervisor / Evaluator | Reviews results, reproduces selected experiments | Medium to high |
| Future developer | Extends the framework (5 GHz, new models) | High |

### 2.5 Operating Environment

- OS: Ubuntu 22.04 LTS or later (Mininet-WiFi/ns-3 require Linux); Windows or macOS users may use a VM or WSL2
- Language: Python 3.10+ (C++ if ns-3 is used)
- Hardware: x86-64 CPU, minimum 16 GB RAM, optional NVIDIA GPU for LSTM/DQN training
- Key libraries: PyTorch, Stable-Baselines3, Gymnasium, scikit-learn, XGBoost, statsmodels, pandas, NumPy, matplotlib, Mininet-WiFi or ns-3 with ns3-gym

### 2.6 Design and Implementation Constraints

- Must not require modification of the IEEE 802.11 standard; client-side software must remain unmodified.
- Per-decision inference latency must fit within the 1 s control cycle (target below 0.2 s, matching TiWA's reported 0.19 s).
- All experiments must be reproducible from a fixed random seed.
- Open-source tools only, so the work can be reproduced without licensing cost.

### 2.7 Assumptions and Dependencies

- A centralized controller has accurate, timely per-AP and per-STA measurements.
- Seamless handover is available (modelled via LVAP-like abstraction).
- The abstract simulator is calibrated to be representative of Mininet-WiFi/ns-3 behaviour.
- Constant AP transmission power (no cell breathing), consistent with TiWA.
- Neighbour AP relationships (shared collision domains) are known to the controller.

---

## 3. Specific Functional Requirements

Priority: **M** = Must, **S** = Should, **C** = Could (stretch goal).

### 3.1 Network Simulation Module (SIM)

| ID | Requirement | Priority |
| --- | --- | --- |
| SIM-01 | The system shall simulate a configurable number of APs (3 to 10), each with position, channel, and fixed transmit power. | M |
| SIM-02 | The system shall simulate a configurable number of STAs (10 to 100), each with position and traffic demand. | M |
| SIM-03 | The system shall model path loss and shadowing to compute RSSI between every STA and AP. | M |
| SIM-04 | The system shall model airtime sharing and contention among STAs on the same AP and among APs sharing a collision domain. | M |
| SIM-05 | The system shall map RSSI to PHY rate (MCS) and derive per-STA throughput, loss, and retransmissions. | M |
| SIM-06 | The system shall support traffic profiles: constant bit rate, bulk TCP-like, VoIP-like, and video-like. | S |
| SIM-07 | The system shall support mobility models: static, random waypoint, and scripted paths. | M |
| SIM-08 | The system shall support flash-crowd scenarios in which many STAs appear near a subset of APs within a short interval. | M |
| SIM-09 | The system shall advance in discrete time steps of 1 second (configurable). | M |
| SIM-10 | The system shall be seedable so identical seeds produce identical runs. | M |
| SIM-11 | The system shall provide equivalent scenarios in Mininet-WiFi or ns-3 for validating the final policy. | S |

### 3.2 Monitoring and Indicator Module (MON)

| ID | Requirement | Priority |
| --- | --- | --- |
| MON-01 | The system shall compute, once per control cycle, the four indicators for every AP: average uplink RSSI, packet loss rate, throughput (Mbps), and channel occupancy (%). | M |
| MON-02 | Channel occupancy shall be computed as the sum of estimated transmission times of all STAs on the channel divided by the elapsed time, including contribution from APs in the same collision domain. | M |
| MON-03 | The system shall store all indicator time series with timestamps in a structured format (CSV or Parquet). | M |
| MON-04 | The system shall provide a smoothed view of each indicator (moving average over a configurable window, default 10 reports) for use by baseline policies. | M |
| MON-05 | The system shall normalize indicators to a common range \[0, 1\] with a defined orientation in which higher is always better. | M |
| MON-06 | The system shall record per-STA RSSI to every AP in range. | M |

### 3.3 Forecasting Module (FCT)

| ID | Requirement | Priority |
| --- | --- | --- |
| FCT-01 | The system shall forecast AP load, channel occupancy, packet loss rate, and throughput for horizons of at least 5, 10, and 30 seconds. | M |
| FCT-02 | The system shall implement at least four forecasters: EWMA/moving-average baseline, ARIMA, gradient-boosted trees (XGBoost or Random Forest) on lag features, and an LSTM or GRU. | M |
| FCT-03 | The system shall provide a training pipeline that generates a dataset from simulation runs, splits it by scenario (train/validation/test), and trains each forecaster. | M |
| FCT-04 | The system shall report MAE, RMSE, and MAPE per forecaster, per indicator, and per horizon. | M |
| FCT-05 | The system shall allow the forecaster to be swapped through configuration without code changes. | S |
| FCT-06 | The system shall support online inference with latency below 50 ms per prediction on CPU. | S |
| FCT-07 | The system shall provide a client RSSI trend predictor (mobility prediction) that estimates which AP a moving STA will be closest to in the next k seconds. | C |

### 3.4 RL Environment Module (ENV)

| ID | Requirement | Priority |
| --- | --- | --- |
| ENV-01 | The system shall expose the simulation as a Gymnasium-compatible environment with `reset()`, `step(action)`, and standard observation/action spaces. | M |
| ENV-02 | The observation shall include, per AP: current indicators, predicted indicators (when enabled), and client count; and for the candidate STA: its RSSI to every AP and its current AP. | M |
| ENV-03 | The action space shall be discrete: *stay* or *move candidate STA to AP j*. | M |
| ENV-04 | The environment shall select the candidate STA by a configurable rule (default: a STA on the most loaded AP, rotating among eligible STAs). | M |
| ENV-05 | The reward shall be `R = w1*throughput_norm + w2*JFI - w3*handover_cost - w4*retransmission_rate`, with weights configurable. | M |
| ENV-06 | The environment shall apply a handover cost (a short period of reduced throughput for the moved STA) to discourage ping-pong. | M |
| ENV-07 | The environment shall support enabling or disabling forecast features in the observation to support ablation studies. | M |
| ENV-08 | The environment shall terminate episodes after a configurable duration (default 600 s) and support randomized scenario sampling per episode. | M |
| ENV-09 | The environment shall mask or penalize invalid actions (moving a STA to an AP out of range). | M |

### 3.5 RL Agent Module (RLA)

| ID | Requirement | Priority |
| --- | --- | --- |
| RLA-01 | The system shall train a DQN agent (or variant such as Double/Dueling DQN) using Stable-Baselines3 or an equivalent library. | M |
| RLA-02 | The system shall support PPO as an alternative algorithm. | S |
| RLA-03 | The system shall log training curves (episode reward, loss, exploration rate) and save checkpoints periodically. | M |
| RLA-04 | The system shall support loading a trained policy for evaluation without training dependencies. | M |
| RLA-05 | Training shall be repeatable across at least 5 random seeds, and results shall report mean and spread. | M |
| RLA-06 | The trained policy shall run inference in under 20 ms per decision on CPU. | S |
| RLA-07 | The system shall support RL-based channel assignment as an extended action space. | C |

### 3.6 Baseline Policy Module (BAS)

| ID | Requirement | Priority |
| --- | --- | --- |
| BAS-01 | The system shall implement a **client-driven** policy (each STA joins the AP with the strongest RSSI). | M |
| BAS-02 | The system shall implement the **original TiWA** policy as published, including indicator thresholds (max-min greater than median), the 10-report averaging, and the Algorithm 2 channel assignment. | M |
| BAS-03 | The system shall document any interpretation made where the paper is ambiguous (for example, identical if/else branches in Algorithm 1; sign and units of the combined score). | M |
| BAS-04 | The system shall implement a **corrected TiWA** with normalized, weighted indicators. | S |
| BAS-05 | The system shall implement **TiWA + forecaster** (TiWA decision formula fed with predicted values). | M |
| BAS-06 | The system shall implement a **random** policy as a sanity check. | M |
| BAS-07 | The system shall implement WiMA and/or Wi-Balance for comparison where published details allow. | C |

### 3.7 Controller and Handover Module (CTL)

| ID | Requirement | Priority |
| --- | --- | --- |
| CTL-01 | The controller abstraction shall invoke the active policy once per control cycle. | M |
| CTL-02 | The controller shall execute a move decision by transferring the STA's association to the target AP with the handover cost applied. | M |
| CTL-03 | The controller shall record every handover (time, STA, source AP, target AP, reason). | M |
| CTL-04 | The controller shall enforce a configurable minimum dwell time per STA to limit ping-pong. | S |
| CTL-05 | The controller shall measure and log decision latency per cycle. | M |

### 3.8 Experiment Management and Analysis Module (EXP)

| ID | Requirement | Priority |
| --- | --- | --- |
| EXP-01 | The system shall run any combination of scenario, policy, and seed from a configuration file. | M |
| EXP-02 | The system shall compute: aggregate throughput, per-STA throughput, JFI, retransmissions per frame, channel occupancy, standard deviation of occupancy across channels, average PHY rate, handover count, and ping-pong rate. | M |
| EXP-03 | The system shall run at least 10 independent seeds per configuration and report mean, standard deviation, and 95% confidence intervals. | M |
| EXP-04 | The system shall perform statistical significance tests (for example, paired t-test or Wilcoxon signed-rank) between policies. | S |
| EXP-05 | The system shall generate plots (bar charts with error bars, time series, CDFs) and tables in a reproducible way. | M |
| EXP-06 | The system shall support a **densification scenario** in which a baseline policy runs with one or two extra APs, to test the claim that better association can delay densification. | M |
| EXP-07 | The system shall support train/test splits by scenario type to measure generalization to unseen traffic and mobility patterns. | M |
| EXP-08 | The system shall support ablation runs: RL without forecast vs. RL with forecast. | M |
| EXP-09 | The system shall store all raw logs and configurations with each result for traceability. | M |

---

## 4. External Interface Requirements

### 4.1 User Interfaces

- **UI-01:** Command-line interface for training, evaluation, and plotting (for example, `python run.py --scenario flash_crowd --policy rl_forecast --seed 3`).
- **UI-02:** YAML or JSON configuration files for scenarios, hyperparameters, and reward weights.
- **UI-03 (Could):** A Jupyter notebook or simple dashboard to visualize AP load and handovers over time.

### 4.2 Software Interfaces

| Interface | Description |
| --- | --- |
| Gymnasium API | Standard RL environment interface used by Stable-Baselines3 |
| Mininet-WiFi Python API / ns3-gym | Validation back-end for realistic simulation |
| PyTorch / scikit-learn / statsmodels | Model training and inference |
| File system | Parquet/CSV for datasets, `.zip` or `.pt` for model checkpoints, YAML for configs |

### 4.3 Hardware Interfaces

None required. Optional GPU (CUDA) for faster training. Physical AP and Raspberry Pi validation is future work.

### 4.4 Communication Interfaces

If Mininet-WiFi is used, the application communicates with the emulated controller through local Python calls or a local REST/socket interface; no external network access is required.

---

## 5. Non-Functional Requirements

### 5.1 Performance

| ID | Requirement |
| --- | --- |
| NFR-P1 | End-to-end decision latency (monitoring to action) shall be below 1 s, with a target below 0.2 s. |
| NFR-P2 | The abstract simulator shall run at least 100x faster than real time to make RL training feasible. |
| NFR-P3 | A full training run on the reference configuration (5 APs, 40 STAs) shall complete within 24 hours on the reference machine. |

### 5.2 Reliability and Reproducibility

| ID | Requirement |
| --- | --- |
| NFR-R1 | Fixed seeds shall produce identical results across runs on the same software versions. |
| NFR-R2 | Dependency versions shall be pinned (`requirements.txt` or `environment.yml`). |
| NFR-R3 | The system shall handle invalid configurations by failing early with a clear error message. |

### 5.3 Maintainability and Extensibility

| ID | Requirement |
| --- | --- |
| NFR-M1 | The code shall be modular, with separate packages for simulation, forecasting, RL, baselines, and analysis. |
| NFR-M2 | New policies, forecasters, and scenarios shall be addable by implementing a defined interface without modifying core modules. |
| NFR-M3 | The code shall follow PEP 8 and include docstrings for public functions. |
| NFR-M4 | Unit tests shall cover the simulator, metrics (including JFI), and baseline policies, with at least 70% coverage on core modules. |

### 5.4 Portability

| ID | Requirement |
| --- | --- |
| NFR-T1 | The abstract simulator, forecaster, and RL components shall run on Linux, and preferably on Windows/macOS without Mininet-WiFi. |
| NFR-T2 | A setup script or container (Docker) should allow environment recreation. |

### 5.5 Usability

| ID | Requirement |
| --- | --- |
| NFR-U1 | A new user shall be able to reproduce the main result by following a README in under 2 hours (excluding training time). |
| NFR-U2 | Output plots and tables shall have labelled axes, units, and legends. |

### 5.6 Security and Data Handling

| ID | Requirement |
| --- | --- |
| NFR-S1 | The system shall use only synthetic data; no real user traffic or personal data shall be collected. |
| NFR-S2 | Third-party libraries shall be sourced from trusted package indexes with pinned versions. |

### 5.7 Scalability

| ID | Requirement |
| --- | --- |
| NFR-C1 | The simulator and policies shall scale to at least 10 APs and 100 STAs without exceeding the decision latency target. |

---

## 6. Verification and Evaluation Requirements

### 6.1 Systems under Comparison

| Label | Description |
| --- | --- |
| S1 | Client-driven (strongest RSSI) |
| S2 | Original TiWA (reproduced) |
| S3 | Corrected TiWA (normalized and weighted), optional |
| S4 | TiWA + forecaster |
| S5 | RL only (no forecast) |
| S6 | **RL + forecast (proposed)** |
| S7 | S2 with additional APs (densification baseline) |

### 6.2 Scenarios

| ID | Scenario | Purpose |
| --- | --- | --- |
| SC1 | Steady load, static STAs | Reproduces TiWA Experiment 1 |
| SC2 | Mobile STAs along fixed paths | Reproduces TiWA Experiment 2 |
| SC3 | Two APs in one collision domain | Reproduces TiWA Experiment 3 |
| SC4 | Flash crowd | Stress test for bursty load |
| SC5 | Mixed traffic with random waypoint mobility | Realism and generalization |
| SC6 | Scaled network (more APs and STAs) | Scalability |

### 6.3 Acceptance Criteria

| ID | Criterion |
| --- | --- |
| AC-01 | The reproduced TiWA shows the qualitative trends reported in the paper (better throughput and fairness than client-driven in SC1 to SC3). |
| AC-02 | The best forecaster achieves lower MAE than the moving-average baseline for at least the 10 s horizon on held-out scenarios. If it does not, the result shall be reported and analysed. |
| AC-03 | The RL agent converges (stable mean reward across seeds) and outperforms the random policy and client-driven baseline. |
| AC-04 | S6 is compared against S2 and S5 with 95% confidence intervals across at least 10 seeds, and the ablation result (forecast vs. no forecast) is reported honestly, whether positive or negative. |
| AC-05 | The densification comparison (S7) is completed and discussed. |
| AC-06 | Decision latency of S6 stays below 1 s per cycle. |
| AC-07 | The final policy is validated in at least one scenario in Mininet-WiFi or ns-3. |
| AC-08 | Code, configuration, and instructions are delivered so results are reproducible. |

### 6.4 Key Performance Indicators

Aggregate throughput, JFI, retransmissions per frame, average PHY rate, handover count and ping-pong rate (a ping-pong is defined as a STA returning to its previous AP within 10 s), channel occupancy standard deviation, forecast MAE/RMSE, and decision latency.

---

## 7. Other Requirements

### 7.1 Known Limitations of the Baseline Paper to Address

| Limitation in TiWA | How this project responds |
| --- | --- |
| Identical if/else branches in Algorithm 1 | Document interpretation (BAS-03); replace with learned policy |
| Unnormalized sum of heterogeneous indicators | Normalization (MON-05); RL learns the trade-off |
| Heuristic median thresholds | Replaced by learned decision rule |
| Reactive behaviour | Forecaster supplies predicted state |
| Small testbed (3 APs, one band) | Scaled scenarios SC5, SC6 |
| No repeated trials or confidence intervals | EXP-03, EXP-04 |
| Densification claim not tested directly | EXP-06, system S7 |

### 7.2 Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| RL training is unstable or slow | High | High | Start with a tiny setup (3 APs, 10 STAs); use a fast abstract simulator; compare with random and heuristic policies |
| Sim-to-real gap | Medium | Medium | Validate in Mininet-WiFi/ns-3; state limits clearly |
| Forecasting adds little over averaging | Medium | Medium | Report as a valid finding; test on bursty and mobile scenarios where prediction should help |
| Reward hacking (for example, starving a STA) | Medium | High | Fairness term; per-STA logging; inspect behaviour |
| Ambiguity in reproducing TiWA | High | Medium | Document assumptions; contact authors if possible |
| Scope creep | High | High | Freeze scope after month 2; treat Could items as stretch |
| Compute limits | Medium | Medium | Use small networks, cloud/Colab credits, or university GPU access |

### 7.3 Suggested Milestones

| Months | Milestone |
| --- | --- |
| 1-2 | Literature review; abstract simulator; reproduce TiWA and client-driven baselines |
| 3-4 | Dataset generation; forecaster training and comparison; Gym environment |
| 5-6 | DQN training without and with forecasts; reward tuning |
| 7-8 | Full evaluation, ablations, densification study, Mininet-WiFi/ns-3 validation |
| 9 | Thesis writing, final report, defense preparation |

### 7.4 Deliverables

1. Source code repository with README and pinned dependencies
2. Generated datasets and trained model checkpoints
3. Configuration files for all scenarios
4. Experimental results, logs, and plots
5. Final report/thesis and presentation

---

## 8. Implementation Guide

Sections 8 to 16 turn the requirements above into build-ready specifications, so a new team member can implement the project from this document alone. Every value marked "default" is a starting point that may be tuned; each tuning must be recorded in the decision log (Section 16).

### 8.1 Development Environment

| Item | Specification |
| --- | --- |
| OS | Ubuntu 22.04 LTS (native, VM, or WSL2) |
| Python | 3.10 or 3.11 |
| Environment | `venv` + `pip` (or conda); exact versions pinned in `requirements.txt` at project kickoff |
| Core libraries (minimum) | numpy 1.26, scipy 1.11, pandas 2.1, pyarrow 14, pyyaml 6, matplotlib 3.8 |
| ML libraries (minimum) | PyTorch 2.2, scikit-learn 1.4, xgboost 2.0, statsmodels 0.14 |
| RL libraries (minimum) | gymnasium 0.29, stable-baselines3 2.3, sb3-contrib 2.3 (action masking) |
| Quality tools | pytest, pytest-cov, ruff, black, mypy |
| Validation emulators | Mininet-WiFi (current release) or a recent ns-3 release with ns3-gym |
| Tracking | TensorBoard for training curves; Git for code; optional DVC for datasets |

### 8.2 Quick Start for a New Contributor

```bash
git clone <repo-url> fa-rlwa && cd fa-rlwa
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest -q                                  # all tests must pass before any work
python -m fa_rlwa.experiments.runner \
    --scenario configs/scenarios/sc1_steady.yaml \
    --policy client_driven --seed 0 --out results/demo
python -m fa_rlwa.experiments.plots --run results/demo
```

Expected result: `results/demo/` contains `ap_indicators.parquet`, `sta_log.parquet`, `handover_log.csv`, `metrics_summary.csv`, and a `figures/` folder with throughput-over-time, per-AP load, and fairness-over-time plots.

### 8.3 Repository Structure

```text
fa-rlwa/
  README.md  requirements.txt  pyproject.toml
  configs/
    scenarios/   sc1_steady.yaml ... sc6_scaled.yaml
    policies/    tiwa_original.yaml tiwa_fixed.yaml ...
    forecast/    ewma.yaml arima.yaml xgb.yaml lstm.yaml
    rl/          dqn_forecast.yaml dqn_noforecast.yaml ppo.yaml
  src/fa_rlwa/
    sim/         propagation.py mobility.py traffic.py mac.py network.py
    monitor/     indicators.py normalize.py store.py
    forecast/    base.py ewma.py arima.py xgb.py lstm.py dataset.py train.py evaluate.py
    env/         wifi_env.py obs.py candidate.py reward.py
    rl/          train_dqn.py train_ppo.py evaluate.py callbacks.py
    policies/    base.py client_driven.py tiwa.py tiwa_fixed.py random_policy.py rl_policy.py
    controller/  controller.py handover.py
    experiments/ runner.py metrics.py stats.py plots.py report.py
    validation/  mininet_wifi/ ns3/
  tests/         unit/ integration/
  data/          (git-ignored: generated datasets)
  models/        (git-ignored or DVC: trained forecasters and RL policies)
  results/       (git-ignored: run outputs)
  docs/          SRS, design notes, decision log
```

### 8.4 Coding and Collaboration Rules

- Work on feature branches (`feat/<module>-<topic>`); merge to `main` through a pull request reviewed by at least one other member; CI runs `pytest`, `ruff`, and `mypy`.
- All randomness comes from a `numpy.random.Generator` created from the configured seed and passed explicitly. No global random state.
- Every function in public modules has type hints and a docstring stating units.
- Every run writes its resolved config, git commit hash, and library versions next to its outputs.
- Use `logging`, not `print`.
- Trained artifacts carry a `meta.yaml` (training seed, data version, git commit, metrics).

**Unit and naming conventions**

| Quantity | Internal unit | Notes |
| --- | --- | --- |
| Time | seconds (integer step index `t`, step = 1 s) |  |
| Power, RSSI | dBm | RSSI is negative |
| Throughput, demand | Mbps |  |
| Loss, occupancy, fairness | fraction in \[0, 1\] | Convert to % only in reports and plots |
| Distance | metres |  |
| IDs | zero-based integers (`ap_id`, `sta_id`) | `ap_id = -1` means unassociated |

---

## 9. Simulation Model Specification

The abstract simulator is a discrete-time model: one step is one second. It is deliberately simple and fast (target 100x real time or better) and must be calibrated against the published TiWA results (Section 9.8) and validated in Mininet-WiFi or ns-3 (Section 14.6).

### 9.1 One Simulation Step

1. Apply pending controller actions (handovers, channel changes) and start their outage timers.
2. Move STAs according to their mobility model; add arriving STAs and remove departing ones.
3. Update shadowing for every STA-AP pair and compute the RSSI matrix.
4. For each associated STA, look up the PHY rate from its RSSI (Section 9.4). STAs below the association threshold are disconnected for this step.
5. Compute collision probabilities and per-attempt failure probabilities (Section 9.5), then recompute PHY rates with the collision adjustment. Repeat this once more (two passes) for consistency.
6. Compute airtime demand per STA, per AP, and per collision neighbourhood; compute the scale factor and per-STA throughput.
7. Apply outage penalties to STAs that moved or whose AP changed channel in this step.
8. Compute the four indicators per AP (Section 10.1), add measurement noise, and append the report to the indicator store.
9. Write the step to the logs.

### 9.2 Default Parameters

| Parameter | Symbol | Default |
| --- | --- | --- |
| Step length | dt | 1 s |
| Floor area SC1 to SC3 |  | 15 m x 9 m (testbed of the TiWA paper, Figures 3 to 5) |
| Floor area SC4, SC5 / SC6 |  | 30 m x 20 m / 60 m x 40 m |
| Path-loss at 1 m | PL0 | 40 dB |
| Path-loss exponent | n | 3.0 |
| Shadowing std. deviation / correlation per step | sigma, rho | 4 dB / 0.95 |
| STA transmit power / AP transmit power |  | 15 dBm / 20 dBm |
| Association threshold (minimum RSSI) |  | -82 dBm |
| AP-to-AP carrier-sense threshold |  | -82 dBm |
| MAC efficiency | eta | 0.6 |
| Maximum usable channel utilisation | C\_max | 0.9 |
| Per-STA transmission probability | tau | 0.01 |
| Collision probability cap |  | 0.5 |
| Retry limit | R | 7 |
| Measurement noise on reported RSSI |  | Gaussian, std. 2 dB |
| Measurement noise on reported throughput |  | multiplicative, +/- 5% (uniform) |
| Warm-up (discarded from metrics) |  | 60 s |
| Run length (evaluation) |  | 1200 s |
| Channels (2.4 GHz, 20 MHz) |  | 1, 6, 11 (non-overlapping; no adjacent-channel leakage) |

### 9.3 Propagation

```latex
PL(d) = PL_0 + 10\,n\,\log_{10}(d / 1\,\mathrm{m}) + X_{\sigma}, \qquad RSSI_{ij} = P_{tx,i} - PL(d_{ij})
```

Shadowing evolves per STA-AP pair as an AR(1) process: X(t+1) = rho \* X(t) + sqrt(1 - rho^2) \* sigma \* eps, with eps drawn from a standard normal. Reported RSSI adds measurement noise (Section 9.2). A STA can associate only with APs for which RSSI is at least -82 dBm.

### 9.4 PHY Rate Table (802.11n, 20 MHz, 1 spatial stream)

The rate follows the highest row whose RSSI threshold is met. Thresholds are typical receiver sensitivities; replace them with the datasheet values of the target NIC (the TiWA testbed used an Atheros AR9220) if available.

| RSSI at least (dBm) | PHY rate (Mbps) |
| --- | --- |
| -64 | 65.0 |
| -65 | 58.5 |
| -66 | 52.0 |
| -70 | 39.0 |
| -74 | 26.0 |
| -77 | 19.5 |
| -79 | 13.0 |
| -82 | 6.5 |
| below -82 | not associable |

The rate used for the lookup is the effective RSSI: RSSI\_eff = RSSI - 10 \* p\_coll (in dB). This emulates rate adaptation (Minstrel) lowering the rate when collisions are frequent, which the paper observes.

### 9.5 Contention, Airtime, and Throughput

For each STA i on AP k with offered load lambda\_i (Mbps):

1. **Collision neighbourhood.** N(k) is AP k plus every AP on the same channel whose AP-to-AP RSSI (AP power, no shadowing) is at least -82 dBm. APs on different channels never interact.
2. **Active STAs** n\_k = number of STAs with lambda > 0 on all APs in N(k).
3. **Collision probability:** p\_coll = min(0.5, 1 - (1 - tau)^(n\_k - 1)).
4. **Channel error probability:** p\_err(RSSI) = 0.01 + 0.19 \* clip((-65 - RSSI) / 17, 0, 1). This gives 0.01 at -65 dBm or better and 0.20 at -82 dBm.
5. **Per-attempt failure probability:** p\_fail = 1 - (1 - p\_coll) \* (1 - p\_err).
6. **Goodput at full airtime:** G\_i = R\_i \* eta \* (1 - p\_fail), where R\_i is the PHY rate.
7. **Airtime demand:** a\_i = lambda\_i / G\_i. AP airtime A\_k = sum of a\_i over STAs on k.
8. **Neighbourhood airtime:** A\_N(k) = sum of A\_l over l in N(k).
9. **Scale factor:** s\_k = min(1, C\_max / A\_N(k)).
10. **Throughput:** T\_i = s\_k \* lambda\_i.
11. **Expected retransmissions per frame:** p\_fail / (1 - p\_fail). **Residual loss:** p\_fail^(R+1).

Because low-rate STAs need more airtime for the same load, this model reproduces the 802.11 performance anomaly, which is the reason that moving distant STAs matters. The "packet loss rate" indicator uses the per-attempt failure rate p\_fail, which is what real MAC counters (tx retries, tx failed) expose.

### 9.6 Mobility and Traffic

| Item | Specification |
| --- | --- |
| Static | STA keeps its position |
| Random waypoint | Speed uniform in 0.5 to 1.5 m/s, pause uniform in 0 to 30 s, waypoints uniform in the floor area |
| Scripted path | List of (x, y, hold-seconds) waypoints from the scenario file (reproduces Figure 4 of the paper) |
| Flash crowd | N\_f STAs arrive within a window \[t0, t0 + w\], positions Gaussian around a hotspot (std. 3 m), dwell time exponential with mean 120 s |
| CBR traffic | Constant offered load, default 1 Mbps per STA (see Section 9.8, note on the paper's figures) |
| VoIP-like | 0.1 Mbps constant |
| Video-like | Mean 4 Mbps, AR(1) variation of +/- 30% |
| Bulk-like | 8 Mbps offered (saturating) |

### 9.7 Handover and Channel-Change Costs

| Event | Cost |
| --- | --- |
| Handover with LVAP-style seamless transfer (default) | The moved STA loses 0.2 s of service: its throughput in that step is multiplied by 0.8 |
| Handover without seamless support (option for sensitivity study) | 2.0 s outage (two steps at zero throughput) |
| Channel change (CSA) | All STAs of the AP lose 0.5 s of service |
| Minimum dwell time after a handover | 10 s (controller refuses earlier moves of the same STA) |
| Minimum interval between channel changes of the same AP | 60 s |

A new STA associates initially with the AP having the strongest RSSI, unless the active policy overrides `initial_ap`.

### 9.8 Calibration and Validation of the Simulator

The simulator must reproduce the qualitative and approximate quantitative behaviour in the paper before any new method is trusted.

- **Note on the paper's numbers.** The paper states 3 Mbps per device and 30 Mbps total with 30 devices, which is inconsistent. Default here: 1 Mbps per device (30 devices give 30 Mbps total), matching the stated total and the throughput levels in Figure 6(a). Record any other choice in the decision log.
- **Target (Experiment 1, Aspect 1, read approximately from Figure 6(a)):** client-driven about 15 to 16 Mbps, TiWA about 25 Mbps. Acceptance: the simulated ordering matches (TiWA above client-driven) and each value is within 25% of the paper.
- **Sanity checks:** with one lightly loaded AP in range, throughput equals offered load; throughput never exceeds eta times the highest PHY rate; doubling offered load on a saturated AP does not increase its throughput.
- **Parameter fitting:** if values differ by more than 25%, adjust only n, tau, and eta, one at a time, and record each change.

---

## 10. Indicators and Baseline Algorithms

### 10.1 Indicator Definitions

Indicators are computed once per step for every AP k from the simulator state (or, in the emulator, from the measurements listed in Section 14.6).

| Indicator | Symbol | Definition | Unit | Normalized form (higher is better) |
| --- | --- | --- | --- | --- |
| Average RSSI | r\_k | Mean uplink RSSI over STAs associated with k; if none, -60 dBm | dBm | r\_hat = clip((r + 90) / 50, 0, 1) |
| Packet loss rate | l\_k | Failed attempts divided by total attempts over STAs of k (per-attempt failure, p\_fail) | fraction | l\_hat = 1 - clip(l / 0.5, 0, 1) |
| Throughput | t\_k | Sum of T\_i over STAs of k | Mbps | t\_hat = 1 - clip(t / 39, 0, 1) (headroom; 39 Mbps = 0.6 x 65 Mbps) |
| Channel occupancy | o\_k | min(1, A\_N(k)), the neighbourhood airtime | fraction | o\_hat = 1 - o |
| Client count | n\_k | Number of associated STAs | count | n / N\_max, with N\_max = 100 |
| AP airtime load | a\_k | A\_k, the airtime of the AP's own STAs | fraction | 1 - a |

The controller keeps the last 10 reports per AP and exposes both the raw latest report and the 10-report moving average, as in the paper.

### 10.2 BAS-01: Client-Driven Policy

Every STA associates with the AP of highest RSSI at arrival and re-evaluates only when its current RSSI drops below -82 dBm, at which point it joins the strongest available AP. No central control.

### 10.3 BAS-02: TiWA as Published

The paper leaves several points open. The table lists each ambiguity and the interpretation this project adopts, so that the reproduction is explicit and can be revisited.

| ID | Ambiguity in the paper | Adopted interpretation (configurable) |
| --- | --- | --- |
| I-1 | Algorithm 1 uses max, min, and median without saying over what | Over the 10-report smoothed values across APs. Alternative `threshold_mode: temporal` uses max, min, median over the last 10 reports of each AP |
| I-2 | The two branches of Algorithm 1 contain identical formulas | Branch A (an AP-level indicator is imbalanced): re-evaluate existing STAs on the overloaded AP. Branch B (no imbalance): apply the formula only to newly arriving STAs |
| I-3 | The score adds load, loss, throughput, and occupancy with different units | Load, loss, occupancy as fractions; throughput divided by 39 Mbps; all non-negative "cost" terms. RSSI is negative, so maximising RSSI x cost prefers strong signal and low cost |
| I-4 | Which STAs move, and how many | All STAs on the AP holding the maximum of the triggered indicator, strongest-alternative first; at most 2 moves per cycle |
| I-5 | "No migration if there isn't a better option" | Move only if the best score exceeds the current score by more than 5% of its magnitude |
| I-6 | Paper describes ARF as additive-increase multiplicative-decrease | Packet loss comes from MAC counters (p\_fail); ARF is not modelled |
| I-7 | Channel set and reassignment rate | Channels 1, 6, 11; at most one reassignment per AP per 60 s |

```python
# Executed once per control cycle (1 s).
def tiwa_original(state, cfg):
    s = {ind: [mean(last10(k, ind)) for k in APs] for ind in INDICATORS}   # I-1
    trig = {ind: (max(s[ind]) - min(s[ind])) > median(s[ind]) for ind in INDICATORS}
    actions = []

    if trig["rssi"] or trig["loss"] or trig["tput"]:                         # Branch A (I-2)
        for ap in aps_with_max(s, triggered=trig):
            for sta in stas_sorted_by_best_alternative(ap):                  # I-4
                score = lambda k: rssi[sta][k] * (occ[k] + load[k] + loss[k] + tput[k] / 39.0)  # I-3
                best = argmax_over_in_range_aps(score)
                # scores are negative: greater means better
                if best != cur(sta) and score(best) > score(cur(sta)) + 0.05 * abs(score(cur(sta))):  # I-5
                    actions.append(Handover(sta, best))
                if len(actions) >= 2:
                    return actions

    if trig["occ"]:                                                          # Algorithm 2
        actions += assign_channels(state)
    return actions                                                           # Branch B: initial_ap() only
```

**Channel assignment (Algorithm 2).** Order the APs by number of neighbours (descending), then by channel utilisation (ascending). For each AP in that order, remove the channels used by its neighbours from {1, 6, 11} and pick one of the remaining channels. If none remain, pick the channel with the lowest summed neighbour occupancy. Complexity is O(x^2) in the number of APs x, as in the paper.

### 10.4 BAS-04: Corrected TiWA

The corrected policy uses the normalized indicators so that higher is always better, and weights them explicitly.

```latex
S_{jk} = w_r\,\hat r_{jk} + w_l\,\hat l_k + w_o\,\hat o_k + w_t\,\hat t_k, \qquad w_r + w_l + w_o + w_t = 1
```

- A STA j on AP c moves to AP k only if S\_jk exceeds S\_jc by more than delta = 0.05, k is in range, and the dwell time has elapsed.
- Default weights are 0.25 each. Tune them by grid search over the weight simplex (step 0.1) on validation scenarios (seeds 100 to 119) and record the result.
- Trigger: an AP is overloaded if its occupancy exceeds 0.7 or its loss exceeds 0.15. Candidates are the STAs of overloaded APs. At most 2 moves per cycle.

### 10.5 BAS-05: TiWA plus Forecaster

Use the TiWA decision logic (original or corrected, set by configuration) but replace each indicator x with x\_used = (1 - beta) \* x\_now + beta \* x\_pred, where x\_pred is the forecast at the 10 s horizon and beta = 0.5 by default.

### 10.6 BAS-06: Random Policy

With probability 0.02 per cycle, choose a random STA that has passed its dwell time and move it to a random in-range AP. This is the sanity floor for RL.

---

## 11. Data and Forecaster Specification

### 11.1 Log Schemas

All logs are written per run into `results/<run_id>/` (Parquet unless noted). `run_id` has the form `<scenario>_<policy>_s<seed>`.

| File | Columns |
| --- | --- |
| `ap_indicators.parquet` | run\_id, t, ap\_id, channel, n\_clients, rssi\_dbm, loss\_frac, tput\_mbps, occ\_frac, airtime\_own\_frac, demand\_mbps |
| `sta_log.parquet` | run\_id, t, sta\_id, ap\_id, x\_m, y\_m, rssi\_dbm\_ap0 ... rssi\_dbm\_ap{K-1}, phy\_mbps, offered\_mbps, tput\_mbps, p\_fail, retx\_per\_frame, traffic\_type |
| `handover_log.csv` | run\_id, t, sta\_id, from\_ap, to\_ap, policy, trigger |
| `metrics_summary.csv` | run\_id, scenario, policy, seed, agg\_tput\_mbps, offered\_mbps, jfi, retx\_per\_frame, occ\_mean, occ\_std, phy\_mean\_mbps, handovers, pingpongs, decision\_ms\_mean, decision\_ms\_p95 |
| `config_resolved.yaml` | Full resolved configuration, git commit, library versions |

Metric definitions: aggregate throughput is the mean over time of the sum of T\_i; JFI is computed per step over x\_i = T\_i / lambda\_i (satisfaction ratio) and averaged over time; occupancy standard deviation is the standard deviation across channels at each step, averaged over time; a ping-pong is a STA returning to its previous AP within 10 s.

### 11.2 Forecast Dataset Generation

| Item | Specification |
| --- | --- |
| Generating policies | client-driven, original TiWA, and random (so the data covers different load patterns, not only one policy's) |
| Scenarios | SC1, SC2, SC4, SC5 (training and validation); SC3 and SC6 are held out as unseen |
| Seeds | Training 0 to 79; validation 80 to 99; in-distribution test 200 to 219 (SC1, SC2, SC4, SC5); unseen-scenario test 200 to 219 (SC3, SC6) |
| Run length | 1200 s per run; first 60 s discarded |
| Size | About 960 training runs (4 scenarios x 3 policies x 80 seeds) |
| Windows | One sample every 5 s per AP |

Splits are made by seed and scenario, never by random rows, to avoid leakage between neighbouring time steps.

### 11.3 Windows and Features

- **Input window:** last 30 steps of six series per AP: r, l, t, o, n\_k, and total offered demand of the AP's STAs.
- **Targets:** r, l, t, o at horizons h = 5, 10, 30 s ahead (12 values per sample).
- **Scaling:** z-score using training-set statistics, stored in `meta.yaml`.
- **Tabular features for tree models:** the 30 lags of each target series, rolling mean and standard deviation over 5, 10, and 30 steps, first differences, n\_k, and demand.
- **Storage:** `data/forecast/<version>/{train,val,test_id,test_unseen}.npz` with arrays `X` \[N, 30, 6\], `y` \[N, 3, 4\], and `meta` (run\_id, t, ap\_id).

### 11.4 Models and Hyperparameters

| Model | Configuration (defaults) |
| --- | --- |
| EWMA (baseline) | alpha chosen from {0.1, 0.2, ..., 0.9} on the validation set; forecast equals the current EWMA for every horizon |
| Moving average (paper's baseline) | Mean of the last 10 reports |
| ARIMA | `statsmodels` ARIMA per AP and indicator; (p, d, q) chosen by AIC over p, q in 0..3 and d in 0..1; refitted every 60 s on the last 300 s |
| XGBoost | One model per (indicator, horizon): 300 trees, max depth 6, learning rate 0.05, subsample 0.8, early stopping (20 rounds) on validation |
| LSTM | 2 layers, 64 hidden units, dropout 0.2; input \[30, 6\]; linear head with 12 outputs; Adam, learning rate 1e-3, batch 128, up to 100 epochs, early stopping with patience 10; MSE on standardized targets |
| GRU (optional) | Same as LSTM with GRU cells |

### 11.5 Training and Evaluation Protocol

1. Generate the dataset with `python -m fa_rlwa.forecast.dataset --config configs/forecast/dataset.yaml`.
2. Train each model: `python -m fa_rlwa.forecast.train --model lstm --config configs/forecast/lstm.yaml`.
3. Evaluate on both test sets with `python -m fa_rlwa.forecast.evaluate --models all`.
4. Report MAE and RMSE per indicator and horizon, plus the percentage improvement over EWMA.
5. Also report error in **event windows** (30 s around any step where the client count of an AP changes by 5 or more within 10 s), because prediction should matter most there.
6. Compare models with a paired Wilcoxon signed-rank test over per-run errors.

### 11.6 Forecaster Acceptance and Integration

- A forecaster is accepted if it beats EWMA on MAE at the 10 s horizon on the in-distribution test set for at least two of the four indicators. If none does, the result is still reported and analysed, and the least complex competitive model is used.
- Inference for all APs in one cycle must take at most 50 ms on CPU (FCT-06).
- Saved model directory: `models/forecast/<name>/` with `model.{pt|json|pkl}` and `meta.yaml` (features, horizons, scaling statistics, training seeds, metrics, git commit).
- Forecaster retraining on data from the RL agent's own behaviour is a stretch goal; by default the forecaster is frozen before RL training.

---

## 12. RL Environment and Agent Specification

### 12.1 Decision Loop

One environment step equals one control cycle (1 s). In each step the environment (1) picks one candidate STA by the rule in Section 12.4, (2) shows the agent the observation, (3) executes the chosen action through the controller, (4) advances the simulator by one step, and (5) returns the reward. Channel assignment stays with the rule-based Algorithm 2 in the first version (RLA-07 is a stretch goal).

### 12.2 Observation Vector

The vector is fixed-size and scaled to \[0, 1\]. The table gives the layout for K\_max = 5 APs with forecasts enabled (64 values). Networks with fewer than 5 APs are zero-padded, and actions for padded APs are invalid. A separate agent is trained for each K\_max (the 10-AP scenario SC6 uses K\_max = 10, giving 124 values); generalisation across K\_max is out of scope.

| Index | Block | Content |
| --- | --- | --- |
| 0 to 44 | Per-AP block, 5 APs x 9 values | For each AP: r\_hat, l\_hat, o\_hat, t\_hat, n/N\_max (current, 10-report average), then r\_hat, l\_hat, o\_hat, t\_hat forecast at +10 s |
| 45 to 49 | Candidate RSSI | Normalized RSSI of the candidate STA to each AP (0 if out of range) |
| 50 to 54 | Candidate current AP | One-hot |
| 55 to 59 | Candidate RSSI trend | Slope over the last 10 s per AP, divided by 2 dB/s, clipped to \[-1, 1\], then mapped to \[0, 1\] |
| 60 | Candidate demand | offered load divided by 10 Mbps, clipped |
| 61 | Time since last handover | min(seconds, 60) / 60 |
| 62 | Network fairness | JFI over the last 10 s |
| 63 | Network size | number of STAs divided by 100 |

With forecasts disabled (ablation) the forecast entries are removed, and the vector has 44 values: the per-AP block shrinks to 5 x 5 = 25 entries.

### 12.3 Action Space

`Discrete(K_max + 1)`: action 0 means stay; action a in 1 to K\_max means move the candidate STA to AP a - 1. Choosing the current AP is treated as stay. An invalid action (padded AP, or AP with RSSI below -82 dBm) is executed as stay and receives the invalid-action penalty. When using PPO, `MaskablePPO` from sb3-contrib may use an explicit action mask instead.

### 12.4 Candidate Selection Rule

Among STAs whose last handover was at least 10 s ago, choose the STA with the highest priority; break ties by the lowest `sta_id`.

```latex
p_j = 0.5\,o_{ap(j)} + 0.3\,(1 - \hat r_j) + 0.2\,l_{ap(j)}
```

If no STA qualifies, the step is a no-op for the agent (stay), and the simulator advances.

### 12.5 Reward

```latex
R_t = w_1\,\frac{T_t}{\Lambda_t} + w_2\,J_t - w_3\,H_t - w_4\,\min(1,\rho_t) - w_5\,I_t
```

T\_t is aggregate throughput, Lambda\_t is total offered load, J\_t is the JFI over per-STA satisfaction T\_i / lambda\_i, H\_t is 1 if a handover was executed in this step (0 otherwise), rho\_t is the mean retransmissions per frame, and I\_t is 1 if the action was invalid. Defaults: w1 = 1.0, w2 = 0.5, w3 = 0.2, w4 = 0.3, w5 = 1.0. Per-step reward lies roughly in \[-1.5, 1.5\]. Reward weights are hyperparameters for the sensitivity study (Section 14.5).

### 12.6 Episodes and Scenario Sampling

- Training episode: 600 steps. Evaluation episode: 1200 steps, with the first 60 steps excluded from reported metrics.
- At `reset()`, sample a scenario from the training mix (SC1, SC2, SC4, SC5) with equal probability, and sample the number of APs, STAs, positions, and traffic mix within the ranges in the scenario file. Draw the episode seed from the training seed stream (1000 and above).
- Validation episodes use seeds 100 to 119. Final test episodes use seeds 200 to 219, and SC3 and SC6 appear only in the final test.

### 12.7 Algorithms and Hyperparameters

| Setting | DQN (primary, Stable-Baselines3) | PPO (alternative) |
| --- | --- | --- |
| Policy network | MLP, hidden layers \[128, 128\], ReLU | MLP, hidden layers \[128, 128\] for policy and value |
| Learning rate | 1e-4 | 3e-4 |
| Discount (gamma) | 0.95 | 0.95 |
| Replay buffer / rollout | Buffer 100,000; learning starts at 5,000 steps; batch 64; train every 4 steps | 2,048 steps per update; batch 64; 10 epochs; GAE lambda 0.95 |
| Target network | Hard update every 1,000 steps | Not applicable |
| Exploration | Epsilon from 1.0 to 0.05 over the first 30% of training | Entropy coefficient 0.01 |
| Clipping | Gradient norm 10 | Clip range 0.2 |
| Total training | 1,500,000 steps (about 2,500 episodes) | 2,000,000 steps |
| Seeds | 5 agent seeds (0 to 4) per configuration | 5 agent seeds |

Budget check: with the simulator at 100x real time, 1.5 million steps take about 4 hours of simulation plus network-update time, which fits NFR-P3 (24 h).

### 12.8 Training Procedure

1. Run `gymnasium.utils.env_checker.check_env` on the environment; it must pass.
2. Measure baseline mean episode reward for the random policy, client-driven policy, and original TiWA on the validation seeds. Store them in `results/rl/baselines.json`.
3. **Stage 1 (sanity):** 3 APs, 10 STAs, SC1 only, 200,000 steps. The agent must exceed the random and client-driven rewards; if not, use the checklist in Section 12.9 before continuing.
4. **Stage 2 (main):** scenario mix of Section 12.6, 1,500,000 steps, 5 agent seeds. Evaluate greedily every 20,000 steps on 10 validation episodes and keep the best checkpoint by validation reward.
5. Repeat Stage 2 with `use_forecast: false` using the same seeds (the ablation).
6. Evaluate the best checkpoints on the final test protocol (Section 14).

### 12.9 Debugging Checklist

| Symptom | Likely cause and check |
| --- | --- |
| Reward does not improve | Reward scale too small or dominated by one term: log each reward component; check observation scaling lies in \[0, 1\] |
| Agent never moves anything | Handover cost too high, or exploration too short: lower w3, extend epsilon decay, inspect action histogram |
| Agent moves every step (ping-pong) | w3 too low or dwell constraint disabled: raise w3, check the ping-pong rate metric |
| Many invalid actions | Missing action masking or wrong padded-AP handling: log invalid-action rate, add the mask or raise w5 |
| Q-values diverge | Learning rate too high or unnormalized inputs: lower the learning rate, clip rewards, check the target update interval |
| Starves some STAs | Reward favours total throughput: raise w2, report per-STA satisfaction distribution |
| Works in training scenarios but fails on SC3 or SC6 | Overfitting to the scenario mix: widen scenario sampling, add noise, report as a generalisation limit |

---

## 13. Module Interfaces and Configuration

### 13.1 Core Data Structures and Interfaces

```python
# src/fa_rlwa/types.py
@dataclass
class APState:  ap_id: int; channel: int; pos: tuple[float, float]; tx_dbm: float
@dataclass
class STAState: sta_id: int; ap_id: int; pos: tuple[float, float]; traffic: str; offered_mbps: float

@dataclass
class IndicatorReport:      # one per step; every array has shape [n_ap]
    t: int; rssi_dbm: np.ndarray; loss: np.ndarray; tput_mbps: np.ndarray
    occ: np.ndarray; n_clients: np.ndarray; airtime_own: np.ndarray

@dataclass
class NetworkState:
    t: int; aps: list[APState]; stas: list[STAState]
    rssi_dbm: np.ndarray               # [n_sta, n_ap]
    reports: list[IndicatorReport]     # most recent last

@dataclass(frozen=True)
class Handover:      sta_id: int; target_ap: int
@dataclass(frozen=True)
class ChannelChange: ap_id: int; channel: int
Action = Handover | ChannelChange
```

```python
# src/fa_rlwa/policies/base.py
class Policy(ABC):
    name: str
    def reset(self, state: NetworkState, rng: np.random.Generator) -> None: ...
    def initial_ap(self, sta_id: int, state: NetworkState) -> int: ...   # default: strongest RSSI
    @abstractmethod
    def decide(self, state: NetworkState) -> list[Action]: ...

# src/fa_rlwa/forecast/base.py
class Forecaster(ABC):
    horizons: tuple[int, ...] = (5, 10, 30)
    def fit(self, train: Dataset, val: Dataset) -> None: ...
    def predict(self, history: np.ndarray) -> np.ndarray: ...     # in [n_ap, 30, 6]; out [n_ap, n_horizon, 4]
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "Forecaster": ...

# src/fa_rlwa/sim/network.py
class WifiNetwork:
    def __init__(self, scenario: ScenarioConfig, seed: int): ...
    def reset(self) -> NetworkState: ...
    def step(self, actions: list[Action]) -> NetworkState: ...
    def step_metrics(self) -> dict: ...      # per-step aggregates used by env and logs

# src/fa_rlwa/env/wifi_env.py
class WifiAssocEnv(gymnasium.Env):
    def __init__(self, scenario_mix: list[ScenarioConfig], reward_cfg: RewardConfig,
                 forecaster: Forecaster | None = None, k_max: int = 5, seed: int = 0): ...
    # observation_space = Box(0, 1, (obs_dim,), float32); action_space = Discrete(k_max + 1)
```

RL policies are wrapped in `policies/rl_policy.py` so that the experiment runner treats them like any other `Policy`: the wrapper builds the observation for each candidate STA, queries the trained network greedily, and returns the resulting `Handover` actions.

### 13.2 Scenario Configuration Example

```yaml
# configs/scenarios/sc4_flash_crowd.yaml
name: sc4_flash_crowd
area_m: [30, 20]
duration_s: 1200
warmup_s: 60
aps:
  - {id: 0, pos: [6, 5],  channel: 1}
  - {id: 1, pos: [15, 10], channel: 6}
  - {id: 2, pos: [24, 15], channel: 11}
background_stas:
  count: 20
  placement: uniform
  mobility: static
  traffic: {type: cbr, mbps: 1.0}
flash_crowd:
  count: 30
  start_s: 300
  window_s: 30
  hotspot: [6, 5]
  spread_m: 3.0
  dwell_mean_s: 120
  traffic: {type: cbr, mbps: 1.0}
propagation: {pl0_db: 40, exponent: 3.0, shadow_sigma_db: 4.0, shadow_rho: 0.95}
mac: {eta: 0.6, c_max: 0.9, tau: 0.01, retry_limit: 7}
handover: {mode: lvap, outage_s: 0.2, dwell_s: 10}
```

The positions of the three APs in SC1 to SC3 are taken from the paper's floor plans (Figures 3 to 5, building about 15 m x 9 m). Read them from the figures, store them in the scenario files, and record the values in the decision log.

### 13.3 Policy and RL Configuration Examples

```yaml
# configs/rl/dqn_forecast.yaml
algorithm: dqn
k_max: 5
use_forecast: true
forecaster: models/forecast/xgb_v1
reward: {w1: 1.0, w2: 0.5, w3: 0.2, w4: 0.3, w5: 1.0}
dqn:
  net_arch: [128, 128]
  learning_rate: 1.0e-4
  gamma: 0.95
  buffer_size: 100000
  learning_starts: 5000
  batch_size: 64
  train_freq: 4
  target_update_interval: 1000
  exploration_fraction: 0.3
  exploration_final_eps: 0.05
total_timesteps: 1500000
eval_every: 20000
seeds: [0, 1, 2, 3, 4]
```

```yaml
# configs/policies/tiwa_original.yaml
policy: tiwa_original
threshold_mode: spatial        # I-1: spatial or temporal
window: 10
max_moves_per_cycle: 2         # I-4
migration_margin: 0.05         # I-5
channel_reassign_interval_s: 60
```

---

## 14. Experiment Protocol, Validation, and Test Plan

### 14.1 Seeds and Splits

| Purpose | Seeds | Used for |
| --- | --- | --- |
| Forecaster training / validation | 0 to 79 / 80 to 99 | Dataset generation (Section 11.2) |
| RL training episodes | 1000 and above (stream) | Agent training |
| Agent initialisation | 0 to 4 | Five independent agents per configuration |
| Validation (tuning weights, selecting checkpoints) | 100 to 119 | Never used for reported results |
| Final test | 200 to 219 | All reported results (20 runs per cell) |

### 14.2 Run Matrix

Up to 7 systems (S1 to S7, Section 6.1) x 6 scenarios (SC1 to SC6) x 20 test seeds = 840 runs of 1200 s each. S7 (densification) runs only on SC1, SC4, and SC5, with 1 and 2 extra APs added at positions listed in the scenario file. The runner must support parallel execution (`--jobs N`).

### 14.3 Commands (Reference Workflow)

```bash
# 1. Generate forecaster data, train, and evaluate forecasters
python -m fa_rlwa.forecast.dataset  --config configs/forecast/dataset.yaml
python -m fa_rlwa.forecast.train    --model xgb  --config configs/forecast/xgb.yaml
python -m fa_rlwa.forecast.train    --model lstm --config configs/forecast/lstm.yaml
python -m fa_rlwa.forecast.evaluate --models all

# 2. Train RL agents (with and without forecasts), 5 seeds each
python -m fa_rlwa.rl.train_dqn --config configs/rl/dqn_forecast.yaml   --seed 0
python -m fa_rlwa.rl.train_dqn --config configs/rl/dqn_noforecast.yaml --seed 0

# 3. Run all systems on the test seeds
python -m fa_rlwa.experiments.runner --matrix configs/experiments/main.yaml --seeds 200-219 --jobs 8

# 4. Statistics, tables, and figures
python -m fa_rlwa.experiments.report --results results/ --out results/report
```

### 14.4 Statistics

- Report the mean, standard deviation, and 95% bootstrap confidence interval (10,000 resamples) over the 20 test seeds for every metric.
- Compare each pair of systems with a paired Wilcoxon signed-rank test over seeds; correct for multiple comparisons with Holm's method; significance level 0.05.
- Report effect size as the relative improvement in the mean (for example, "+12% aggregate throughput versus S2").
- For RL, also report the spread across the 5 agent seeds (best, median, worst) so that training variance is visible.

### 14.5 Ablations and Sensitivity Studies

| Study | Variants |
| --- | --- |
| Forecast value | RL with and without forecasts (same seeds) |
| Forecast horizon | 0 (current only), 5, 10, 30 s |
| Forecaster type | EWMA, ARIMA, XGBoost, LSTM feeding the same RL agent |
| Reward weights | w3 in {0, 0.1, 0.2, 0.5}; w2 in {0, 0.25, 0.5, 1.0} |
| Handover cost | LVAP (0.2 s) versus non-seamless (2 s) |
| Network scale | 3, 5, and 10 APs; 20, 50, and 100 STAs |
| Densification | S2 with +1 and +2 APs versus S6 on the original number of APs |

### 14.6 Validation in Mininet-WiFi (or ns-3)

The goal is to show that the policy trained in the abstract simulator still behaves sensibly in a more realistic stack. Full numerical agreement is not expected.

1. Build a topology matching SC1 (3 APs on channels 1, 6, 11, 802.11n, positions from the scenario file) with 30 stations sending UDP traffic at 1 Mbps each (iperf3 or D-ITG) to a server.
2. Implement a controller loop (Python) that, once per second, reads the measurements below, builds the same observation vector, queries the trained policy, and executes handovers through Mininet-WiFi's association-control interface (check the API of the installed version).
3. Measurements: per-AP station signal strength, TX retries, and TX failed counters from `iw dev <ap-iface> station dump`; channel busy time and active time from `iw dev <ap-iface> survey dump` (occupancy is busy time divided by active time); throughput from the iperf3 server logs.
4. Run S1, S2, S5, and S6 for 10 repetitions of 600 s in SC1 and SC4.
5. Pass criteria: the ranking of systems matches the simulator's ranking in at least one scenario, and decision latency stays below 1 s. Document differences and their likely causes (propagation model, rate adaptation, TCP/UDP behaviour).

### 14.7 Unit Tests (with Expected Values)

| Test | Input | Expected |
| --- | --- | --- |
| JFI, equal shares | \[2, 2, 2, 2\] | 1.0 |
| JFI, one STA takes all | \[4, 0, 0, 0\] | 0.25 |
| JFI, unequal | \[1, 2, 3, 4\] | 100 / (4 x 30) = 0.8333 |
| Path loss, no shadowing | d = 10 m, PL0 = 40, n = 3, P\_tx = 15 dBm | RSSI = -55 dBm |
| Path loss at 20 m | same parameters | PL = 79.03 dB, RSSI = -64.03 dBm |
| PHY rate lookup | RSSI -60, -68, -75, -83 dBm (p\_coll = 0) | 65, 39, 19.5 Mbps, not associable |
| Airtime demand | 3 Mbps at 39 Mbps PHY, eta = 0.6, p\_fail = 0 | a = 3 / (39 x 0.6) = 0.1282 |
| Collision probability | 10 active STAs, tau = 0.01 | 1 - 0.99^9 = 0.0865 |
| Normalization | r = -65 dBm | r\_hat = 0.5 |
| Moving average | reports 1 to 12 | mean of reports 3 to 12 |
| Throughput bound | any scenario | T\_i never exceeds lambda\_i; AP throughput never exceeds eta x 65 Mbps |
| Env shapes | K\_max = 5 with forecasts | observation shape (64,), action space Discrete(6) |
| Env without forecasts | K\_max = 5 | observation shape (44,) |
| Determinism | same config and seed, two runs | identical `metrics_summary.csv` |
| Dwell constraint | request two moves of one STA within 10 s | second move rejected and logged |
| Invalid action | move to an out-of-range AP | executed as stay, I\_t = 1 |

### 14.8 Integration Tests

- **Smoke test:** a 120 s run of every policy on SC1 finishes without error and writes all log files.
- **Reproduction test:** client-driven versus original TiWA on SC1 shows TiWA at least as good in aggregate throughput (Section 9.8).
- **Env test:** a random-action rollout of 600 steps never produces NaN observations or rewards.
- **Latency test:** the full decision cycle (forecast plus policy) stays under 0.2 s for 10 APs and 100 STAs.

### 14.9 Required Figures and Tables for the Final Report

1. Aggregate throughput by system and scenario (bars with 95% CI).
2. Jain's fairness index by system and scenario.
3. Handovers per run and ping-pong rate.
4. AP load over time during a flash crowd for S1, S2, and S6.
5. Forecast error by model and horizon, and an example forecast against truth.
6. RL training curves (mean and spread across 5 seeds) with and without forecasts.
7. Densification comparison: S2 with extra APs versus S6.
8. Latency distribution of decisions.
9. Table of significance tests with Holm-corrected p-values.

---

## 15. Work Breakdown and Definition of Done

### 15.1 Work Packages

| ID | Work package | Depends on | Main outputs | Done when | Effort (weeks) |
| --- | --- | --- | --- | --- | --- |
| WP1 | Setup: repository, environment, CI, quick start | none | Repo with passing empty test suite, README | A new member completes Section 8.2 in under 2 hours | 1 |
| WP2 | Simulator core (propagation, mobility, traffic, MAC) | WP1 | `sim/` package, scenario configs SC1 to SC6 | Unit tests in 14.7 pass; sanity checks in 9.8 pass | 3 |
| WP3 | Indicators, logging, metrics | WP2 | `monitor/`, `experiments/metrics.py`, log files | Schemas in 11.1 implemented; JFI tests pass | 1.5 |
| WP4 | Client-driven and random baselines, controller | WP2, WP3 | `policies/`, `controller/` | Smoke test passes for both | 1 |
| WP5 | TiWA reproduction and corrected TiWA | WP4 | `tiwa.py`, `tiwa_fixed.py`, decision log entries I-1 to I-7 | Calibration target in 9.8 met; ambiguities documented | 2 |
| WP6 | Forecast dataset and forecasters | WP3, WP4 | `forecast/`, trained models, error tables | Acceptance in 11.6 evaluated and reported | 4 |
| WP7 | Gymnasium environment | WP3, WP4 (WP6 for forecast features) | `env/`, env tests | `check_env` passes; shapes as in 12.2 | 2 |
| WP8 | RL training and tuning | WP6, WP7 | `rl/`, checkpoints, training curves | Stage 1 and Stage 2 in 12.8 completed for 5 seeds, with and without forecasts | 4 |
| WP9 | Full experiments, ablations, statistics | WP5, WP8 | `experiments/`, report tables and figures | All items in 14.9 produced | 3 |
| WP10 | Mininet-WiFi or ns-3 validation | WP5, WP8 | `validation/`, validation results | Pass criteria in 14.6 evaluated | 3 |
| WP11 | Documentation, thesis, defense | all (runs in parallel) | Final report, slides, reproducible repo | Supervisor sign-off | 4 |

Work packages WP6 and WP7 can proceed in parallel once WP4 is done, and WP5 can proceed in parallel with WP6 and WP7. WP10 can start its topology scripts as early as WP4 and should not wait for WP8.

### 15.2 Suggested Roles for a Team of Three

| Role | Primary packages | Secondary |
| --- | --- | --- |
| A: Networking and simulation lead | WP2, WP5, WP10 | WP3 |
| B: Machine learning (forecasting) lead | WP6 | WP3, WP9 |
| C: Reinforcement learning and experiments lead | WP7, WP8, WP9 | WP4 |

For a team of two, A and B merge as one role covering WP2, WP5, WP6, WP10, and the other member covers WP7 to WP9.

### 15.3 Definition of Done (applies to every package)

- Code merged to `main` through a reviewed pull request, with type hints and docstrings.
- Unit tests for new code pass; coverage of core modules is at least 70%.
- Configuration files committed; all runs reproducible from a seed.
- Results (where applicable) saved with their resolved config and git commit.
- README or `docs/` updated; decisions recorded in the log (Section 16).
- Supervisor informed at the next weekly meeting; milestone review at the end of each phase.

---

## 16. Decision Log and Open Issues

### 16.1 Design Decisions Already Taken

| ID | Decision | Rationale |
| --- | --- | --- |
| DD-01 | Corrected TiWA uses throughput headroom (1 - t / 39 Mbps) instead of raw throughput | The paper treats throughput both as a quality and a load signal; headroom makes "higher is better" consistent |
| DD-02 | Packet loss indicator is the per-attempt failure rate | Observable from real MAC counters; the paper's ARF description is not implementable as stated |
| DD-03 | Abstract Python simulator for training; Mininet-WiFi or ns-3 for validation only | RL needs fast simulation; the emulator is too slow for millions of steps |
| DD-04 | One candidate STA per step, with a discrete action space | Keeps the action space small and training tractable |
| DD-05 | Channel assignment stays rule-based in the first version | Limits scope; RL-based channel assignment is a stretch goal |
| DD-06 | Default offered load is 1 Mbps per STA | The paper's 3 Mbps and 30 Mbps total are inconsistent with 30 STAs (Section 9.8) |
| DD-07 | Forecaster frozen before RL training | Avoids a moving target; retraining on agent data is a stretch goal |
| DD-08 | DQN is the primary algorithm; PPO is an alternative | Discrete actions, simpler tuning |

### 16.2 Open Issues (to Resolve Before the Date Shown)

| ID | Open question | Owner | Resolve by |
| --- | --- | --- | --- |
| OI-1 | Contact the TiWA authors to confirm Algorithm 1 interpretation (I-1 to I-5) | Supervisor and role A | End of month 1 |
| OI-2 | Choose Mininet-WiFi or ns-3 for validation after a one-week feasibility trial | Role A | End of month 2 |
| OI-3 | Decide whether to emulate non-seamless handover as a second condition | Team | Before WP9 starts |
| OI-4 | Obtain NIC datasheet sensitivities (or measure) to replace the rate table | Role A | Before WP5 completes |
| OI-5 | Confirm AP coordinates from the paper's floor plans | Role A | Before WP2 completes |
| OI-6 | Decide whether 5 GHz or downlink-aware association is attempted (stretch) | Team and supervisor | Month 6 review |

New decisions and resolved issues are appended to these tables with a date and the name of the person who made the change.

---

## 17. Requirements Traceability Matrix

| Project objective | Requirements | Acceptance criteria |
| --- | --- | --- |
| O1. Reproduce and critically evaluate TiWA | SIM-01 to SIM-10, MON-01 to MON-04, BAS-01 to BAS-03, BAS-06 | AC-01 |
| O2. Develop a forecaster for AP indicators | FCT-01 to FCT-06, MON-03, MON-05 | AC-02 |
| O3. Design an RL association policy using current and predicted indicators | ENV-01 to ENV-09, RLA-01 to RLA-06, CTL-01 to CTL-05 | AC-03, AC-06 |
| O4. Evaluate against baselines, including densification | EXP-01 to EXP-09, BAS-04, BAS-05, SIM-11, Section 6 | AC-04, AC-05, AC-07, AC-08 |

---

## 18. Appendix: Reference Formulas

**Jain's Fairness Index** for per-STA throughputs x\_1 ... x\_n:

JFI = (sum of x\_i)^2 / ( n \* sum of x\_i^2 )

**Channel occupancy** of channel c over interval T:

Occupancy\_c = (sum over STAs on c of estimated airtime) / T

**Forecast error metrics:**

MAE = mean(|y - y\_hat|), RMSE = sqrt(mean((y - y\_hat)^2)), MAPE = mean(|y - y\_hat| / |y|) x 100%

**Reward (default):**

R\_t = w1 \* T\_norm(t) + w2 \* JFI(t) - w3 \* H(t) - w4 \* Retx(t)

where T\_norm is aggregate throughput normalized by total offered load, H is the handover cost incurred at step t, and Retx is the mean retransmission rate. Initial weights to be tuned: w1 = 1.0, w2 = 0.5, w3 = 0.2, w4 = 0.3.

---

**Approval**

| Role | Name | Signature | Date |
| --- | --- | --- | --- |
| Student(s) |  |  |  |
| Supervisor |  |  |  |
