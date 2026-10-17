# Design decisions and implementation log

This log complements the SRS Section 16. It records the concrete choices made in the initial runnable repository scaffold.

| ID | Decision | Rationale / consequence |
| --- | --- | --- |
| DD-09 | Keep the initial simulator pure Python and discrete at one-second steps. | Matches the SRS's training/evaluation model and keeps baseline tests portable. It is a model, not a packet simulator. |
| DD-10 | Write run traces as CSV, not Parquet. | MON-03 permits CSV or Parquet. CSV keeps the simulator install small and makes logs inspectable without PyArrow; filenames use `.csv` to avoid format ambiguity. |
| DD-11 | Keep core runtime dependencies small; pin the heavier forecaster/RL stack in `requirements-ml.txt`. | Simulation, Gymnasium, metrics, plots, and unit tests can be set up without pulling in PyTorch/XGBoost. Install the optional file for those workflows. |
| DD-12 | Model AP positions for SC1–SC3 as approximate digitizations from the paper floor plans. | The paper gives a 15 m × 9 m floor outline but no coordinate table. The values are marked in the scenario YAML; they must be refined if the source diagram can be measured more precisely. |
| DD-13 | Implement the paper's TiWA ambiguities as separate, configurable assumptions, rather than presenting the heuristic as exact source code. | The paper's two Algorithm 1 branches are identical and do not fully specify threshold axes, score units, migration margin, or migration count. We use SRS I-1–I-7 and document the limits in `paper-notes.md`. |
| DD-14 | Provide both the original TiWA interpretation and a normalized corrected variant. | The original is needed as a baseline; the corrected score uses the SRS's higher-is-better normalization and a 0.05 migration margin. |
| DD-15 | Default forecast-aware heuristic uses EWMA/persistence unless a saved model is explicitly supplied. | This makes a forecast-baseline experiment runnable without a trained model. It is not evidence that EWMA beats a moving average or that forecasts improve RL. |
| DD-16 | Use independent Gymnasium environments for `K_max=5` and `K_max=10`. | The SRS observation dimensions and action spaces differ. A checkpoint trained for five APs cannot be evaluated on ten APs without a matching network architecture/checkpoint. |
| DD-17 | Keep generated data, models, and results out of Git. | These artifacts can be large and are reproducible from configs and seeds; `.gitignore` excludes `data/`, `models/`, and `results/`. |
| DD-18 | Add an explicit, exploratory Experiment 1 layout and calibrate only the SRS-approved model parameter `eta` from 0.6 to 0.2 for an Aspect 1 throughput checkpoint. | The paper gives 30 clients and illustrates positions, but does not map all clients to coordinates or clarify how symbols correspond to the total. The assumed 6-local/24-overlap grouping and center/spreads were outcome-informed; seed 0 was used in exploration and seeds 200–209 for fixed-config paired runs. Aspect 1 is numerically within SRS tolerance, but this is provisional rather than a signed-off calibration; the relative gain is larger than the paper's and PHY/occupancy/testbed results remain unmatched. See `paper-reproduction-results.md`. |

## Open implementation work

1. Extend the paper-derived calibration beyond the Experiment 1 throughput checkpoint: investigate the gain/PHY-rate/occupancy mismatch, validate the topology assumptions, and reproduce SC2/SC3 while continuing to vary only SRS-approved parameters (`n`, `tau`, `eta`) one at a time. The first Aspect 1 calibration and held-out evidence are recorded in DD-18 and `paper-reproduction-results.md`.
2. Verify AP and mobile path coordinates from the paper figures or testbed documentation.
3. Run all forecaster models and report test-set errors, including event windows.
4. Train and compare five-seed DQN checkpoints with/without forecast input; check fairness and per-STA starvation, not just aggregate throughput.
5. Run the Mininet-WiFi backend on a supported Linux host, first with SC1 and then extend it for SC4 mobility/flash-crowd validation; report sim-to-emulator differences. The backend code is present, but no emulator evidence exists yet.
6. Select a production SDN controller/AP API before implementing or claiming physical-client control; the current emulator hook is not a vendor API integration.
