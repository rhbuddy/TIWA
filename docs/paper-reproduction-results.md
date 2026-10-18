# Paper Experiment 1 reproduction checkpoint

**Run date:** 17 October 2026  
**Scope:** approximate, calibrated replay of the static-client throughput experiment (paper Experiment 1, Aspects 1 and 2); this is simulator evidence, not a physical-testbed reproduction.

## Verdict

The default, uniformly placed SC1 did **not** reproduce a TiWA throughput gain. An explicit paper-informed but outcome-informed placement assumption was therefore added. With `n=3.0` and `tau=0.01` held fixed, only the SRS-approved model parameter `eta` was changed from `0.6` to `0.2`. The exploratory 300-second seed-0 sweep informed both the overlap center/spreads and the efficiency choice; the final fixed configuration was then run on ten distinct paired seeds (200–209), for 1,200 seconds per run with a 60-second warm-up.

For Aspect 1, those ten-seed means fall within the SRS Section 9.8 tolerance around its approximate targets (15–16 Mbps client-driven; 25 Mbps TiWA), with TiWA above client-driven in all ten pairs. This is evidence that this **specified abstract model and exploratory topology assumption** can generate the target throughput ordering and scale on new random seeds. It is not a blind confirmatory reproduction: the scenario geometry was selected using exploratory outcomes rather than independently reconstructed from Figure 3, and so its fidelity still needs validation. The original paper's testbed result has not been independently reproduced; the Aspect 1 relative gain is larger than reported, and important secondary metrics do not match.

## Method and assumptions

- Paper Table 2/text give 30 STAs in Aspect 1 and say their positions are illustrated in Figure 3. The figure depicts client symbols, but the article does not provide coordinates or explain how those symbols map to the 30 devices. For this run we assumed six STAs near APs (two per AP) and 24 around one overlap center. This split, center `(6.5, 6.0)`, and spreads are explicit **exploratory assumptions**, not paper-reported facts or recovered measurements.
- Aspect 2 adds ten STAs near AP 1, following the paper's description that the extra clients are in one AP's coverage area. Their exact coordinates are also not published; the hotspot is an approximation.
- Traffic is 1 Mbps per STA, following the SRS. The paper's prose also says 3 Mbps per device while stating 30 Mbps total for 30 devices; those statements conflict.
- Among the SRS-approved model parameters, only `eta` changed for the throughput fit: `0.6 -> 0.2`. Propagation exponent `n=3.0` and contention parameter `tau=0.01` were held constant. The exploratory topology choice itself was outcome-informed and must not be mistaken for an independently justified paper topology. The choices are visible in `configs/scenarios/sc1_paper_aspect1.yaml` and `sc1_paper_aspect2.yaml`.
- Seed 0 was used during calibration/exploration, including the short placement/efficiency sweep. Seeds 200–209 were not used for that sweep and are paired across `client_driven` and `tiwa_original`; they are held-out random seeds for the selected fixed configuration, not a blind test of an independently reconstructed topology.
- Each mean below is across ten seeds; the parenthesized value is the sample standard deviation. Difference confidence intervals are paired percentile-bootstrap 95% CIs (10,000 resamples, seed 0). P-values are paired two-sided Wilcoxon tests with Holm correction over all reported scenario/metric comparisons.

## Results

| Scenario / metric | Client-driven mean (SD) | Original TiWA mean (SD) | Paired TiWA − client difference (95% CI) | Relative change | Holm-adjusted p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aspect 1 aggregate throughput (Mbps) | 14.146 (1.075) | 24.814 (2.249) | +10.668 [9.799, 11.653] Mbps | +75.4% | 0.0430 |
| Aspect 1 Jain fairness | 0.8050 (0.0096) | 0.9852 (0.0108) | +0.1802 [0.1752, 0.1848] | +22.4% | 0.0430 |
| Aspect 1 retransmissions / frame | 0.2413 (0.0177) | 0.1277 (0.0114) | −0.1136 [−0.1199, −0.1076] | −47.1% | 0.0430 |
| Aspect 2 aggregate throughput (Mbps) | 13.275 (1.065) | 27.302 (1.819) | +14.027 [13.135, 14.914] Mbps | +105.7% | 0.0430 |
| Aspect 2 Jain fairness | 0.6394 (0.0056) | 0.9366 (0.0199) | +0.2972 [0.2861, 0.3080] | +46.5% | 0.0430 |
| Aspect 2 retransmissions / frame | 0.3706 (0.0210) | 0.1640 (0.0096) | −0.2066 [−0.2175, −0.1957] | −55.7% | 0.0430 |

For the primary Aspect 1 target, the client-driven mean is 8.7% below the 15.5 Mbps midpoint and TiWA is 0.7% below 25 Mbps; both are inside the SRS's 25% tolerance. The paired throughput test has unadjusted `p=0.00195` and Holm-adjusted `p=0.0430`. TiWA's throughput exceeded client-driven in all ten paired seeds. This is a numerical match for the selected configuration, **not a completed Section 9.8 calibration sign-off**, because the station grouping/coordinates were not independently established from Figure 3 and were informed by exploratory outcomes.

The paper text reports approximately +52% for Aspect 1 and +58% for Aspect 2. The simulated gains (+75.4% and +105.7%) are larger, particularly for Aspect 2; this discrepancy is a fidelity warning, not a close numerical match. The paper's relative gains are not reproduced.

## Important mismatches and limits

- The simulated mean PHY rate is effectively 65 Mbps for both policies in both aspects, so the PHY-rate improvement shown in the paper is **not** reproduced. The current RSSI-to-rate abstraction saturates at its top rate for these clients.
- The simulator's per-AP mean occupancy rises with TiWA (Aspect 1: 0.468 to 0.772; Aspect 2: 0.468 to 0.875), rather than providing a directly comparable reproduction of the paper's occupancy bars. Its across-channel occupancy standard deviation falls (Aspect 1: 0.379 to 0.299; Aspect 2: 0.379 to 0.176), but the definitions and underlying radio measurements are not identical to the paper's testbed measurements.
- Simulator decision times (about 1.0–1.6 ms in this run) cannot be compared with the paper's reported 0.1889 ± 0.088 seconds: the former are local abstract-model computation timings, not a physical controller/network decision loop.
- Aspect 2's larger-than-paper gain, approximate and outcome-informed AP/client placement, no packet-level 802.11n/Atheros/OpenWRT model, and the paper's traffic-load inconsistency limit causal or testbed-level interpretation. Only the client-driven and original-TiWA policies were used in this matrix; it is not the full comparator set.
- SC2 mobility, SC3 shared-channel behavior, forecast/RL methods, and live Mininet-WiFi/ns-3/physical hardware were not tested by this run. Thus AC-01 is only partially evidenced, and no overall paper-reproduction or end-to-end SRS claim is made.

## Reproduction commands

From the repository root, install the project and development dependencies in a Python 3.10–3.12 environment, then run:

```bash
python -m fa_rlwa.experiments.runner \
  --matrix configs/experiments/tiwa_paper_experiment1.yaml \
  --seeds 200-209 --jobs 1 --no-plots

python -m fa_rlwa.experiments.report \
  --results results/tiwa_paper_experiment1 \
  --out results/tiwa_paper_experiment1/report
```

To repeat the calibration seed instead, use `--seeds 0` in the first command. The matrix file records the scenarios, policies, and default seed range. Run summaries, raw traces, aggregate CSVs, paired-test CSVs, and plots are generated under the ignored `results/` directory; no generated datasets are committed.
