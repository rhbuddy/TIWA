# Base-paper reading notes: TiWA

**Source file:** `TiWA_Achieving_Tetra_Indicator_Wi-Fi_Associations_.pdf`

**Citation:** S. Manzoor et al., “TiWA: Achieving Tetra Indicator Wi-Fi Associations in Software Defined Wi-Fi Networks,” *IEEE Access*, vol. 11, pp. 89520–89534, 2023. DOI: 10.1109/ACCESS.2023.3307476.

These notes record the parts of the paper that inform FA-RLWA; the full paper is retained in the repository. Page references below use the IEEE page numbers printed in the article.

## Paper's problem and design

The paper observes that IEEE 802.11 does not prescribe client/AP association. A conventional client therefore often chooses the strongest-RSSI AP, ignoring AP/channel load, collisions, and link loss. An SDN controller can monitor the network and use LVAP-style virtual APs for seamless reassociation without changing the 802.11 client behavior.

TiWA uses four per-AP indicators:

1. Average uplink RSSI of associated clients.
2. Packet-loss rate, estimated from per-client transmission/rate behavior.
3. Throughput handled by the AP.
4. Channel occupancy, estimated from each client's transmitted bytes, delivery probability, and MCS airtime.

The controller smooths the **most recent ten reports** to damp transient changes and ping-pong. Algorithm 1 uses an AP imbalance test, then maximizes a score based on a client's RSSI and AP/channel metrics. Algorithm 2 reassigns channels, considering AP neighbors, channel overlap, and channel load (article pp. 89524–89525).

## Testbed and reported findings

The authors used a three-AP, 802.11n testbed with Atheros AR9220 radios, EmPOWER-LEDE/OpenWRT APs, a 5G-EmPOWER controller, three non-overlapping 2.4 GHz channels (1, 6, 11), and LVAP-based handovers. Traffic was generated with D-ITG; three monitor-mode laptops captured frames for performance measurements (pp. 89526–89528).

The evaluation comprises:

- **Experiment 1:** static clients, 30 clients in Aspect 1 and 10 additional clients in Aspect 2; studies load distribution and throughput.
- **Experiment 2:** two mobile clients move through the overlapping coverage areas, with 20 static background clients; studies RSSI/mobility behavior.
- **Experiment 3:** two APs share a collision domain and the channel assignment behavior is examined.

The paper reports that TiWA's aggregate throughput is higher than the client-driven, WiMA, and Wi-Balance comparators: its text cites +52% over the client-driven method in Experiment 1 Aspect 1 and +58% in Aspect 2; +20% aggregate throughput for the mobility experiment; and +37% over client-driven in Experiment 3 (pp. 89528–89531). The paper also reports better fairness, fewer retransmissions, and a mean decision time of 0.1889 ± 0.088 s over 60 decisions (p. 89532). These values describe the original paper's testbed and remain the comparison reference; the repository's more limited approximate replay is reported separately in [`paper-reproduction-results.md`](paper-reproduction-results.md).

## Important ambiguities and known issues

- Algorithm 1 prints **identical formulas in both branches**; the paper does not fully define their different operational meaning.
- The threshold description uses max, min, and median without a consistently clear aggregation axis.
- The score combines RSSI with heterogeneous load/loss/throughput/occupancy quantities without a complete unit/orientation definition. The paper notes RSSI is negative and says to maximize the product.
- The exact selection/migration count and the “no better option” threshold are not completely specified.
- The paper describes ARF in a way the SRS considers ambiguous; FA-RLWA uses a simulator-side failure estimate and treats MAC-counter loss as the observable interpretation.
- The paper states 3 Mbps per client but also states 30 Mbps total for 30 clients. Those numbers do not agree. The SRS selects 1 Mbps per client as its default, so the scenario configs follow the SRS rather than silently “correcting” the paper.

The SRS resolves these ambiguities in Section 10.3 (I-1–I-7). The reproduction policy uses those explicit assumptions and has a separate normalized corrected-TiWA baseline; they should not be mistaken for verbatim executable author code, which the paper does not provide.

## How the repository relates to the paper

- `configs/scenarios/sc1_steady.yaml` through `sc3_shared_channel.yaml` represent simplified analogues of the three paper experiments. The separate `sc1_paper_aspect1.yaml` and `sc1_paper_aspect2.yaml` configurations encode one explicit approximation of Experiment 1; Table 2's count of 30 devices and Figure 3's symbols are not tied to a complete set of coordinates/counts in the article text.
- AP locations are approximate coordinates read from Fig. 3's 15 m × 9 m floor plan; the scripted route in SC2 is likewise an approximate digitization of Fig. 4. The new station-group placement supports seeded per-AP and overlap clusters. The chosen client grouping, centers, and spreads are made explicit, but are exploratory assumptions informed by initial outcomes, not independently surveyed or fully recovered from Fig. 3.
- The Experiment 1 replay is numerically within the SRS Aspect 1 throughput tolerance and shows higher simulated throughput/fairness and fewer retransmissions under original TiWA, but its topology selection is outcome-informed, it does not match every Figure 6 metric, and its relative-gain percentages are larger than the paper's. Treat this as provisional simulator evidence rather than a completed reproduction; see [`paper-reproduction-results.md`](paper-reproduction-results.md).
- The simulator follows the SRS's fast log-distance/airtime model, not the paper's testbed radio/MAC stack. Its outputs are evidence for this specified model, not a physical or packet-level reproduction.
- A Mininet-WiFi validation path is now implemented but has not been run in this environment; ns-3 validation is also outstanding. See [`limitations.md`](limitations.md).
