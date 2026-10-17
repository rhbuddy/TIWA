# Wi-Fi preflight and Mininet-WiFi validation

## Linux Wi-Fi scan preflight

Use `fa-rlwa wifi-check --interface wlan0` to test whether a selected Linux interface can complete an active scan. The experiment runner can invoke the same preflight with `--wifi-adapter wlan0`. It uses `iw`, does not change interface state, and distinguishes a successful empty scan from a scan failure. It only verifies scan capability; it does not connect a client to an AP or validate roaming/throughput.

## Mininet-WiFi closed-loop backend

`fa-rlwa validate-mininet` is the separate emulator integration. It builds APs, static stations, a wired traffic sink, and optional per-station UDP iperf3 flows; reads Mininet-WiFi and `iw` telemetry; invokes a registered FA-RLWA policy on a one-second cycle; and applies validated association moves. Generated CSV schemas match the abstract runner so experiment outputs can be compared. See the repository README for invocation examples.

The backend currently supports static station groups only. It derives candidate-AP RSSI from Mininet-WiFi's configured propagation model and reads serving-link signal/counters and channel survey data with `iw`. Survey occupancy falls back to estimated own-airtime when hardware/emulation does not expose survey counters. Channel changes are disabled unless explicitly enabled. Use `--forever` for a continuous loop and stop with Ctrl-C; the loop and emulator are torn down cleanly. In this mode iperf3 flows are bounded to 24 hours.

This backend does **not** touch host Wi-Fi interfaces and does **not** integrate with a production 5G-EmPOWER/OpenWrt controller. The sandbox used for development has no `mn`, `iw`, or Linux wireless interfaces, so only fake-backend tests have run; no AC-07 emulator result is claimed. A supported Linux host with Mininet-WiFi, root access, `iw`, and iperf3 is required for an actual run. Keep emulator-specific dependencies out of base `requirements.txt`.

## Remaining validation work

Run SC1 and SC4 as specified in SRS Section 14.6, compare policy rankings with the abstract simulator, and report actual controller decision latency. SC4 flash-crowd arrivals and mobility are not yet supported by the Mininet topology builder and need implementation before the complete SRS matrix can be run. Physical-radio integration additionally requires a target controller/AP API, which has not been specified or tested.
