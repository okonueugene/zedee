# Ziidi Signal Engine v3.0

A self-healing, rules-based trading signal system for the Nairobi Securities Exchange (NSE). The system scans a watchlist every 15 minutes, detects structured price patterns, fuses those signals with corporate action intelligence, enforces hard risk limits, and learns from its own trade history to calibrate itself over time.

All alerts are manual — the system tells you what to buy and sell, and you place the orders.

Optional **MVP path** (`use_mvp_engine`): `core/mvp_engine.py` + `journal.mvp_tick` + `training/mvp_sweep.py` — frozen entry/stop/exit and a strict threshold sweep for calibration (see Module Reference).

### Calibration starvation (why `score_threshold` is 55)

If nothing ever passes the ML gate, the journal stays empty, the calibrator never reaches `min_trades_to_calibrate`, and the system cannot self-improve. In live runs the score often tops out in roughly the 40–58 range; a threshold at **60** sits above that ceiling most days, so you get **zero trades** and permanent calibration starvation.

**Operating default: `score_threshold` 55** (strict inequality: entry requires `ml_score > threshold`, so a day whose peak score equals the threshold still does not enter — thin sessions stay quiet). Strong days (e.g. scores in the high 50s) can pass; the **`high_5d` guard** remains unchanged and should not be relaxed — it blocks extended/stale prints near the rolling high (e.g. bad post-close entries). **Do not** push the threshold below 55 for “more trades” without re-evaluating that trade-off.

Live MVP defaults in `system_params.json` are aligned with weekly-style holds: **`trailing_stop_pct` 0.07** (legacy trailing path), **`max_hold_days` 10** (MVP `TIME_STOP` session count via `mvp_tick`).

---

## Architecture

```
┌─────────────────────────┐
│     Market Data Layer   │  core/data.py
│  (NSE scraper + SQLite) │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│     Signal Engine       │  core/signals.py
│  (phase + scoring)      │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│     Event Engine        │  core/events.py
│  (corporate actions)    │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│     Score Fusion        │  core/signals.py → fuse_score()
│  (signal + event)       │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│     Risk Governor       │  core/risk.py
│  (drawdown + exposure)  │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│     Execution Layer     │  core/execution.py
│  (manual trade alerts)  │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│   Notification Layer    │  core/notifications.py
│  (Telegram + log)       │
└────────┬────────┬────────┘
         ↓        ↓
┌──────────────┐ ┌──────────────────┐
│ Trade Journal│ │  Log Forwarder   │  core/log_forwarder.py
│learning/     │ │  (ERROR/CRITICAL │
│journal.py    │ │   → Telegram)    │
└──────┬───────┘ └──────────────────┘
       ↓
┌─────────────────────────┐
│   Analytics Engine      │  learning/analytics.py
│  (what is working?)     │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│   Calibration Engine    │  learning/calibrator.py
│  (parameter adjustment) │
└─────────────────────────┘
```

---

## Project structure (complete)

```
claude_bet/
├─ main.py                      # Runtime orchestrator / scan loop / structured logging
├─ README.md                    # This document
├─ requirements.txt             # Python deps
├─ system_params.json           # Live-tuned params (written by calibrator)
├─ ziidi_alert_log.jsonl        # Structured event log (source of truth for analytics)
├─ ziidi_breakouts.json         # Persisted breakout state across restarts
├─ ziidi_equity.csv             # Equity snapshots (one row per scan)
├─ ziidi_positions.db           # SQLite: open positions + canonical trades table
├─ ziidi_prices.db              # SQLite: historical prices per symbol
├─ backtest/
│  └─ validate_expansion.py      # Offline validation for watchlist expansion filters
├─ logs/
│  └─ system.log                # File logger (grep/tail friendly)
├─ config/
│  └─ params.py                 # Constants + load/save system_params helpers
├─ core/
│  ├─ activation.py             # Activation filter (volume spike / near-high participation)
│  ├─ data.py                   # Scraper + SQLite I/O (positions/prices/equity)
│  ├─ events.py                 # Corporate action detection + score boost
│  ├─ execution.py              # Manual buy/sell alert formatting + JSONL writes
│  ├─ features.py               # Price-history enrichment (avg volume/value, high_5d)
│  ├─ log_forwarder.py          # Tails system.log → forwards ERROR/CRITICAL to Telegram
│  ├─ ml_scorer.py              # Optional ML scorer (drop-in for detect_phase + score_signal)
│  ├─ mvp_engine.py             # Frozen MVP modules 1–5 (entry/stop/exit)
│  ├─ notifications.py          # Telegram routing + severity + message formatting
│  ├─ risk.py                   # Drawdown / risk cap / trailing stop checks
│  ├─ signals.py                # Phase detection, scoring, fusion, entry validation
│  └─ universe.py               # Liquidity filter (ADVT proxy) for watchlist expansion
├─ learning/
│  ├─ analytics.py              # Metrics + breakdown reports
│  ├─ calibrator.py             # Parameter adjustment + pause/resume recommendations
│  └─ journal.py                # Trades lifecycle + MAE/MFE tracking + MVP tick
├─ training/
│  ├─ backtest.py               # ML-driven backtest on held-out test split (2023+)
│  ├─ features.py               # Shared FEATURE_COLS + split boundaries + constants
│  ├─ label_generator.py        # Build labelled parquet dataset + features
│  ├─ ml_scorer.py              # Training-time ML scoring helpers (offline)
│  ├─ mvp_sweep.py              # Frozen MVP threshold sweep + plateau validation
│  ├─ param_optimizer.py        # Grid search + optional system_params update
│  └─ train_model.py            # Train model + eval + model pickle output
└─ nse_master/                  # Historical raw data input directory (for training scripts)
```

---

## Function index (by module)

This is a “where to look” map of **every top-level function** in the codebase, with a one-line purpose. Many functions already have detailed docstrings; this index is intentionally brief and points you at the right file/function fast.

<details>
<summary><b>main.py</b></summary>

- **`log_event(event_type, data)`**: emit one event to console + `logs/system.log` + JSONL + Telegram routing.
- **`log_rejection(sym, gate, **extra)`**: structured `SIGNAL_REJECTED` helper for the 8-gate chain (and MVP path).
- **`_daily_rollover(now)`**: reset daily counters when the EAT calendar day changes.
- **`send_daily_summary()`**: `DAILY_SUMMARY` snapshot of scans, gating, and open-position state.
- **`_load_breakout_state()` / `_save_breakout_state()`**: persist `breakout_detected` across restarts.
- **`_expire_stale_breakouts()`**: delete stale breakouts older than `breakout_expiry_sessions`.
- **`_market_open(now)`**: market-open heuristic for health-ping visibility (not an entry gate).
- **`_mvp_high_5d(history)`**: 5-bar rolling high proxy for the MVP `high_5d` guard.
- **`manage_positions(df, positions_df, now)`**: update MAE/MFE + stops; emit sells + close trades when exits trigger.
- **`scan_mvp_entries(df, now, current_equity)`**: MVP entry scan (frozen Module 1 only) and `SIGNAL_REJECTED` logging.
- **`scan_for_entries(df, now, current_equity)`**: legacy 8-gate chain; emits top ranked manual BUY alerts.
- **`run_scan()`**: orchestrate the 5 scan phases; enforce global entry gates; never skip position management.
- **`run_calibration()`**: run the calibrator, apply parameter updates, manage pause/resume state.
- **`send_health_ping()`**: compute rolling expectancy/win-rate, emit `HEALTH_PING`.
</details>

<details>
<summary><b>config/params.py</b></summary>

- **`load_params()`**: read `system_params.json` (or defaults) into the live `params` dict.
- **`save_params(p)`**: write updated params back to `system_params.json`.
- **Watchlist expansion constants**:
  - **`WATCHLIST_EXTENDED`**: extended symbol universe (filtered down by liquidity + activation before scoring)
  - **`MIN_ADVT_KES`**: liquidity floor (20-session ADVT proxy) in KES
  - **`ACTIVATION_VOLUME_SPIKE`**: activation threshold for volume ratio vs 20-session average
  - **`ACTIVATION_MIN_PARTICIPATION`**: minimum participation floor vs average volume
  - **`ACTIVATION_NEAR_HIGH_5D`**: “near structure” threshold vs 5-session high
  - **`ENRICHMENT_MIN_BARS`**: minimum bars required before enrichment is attempted
</details>

<details>
<summary><b>core/data.py</b></summary>

- **`_log(event_type, data)`**: lightweight JSONL writer (used by the data layer for fetch diagnostics).
- **`init_dbs()`**: create SQLite tables (positions/prices) and run safe schema migrations.
- **`load_equity(initial_capital)`**: read last equity snapshot and compute peak (fallback to initial capital).
- **`save_equity_snapshot(value)`**: append one equity row to `ziidi_equity.csv`.
- **`load_positions()`**: read the full `positions` table into a DataFrame.
- **`save_position(...)`**: insert/replace an open position record (including event context).
- **`delete_position(sym)`**: delete one open position row.
- **`update_highest_price(sym, new_highest)`**: update trailing-stop anchor in `positions`.
- **`save_price_data(df)`**: append the latest scrape to the `prices` table with a timestamp.
- **`get_history(sym, n=30)`**: fetch last `n` price rows for a symbol (DESC order).
- **`fetch_nse_data()`**: scrape `afx.kwayisi.org/nse` into `[symbol, price, chg_pct, volume]` (strict + fallback parsers).
</details>

<details>
<summary><b>core/features.py</b></summary>

- **`get_enriched_metrics(symbol, conn)`**: compute `avg_volume_20d`, `avg_value_20d` (ADVT proxy), and `high_5d` from the `prices` table.
- **`enrich_universe(symbols, conn)`**: enrich all symbols; returns `symbol -> metrics` for those with sufficient history.
</details>

<details>
<summary><b>core/universe.py</b></summary>

- **`passes_liquidity(metrics)`**: `True` iff `avg_value_20d >= MIN_ADVT_KES`.
- **`build_universe(enriched_data)`**: filter enriched symbols by liquidity into a deterministic list.
</details>

<details>
<summary><b>core/activation.py</b></summary>

- **`is_activated(live_row, metrics)`**: activation decision (volume spike OR near-high + participation).
- **`activation_reason(live_row, metrics)`**: rejection reason string for logging (`LOW_VOLUME`, `BELOW_STRUCTURE`, etc).
- **`run_activation_check(symbol, live_row, metrics, log_fn)`**: runs activation and logs `ACTIVATION_CHECK` when not activated.
</details>

<details>
<summary><b>core/signals.py</b></summary>

- **`detect_phase(recent, price, chg, params)`**: classify current market regime from recent price/volume.
- **`score_signal(phase, chg)`**: map phase+direction to a raw long-entry score.
- **`fuse_score(phase, chg, event_info)`**: apply corporate action boost to the phase score.
- **`entry_time_ok(now, params)`**: time-window gate; inclusive open, exclusive close (EAT).
- **`vol_ok(current_vol, avg_vol, now, params)`**: time-adaptive volume threshold check.
- **`validate_entry(...)`**: compute sizing + expected net return and reject if below fee-aware target.
</details>

<details>
<summary><b>core/events.py</b></summary>

- **`_classify(snippet)`**: classify announcement text into `DIVIDEND/BONUS/SPLIT/RIGHTS/OTHER`.
- **`fetch_active_events(watchlist)`**: best-effort scrape of current-year corporate actions for watchlist symbols.
- **`apply_event_boost(base_score, phase, event_info)`**: phase-multiplied, hard-capped (+10) score boost.
</details>

<details>
<summary><b>core/risk.py</b></summary>

- **`equity_drawdown_ok(equity, peak_equity)`**: account-level kill switch (10% drawdown from peak).
- **`portfolio_risk_ok(n_open_positions, params)`**: cap total open risk (`n * max_risk_pct < max_total_risk_pct`).
- **`trailing_stop_hit(price, highest_price, params)`**: trailing stop trigger check (legacy mode).
</details>

<details>
<summary><b>core/execution.py</b></summary>

- **`_log(event_type, data)`**: append one alert event to JSONL.
- **`emit_buy_alert(...)`**: format + write `ZIIDI_BUY_ALERT` (manual limit order).
- **`emit_sell_alert(sym, price, reason)`**: format + write `ZIIDI_SELL_ALERT`.
</details>

<details>
<summary><b>core/notifications.py</b></summary>

- **`get_severity(event_type)`**: map event type → severity label used for routing.
- **`test_telegram()`**: one-shot credential + send test.
- **`_get_credentials()`**: env-var override → `config/params.py` fallback.
- **`_notify_info_enabled()`**: whether INFO-level events should be pushed to Telegram.
- **`_send_telegram(message)`**: send one HTML-formatted message to Telegram.
- **`_log_send_failure(reason)`**: best-effort failure visibility for Telegram delivery errors.
- **`_fmt_trade_alert(...)` / `_fmt_kill_switch(...)` / `_fmt_health_ping(...)` / `_fmt_generic(...)` / `_fmt_system_start(...)`**: message formatters.
- **`_format(event_type, severity, data)`**: choose the right formatter for the event.
- **`dispatch_notification(event_type, data)`**: route an event to Telegram based on severity and `NOTIFY_INFO`.
</details>

<details>
<summary><b>core/log_forwarder.py</b></summary>

- **`start_log_forwarder(log_path)`**: start a daemon tailer that forwards `ERROR/CRITICAL` lines to Telegram.
</details>

<details>
<summary><b>core/ml_scorer.py</b></summary>

- **`score_from_history(history)`**: build close/volume-based features and return `(ml_score_0_100, phase_hint)` or `None`.
- **`reload_model()`**: hot-reload `training/ziidi_signal_model.pkl` without restarting.
- **`_build_features(history)`**: feature computation from DESC-ordered history (reverses to chronological internally).
- **`_infer_phase(feats)`**: coarse phase label used only for event multipliers in ML mode.
- **`_rsi` / `_ema_last` / `_bb_pct` / `_atr_norm` / `_compression_score` / `_expansion_trigger` / `_vol_features` / `_safe_float`**: numeric feature helpers (live-safe; return NaN on bad inputs).
</details>

<details>
<summary><b>core/mvp_engine.py</b></summary>

- **`trading_days_since_entry(entry_date, as_of)`**: weekday-session counter used by MVP time exits.
- **`mvp_entry_rejection(row, threshold)`**: frozen entry gate; returns structured rejection payload or `None`.
- **`should_enter(row, threshold)`**: `True` iff `mvp_entry_rejection(...) is None`.
- **`new_position(entry_price)`**: initialise MVP position state (excursions + stop).
- **`update_excursions(pos, high, low)`**: update MFE/MAE/high/low vs entry.
- **`compute_stop(pos, current_price)`**: tiered monotonic stop tightening.
- **`should_exit(pos, current_price, low)`**: `STOP` or `TIME_STOP` decision.
- **`exit_fill_price(pos, low, close)`**: stop-fill vs close-fill rule.
</details>

<details>
<summary><b>learning/journal.py</b></summary>

- **`_now()`**: EAT ISO timestamp helper.
- **`_log(event_type, data)`**: journal-specific JSONL logging helper.
- **`_holding_hours(entry_time_str, exit_time_str)`**: compute holding duration in hours.
- **`migrate_positions_schema()` / `migrate_trades_schema()`**: idempotent schema migrations for SQLite tables.
- **`update_excursions(sym, current_price)`**: legacy MAE/MFE tracking for open positions (must run every scan).
- **`mvp_tick(sym, high, low, close, now, ...)`**: MVP lifecycle controller (excursions + stop + exit evaluation).
- **`open_trade(...)`**: create canonical `trades` row on entry.
- **`update_trade(symbol, current_price)`**: real-time updates for an open trade (used by excursions).
- **`close_trade(symbol, exit_price, reason)`**: close the trade + compute realised P&L.
- **`log_trade(...)`**: append the legacy CSV journal row (compat) and compute P&L.
- **`get_last_exit_reason(sym)`**: last trade reason lookup for cooldown logic.
- **`get_cooldown_hours(sym, params)`**: reason-based cooldown duration (params-driven).
- **`load_trades()` / `load_trades_db()`**: load closed trades from CSV / from canonical SQLite table.
- **`excursion_diagnostics(df)`**: sanity checks for MAE/MFE vs exit behaviour.
</details>

<details>
<summary><b>learning/analytics.py</b></summary>

- **`core_metrics(df)`**: win rate, expectancy, profit factor, max drawdown.
- **`by_phase(df)`**: performance grouped by phase.
- **`by_hold_time(df)`**: performance grouped by holding duration.
- **`by_score(df)`**: performance grouped by score buckets.
- **`full_report(df)`**: combined report dict.
- **`_normalise_pnl_column(df)`**: internal helper to normalise return column naming.
</details>

<details>
<summary><b>learning/calibrator.py</b></summary>

- **`run(df, params, equity, peak_equity)`**: compute `new_params`, `should_pause`, `reason` from recent trade outcomes.
</details>

<details>
<summary><b>training/label_generator.py</b></summary>

- **`load_master(data_dir)`**: load raw NSE master data and normalise columns.
- **`compute_features(df_sym)`**: compute per-symbol feature columns for training.
- **`add_labels(df_sym)`**: add forward-return label (`label_5d`) for supervised learning.
- **`build_labelled_dataset(data_dir, out_path)`**: end-to-end dataset build → parquet.
- **`main()`**: CLI entry point.
- **Internal helpers**: `_canonicalise_candles`, `_normalise_columns`, `_rsi`, `_bb_pct`, `_atr_norm`, `_compression`, `_expansion_trigger`, `_vol_features`.
</details>

<details>
<summary><b>training/train_model.py</b></summary>

- **`chronological_split(df)`**: split labelled dataset into train/val/test by fixed date bands.
- **`simulate_expectancy(...)`**: compute fee-aware expectancy from predicted probabilities at a threshold.
- **`train(labelled_path, threshold=0.50)`**: train the classifier, evaluate, write `ziidi_signal_model.pkl` + report.
- **`main()`**: CLI entry point.
- **Internal helpers**: `_check_xgboost`, `_check_feature_coverage`.
</details>

<details>
<summary><b>training/backtest.py</b></summary>

- **`run_backtest(...)`**: sequential ML backtest on the held-out test period (no lookahead).
- **`main()`**: CLI entry point.
- **Internal helpers**: `_load_model`, `_mae_mfe`.
</details>

<details>
<summary><b>training/mvp_sweep.py</b></summary>

- **`run_sweep(labelled_path)`**: run frozen MVP trades across the fixed threshold grid and write metrics/trades.
- **`validate_mvp_sweep(results)`**: plateau validation + consistency checks.
- **`main()`**: CLI entry point.
- **Internal helpers**: `_load_model`, `_prepare_df`, `_run_symbol`, `_equity_max_drawdown`, `_metrics`.
</details>

<details>
<summary><b>training/param_optimizer.py</b></summary>

- **`run_grid_search(...)`**: grid search over entry/vol/stop params on validation data.
- **`update_system_params(best)`**: write best params into `system_params.json` (optional automation).
- **`main()`**: CLI entry point.
- **Internal helpers**: `_simulate`, `_expectancy`.
</details>

## Module Reference

### `main.py` — Orchestrator

The entry point and thin runtime shell. Owns no business logic — its only job is to wire modules together and manage shared state.

**Runtime state it holds:**

| Variable | Purpose |
|---|---|
| `params` | Live calibrated parameters (persisted to `system_params.json`) |
| `equity` / `paper_equity` | Account equity; paper tracks closed P&L in paper mode |
| `peak_equity` | Highest equity ever reached; used by the drawdown kill switch |
| `breakout_detected` | `{sym: {"price": float, "detected_at": ISO}}` — persisted to `ziidi_breakouts.json` |
| `last_trade_time` | `{symbol: datetime}` for cooldown enforcement |
| `consecutive_fails` | Fetch failure counter; triggers safety pause at 3 |
| `pause_entries` | Global entry gate; set/cleared by risk and calibration logic |
| `fetch_pause_until` | If paused due to fetch failures, auto-resume time (backoff) so the system retries within the same session |

**Scheduled jobs:**

| Job | Frequency |
|---|---|
| `run_scan()` | Every 15 minutes (runs 24/7; time gate inside blocks entries outside 10:30–14:45) |
| `send_daily_summary()` | Daily at 15:00 EAT (structured day recap to Telegram) |
| `run_calibration()` | Daily at 15:10 EAT |
| `send_health_ping()` | Every 60 minutes |

**Three-output logging system:**

Every call to `log_event()` writes to three places simultaneously:

| Output | Format | Purpose |
|---|---|---|
| Console | `[HH:MM:SS EAT] SEVERITY  EVENT_TYPE \| {...}` | Human-readable during live sessions |
| `logs/system.log` | `YYYY-MM-DD HH:MM:SS \| LEVELNAME \| message` | Persists after terminal closes; grep/tail friendly |
| `ziidi_alert_log.jsonl` | `{"timestamp":..., "event":..., "severity":..., ...}` | Structured; source of truth for analytics |

**Four-phase `run_scan()` architecture:**

The scan is split into deterministic phases with hard boundaries. The critical property: **position management is never gated**. Skipping it during a kill switch would mean trailing stops don't fire and MAE/MFE data goes dark.

| Phase | Gated? | What it does |
|---|---|---|
| 1 — Market Data | Never | Fetch NSE prices; abort on failure |
| 2 — Position Mgmt | Never | `manage_positions()` — **legacy**: excursions + trailing stops; **MVP**: `mvp_tick()` only (frozen excursion + monotonic stop + time exit) |
| 3 — Entry Gates | — | Time → drawdown → system pause → risk cap; return if any fires |
| 4 — Signal Gen | Yes | `scan_for_entries()` — **legacy**: 8-gate chain; **MVP**: `scan_mvp_entries()` (ML score **>** threshold + one `high_5d` guard) |
| 5 — State Save | — | Persist breakout state + equity snapshot |

**`manage_positions(df, positions_df, now)`**

Runs unconditionally on every scan.

- **Legacy (default):** For each open position, updates MAE/MFE/lowest_price via `update_excursions()`, keeps the trailing stop anchor current, and fires a sell + closes if target, explicit stop, or `trailing_stop_hit()` applies. Exceptions per symbol are logged as `POSITION_ERROR`.
- **MVP (`use_mvp_engine: true` in `system_params.json`):** For each open position, calls `learning/journal.py` → `mvp_tick(symbol, high, low, close, now)` (live uses one scraped price for high/low/close). Implements `core/mvp_engine.py`: excursion updates, `compute_stop()` (stop never loosens), exits with reasons `STOP` or `TIME_STOP` only.

**Explicit exits (target/stop) — legacy mode only**

When `use_mvp_engine` is **false**, positions carry explicit `target_price` and `stop_price` fields. On every scan tick, `manage_positions()` will close a position if:

- `price >= target_price` → exit reason `TARGET`
- `price <= stop_price` → exit reason `STOP_LOSS`
- trailing-stop hit → exit reason `TRAIL_STOP`

**`scan_for_entries` / `scan_mvp_entries`**

When **`use_mvp_engine`** is **true** in `system_params.json`, **`scan_mvp_entries()`** runs instead of the 8-gate chain. ML must be loaded (`training/ziidi_signal_model.pkl`). Entry logic is **frozen** in `core/mvp_engine.py`: `ml_score > score_threshold` and a single guard (`close` must not be above `high_5d × 0.98`). Rejections log as **`SCORE_BELOW_THRESHOLD`** (`ml_score`, `threshold`) or **`HIGH_5D_GUARD_BLOCKED`** (`close`, `high_5d`, `ratio`) — not a single opaque `SHOULD_ENTER_FALSE`. Operational checks only (history, live data, cooldown, no duplicate open position). No volume, breakout, net-target, or bootstrap forcing.

**Watchlist expansion (liquidity + activation pre-filters, additive)**

To scale the MVP scan from a small watchlist to ~10–12 names without wasting cycles on illiquid or “dead” prints, the engine adds an optional **pre-filter pipeline** in front of MVP scoring (without touching `core/mvp_engine.py`):

```
FETCH (live scrape)
  -> ENRICH (price-history metrics from ziidi_prices.db)
  -> UNIVERSE (ADVT >= MIN_ADVT_KES)
  -> ACTIVATION (volume spike OR near 5d high + participation)
  -> MVP scoring loop (unchanged: ml_score > threshold AND high_5d guard)
```

- **Enrichment**: `core/features.py` computes `avg_volume_20d`, `avg_value_20d` (ADVT proxy), and `high_5d`.
- **Liquidity**: `core/universe.py` filters to symbols with `avg_value_20d >= MIN_ADVT_KES`.
- **Activation**: `core/activation.py` keeps only symbols active enough for today’s scan.
- **Offline sanity check**: run `python backtest/validate_expansion.py` against your current `ziidi_prices.db`.

**New log events produced by the pipeline** (in `ziidi_alert_log.jsonl`):

- **`ENRICHMENT_SKIP`**: symbol missing sufficient history (`ENRICHMENT_MIN_BARS`).
- **`ACTIVATION_CHECK`**: emitted for symbols that fail activation (includes `volume_ratio`, `distance_to_high_5d`, `reason`).
- **`SCAN_SUMMARY`**: one per scan: sizes of extended list → enriched → universe → activated, plus `symbols_to_score`.

**Legacy `scan_for_entries` — 8-gate decision chain**

| Gate | Name | Rejection logged as |
|---|---|---|
| 1 | History depth | `SIGNAL_REJECTED gate=HISTORY rows=N required=15` |
| 2 | Live price data present | `SIGNAL_REJECTED gate=DATA` |
| 3 | Phase (CAPITULATION/NEUTRAL/INSUFFICIENT_DATA blocked) | `SIGNAL_REJECTED gate=PHASE phase=X` |
| 4 | Volume (adaptive morning/afternoon threshold) | `SIGNAL_REJECTED gate=VOLUME reason=BELOW_THRESHOLD vol_ratio=X` |
| 5 | Breakout detected + pullback met | `SIGNAL_REJECTED gate=BREAKOUT reason=PULLBACK_NOT_MET` |
| 6 | Cooldown (position open or within reason-based period) | `SIGNAL_REJECTED gate=COOLDOWN reason=WITHIN_COOLDOWN hours_remaining=X` |
| 7 | Score ≥ threshold (from `score_threshold`, default 55) | `SIGNAL_REJECTED gate=SCORE score=X threshold=…` |
| 8 | Expected net return ≥ min target after fees | `SIGNAL_REJECTED gate=NET_TARGET` |
| ✓ | All gates passed | `SIGNAL_DECISION` logged; alert queued |

Top 2 signals by fused score are emitted as `ZIIDI_BUY_ALERT` and recorded as open positions.

**`log_rejection(sym, gate, **extra)`**

One-liner helper that emits a structured `SIGNAL_REJECTED` event. After one week of paper trading, this query shows exactly where the system is getting stuck:

```python
import pandas as pd
df = pd.read_json("ziidi_alert_log.jsonl", lines=True)
df[df["event"] == "SIGNAL_REJECTED"].groupby("gate").size().sort_values(ascending=False)
```

**Breakout state persistence:**

`breakout_detected` is now saved to `ziidi_breakouts.json` after every scan and loaded on startup. This means a breakout detected on Monday still triggers a pullback check on Tuesday — the system no longer loses tracking state on restart. Breakouts older than `breakout_expiry_sessions` days are expired automatically via `_expire_stale_breakouts()` and logged as `BREAKOUT_EXPIRED`.

**Key functions:**

- `log_event(event_type, data)` — console + `logs/system.log` + JSONL + Telegram dispatch.
- `run_calibration()` — loads the trade journal, runs the calibrator, applies parameter updates, manages pause/resume state.
- `send_health_ping()` — computes rolling win rate and expectancy from the last 20 trades, determines system state (`HEALTHY` / `BELOW_FEE_FLOOR` / `DEGRADED` / `PAUSED`), logs `HEALTH_PING`.

---

### `core/mvp_engine.py` — MVP frozen engine (modules 1–5)

Minimum-viable entry/stop/exit logic for **stability probing** and calibration. **Do not** add filters or “improve” conditions here without treating it as a spec change.

| Piece | Role |
|---|---|
| `mvp_entry_rejection(row, threshold)` | `None` if allowed; else `reason` **`SCORE_BELOW_THRESHOLD`** or **`HIGH_5D_GUARD_BLOCKED`** (+ fields for `SIGNAL_REJECTED`) |
| `should_enter(row, threshold)` | `mvp_entry_rejection(...) is None` — score gate is **strict** `ml_score > threshold`; reject if `close > high_5d × 0.98` |
| `new_position(entry_price)` | Initial `highest_price` / `lowest_price` / `mfe` / `mae` / `days` / `stop` |
| `update_excursions(pos, high, low)` | Bar updates for highs/lows and MFE/MAE vs entry |
| `compute_stop(pos, current_price)` | Tiered stop; **never** loosens (`max` with prior stop) |
| `should_exit(pos, current_price, low)` | `STOP` if `low` breaches stop; `TIME_STOP` if `days >= max_hold_days` (default 10; live: `params["max_hold_days"]` on `pos` from `mvp_tick`) and profit on **close** is under 3% (`days` on live = weekday sessions since entry via `trading_days_since_entry`) |
| `exit_fill_price` | Long fill at stop when `low` breaches it, else **close** |

Backtest driver and validation: **`training/mvp_sweep.py`** — threshold grid **only** `{55, 60, 65, 70, 75}`.

**Outputs**

| File | Contents |
|---|---|
| `training/mvp_backtest/mvp_sweep_metrics.csv` | One row per threshold |
| `training/mvp_backtest/mvp_validation.json` | Aggregated results + validation lines + MFE/gross check count |
| `training/mvp_backtest/trades.csv` | Every simulated trade: `symbol`, `threshold`, `entry_price`, `exit_price`, `gross_ret`, `net_ret`, `mfe`, `mae`, `reason` |

**Metrics (read these consistently)**

- **`avg_win`** — mean **net** return on **winning** trades only (after `ROUND_TRIP_FEE`).
- **`avg_mfe`** — mean **MFE** across **all** trades (winners and losers). Comparing `avg_win` to `avg_mfe` is **misleading** (losers drag `avg_mfe` down while winners drive `avg_win` up).
- **`avg_mfe_winners`** / **`avg_gross_win`** — use these to sanity-check capture: for each winning trade, **gross_ret ≤ mfe** (path max favourable move). **`mfe_capture_ratio`** = mean(**gross** / **mfe**) on winners (≤ 1 when data and logic align).
- **`max_drawdown`** — drawdown on a **compounded** equity curve starting at 1.0: `equity *= (1 + net_ret)` per trade; DD = max `(peak − equity) / peak` in **[0, 1]** (not additive cumsum of returns, which can exceed 100% incorrectly).

**OHLC hygiene:** each bar uses `high_eff = max(high, close)` and `low_eff = min(low, close)` so bad rows do not break MFE vs exit.

**Module 7 validation (plateau, not “all thresholds pass”)** — **ACCEPT** if there is at least one **contiguous** band of thresholds where **each** has `trades >= 30` and `expectancy >= ROUND_TRIP_FEE` (~3.874%). You want a **stable plateau** (e.g. 55–65 with similar expectancy), not a rule that rejects the whole run because 70 or 75 is weak. Isolated expectancy **spikes** vs neighbours are **WARN** only.

**Live operating default:** keep **`score_threshold`: `55`** in `system_params.json` unless your sweep shows a different plateau centre; do not re-tune entry logic for small threshold moves inside a flat band. Avoid thresholds **below 55** unless you deliberately accept more thin-day noise; keep the **`high_5d`** guard intact.

```bash
python training/mvp_sweep.py
python training/mvp_sweep.py --labelled training/nse_labelled.parquet
```

**Quick trade-level check** (after a sweep):

```bash
python -c "import pandas as pd; t=pd.read_csv('training/mvp_backtest/trades.csv'); print(t[['entry_price','exit_price','mfe','gross_ret','net_ret']].head(20).to_string(index=False))"
```

Requires a working Parquet engine (**pyarrow** or **fastparquet**) and the trained model pickle.

---

### `config/params.py` — Configuration Store

Single source of truth for all constants. Nothing else in the codebase hard-codes values.

**Telegram credentials** (fill these in to enable notifications):

```python
TELEGRAM_TOKEN   = ""    # from @BotFather
TELEGRAM_CHAT_ID = ""    # your personal or group chat ID
NOTIFY_INFO      = False # set True to also send INFO-level events
```

Credentials are read at call time. Env vars `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` take priority over file values if both are set.

**File path constants:**

| Constant | File | Notes |
|---|---|---|
| `DB_FILE` | `ziidi_positions.db` | Open positions |
| `PRICE_DB` | `ziidi_prices.db` | Price history |
| `LOG_FILE` | `ziidi_alert_log.jsonl` | Structured event log |
| `TRADES_FILE` | `ziidi_trades.csv` | Closed trade journal |
| `EQUITY_FILE` | `ziidi_equity.csv` | Equity curve |
| `PARAMS_FILE` | `system_params.json` | Live calibrated parameters |
| `BREAKOUTS_FILE` | `ziidi_breakouts.json` | Persisted breakout state |

**Key constants:**

| Constant | Value | Meaning |
|---|---|---|
| `INITIAL_CAPITAL` | `10_000.0` | Starting equity in KES |
| `PAPER_TRADE` | `True` | If True, P&L is tracked on paper only |
| `WATCHLIST` | `['SCOM', 'EQTY']` | Symbols scanned each cycle |
| `SCAN_INTERVAL_MIN` | `15` | Minutes between scans |
| `SLIPPAGE_BUFFER` | `1.005` | 0.5% limit-order slippage assumption |
| `ROUND_TRIP_FEE_PCT` | `0.03874` | NSE round-trip cost (~3.87%); kill-switch floor and fee deduction in `validate_entry` |

**Calibration defaults** (written to `system_params.json` on first run, then live-adjusted):

| Key | Default | Meaning |
|---|---|---|
| `min_net_target_pct` | `0.045` | Minimum net return to accept an entry |
| `compression_range_pct` | `0.025` | Price range width that defines compression phase |
| `max_risk_pct` | `0.02` | Max equity risked per open position |
| `max_total_risk_pct` | `0.04` | Max total portfolio risk across all open positions |
| `trailing_stop_pct` | `0.07` | Trailing stop distance from highest recorded price (legacy live path) |
| `max_hold_days` | `10` | MVP: minimum weekday sessions before `TIME_STOP` can apply (profit on close still under 3%) |
| `stop_floor_pct` | `0.04` | Minimum stop distance as % of entry price |
| `cooldown_hours` | `2` | Fallback flat cooldown (overridden by `cooldown_by_reason`) |
| `min_trades_to_calibrate` | `20` | Journal must have this many trades before calibration runs |
| `scan_start_h / scan_start_m` | `10 / 0` | **Fetch loop** start time (EAT). Prevents burning fetch failures at the open before the NSE page populates |
| `fetch_recovery_minutes` | `60` | If paused due to consecutive fetch failures, auto-resume after this backoff and retry within the same session |
| `score_threshold` | `55` | ML gate (legacy: ≥; **MVP: strict `>`**) — tuned to avoid starving the calibrator when daily score ceilings sit below ~60; do not set below 55 without deliberate review |
| `use_mvp_engine` | `false` | If **true**, live entry/exit use **only** `core/mvp_engine.py` + `mvp_tick()` (see MVP section) |
| `min_history_rows` | `15` | Minimum price bars before phase detection (Gate 1); MVP ML path still needs **≥22** bars from `score_from_history()` |
| `vol_mult_morning` | `1.2` | Volume must be ≥ avg × this before noon (calibratable) |
| `vol_mult_afternoon` | `1.1` | Volume must be ≥ avg × this after noon (calibratable) |
| `breakout_expiry_sessions` | `3` | Days before a stale breakout is expired and cleared |
| `entry_open_h / entry_open_m` | `10 / 30` | Entry window open time (EAT) |
| `entry_close_h / entry_close_m` | `14 / 45` | Entry window close time (EAT) |
| `cooldown_by_reason` | see below | Per-exit-type cooldown hours |

---

## ML Model (optional) — how it connects to the Signal Engine

The ML model **replaces `detect_phase()` + `score_signal()` only** — not the rest of the system.

- With **`use_mvp_engine: false`** (default), the **8-gate architecture stays intact**.
- With **`use_mvp_engine: true`**, entry uses **ML score + frozen `should_enter()` only** (no phase/volume/breakout gates, no event boost on entry).
- The model produces one output: \(P(\text{5-day net return} > \text{fee floor})\).
- That probability maps directly to the existing 0–100 score gate:

```python
ml_prob  = model.predict_proba(features)[0][1]  # P(profitable)
ml_score = ml_prob * 100                       # 0–100
```

**Live integration (`core/ml_scorer.py`):**

- `main.py` Gate 3: tries ML first (`score_from_history(history)`), otherwise falls back to rule-based `detect_phase()`.
- `main.py` Gate 7: uses `apply_event_boost(ml_score, phase_hint, event)` in ML mode, or `fuse_score()` in rule mode.
- `SYSTEM_START`, `SIGNAL_DECISION`, and `SIGNAL_REJECTED` include `"scorer": "ML"` or `"scorer": "RULE"` so you can verify which path is active from `ziidi_alert_log.jsonl`.

**Graceful fallback:** if `training/ziidi_signal_model.pkl` is missing or the feature build is unreliable (cold start / too few bars), the system continues running using the rule-based path unchanged — **except in MVP mode**, where `scan_mvp_entries()` logs `MVP_SCAN_SKIP` / `MVP` rejections and does not fall back to rule-based entry.

### Training data location

Place historical data in the project-root folder `nse_master/`. The training scripts default to reading from there.

### Training pipeline (scripts)

| Script | Purpose | Output |
|---|---|---|
| `training/label_generator.py` | Load master data, enforce canonical candle contract, compute features (if missing), generate forward-return labels | `training/nse_labelled.parquet`, `training/nse_candles.parquet`, `training/label_stats.json` |
| `training/train_model.py` | Chronological split (≤2020 / 2021–2022 / ≥2023), train XGBoost, evaluate | `training/ziidi_signal_model.pkl`, `training/model_report.json` |
| `training/backtest.py` | Simulate ML-driven entry chain on the **2023+ test set** (sequential, no lookahead leak) | `training/backtest/results_summary.json`, `trades.csv`, `equity_curves.csv` |
| `training/mvp_sweep.py` | Frozen MVP modules 1–5; threshold sweep **55–75** only; plateau validation | `mvp_backtest/mvp_sweep_metrics.csv`, `mvp_validation.json`, `mvp_backtest/trades.csv` |
| `training/param_optimizer.py` | Grid search score/vol/trailing-stop on the **validation set only** | `training/optimal_params.json`, `training/grid_search_log.csv` |

To run end-to-end (Git Bash / WSL / Linux/macOS):

```bash
pip install -r requirements.txt
chmod +x training/05_run_pipeline.sh
./training/05_run_pipeline.sh
```

**`cooldown_by_reason` defaults:**

| Exit reason | Hours | Rationale |
|---|---|---|
| `TRAILING_STOP` | 6 | Stop-out needs time for trend to re-establish |
| `TARGET_HIT` | 2 | Clean exit; can re-enter sooner |
| `MAX_HOLD_EXPIRED` | 4 | Timed-out trend may be exhausted |
| `MANUAL` | 2 | |
| `NONE` | 0 | No prior trade on this symbol; no cooldown |
| `STOP` | 6 | MVP monotonic stop exit |
| `TIME_STOP` | 4 | MVP time-decay exit (10+ sessions, low profit) |
| `STOP_LOSS` | 6 | Legacy explicit stop hit |
| `<unknown>` | 3 | Conservative fallback for any unrecognised reason |

---

### `core/data.py` — Market Data Layer

All database I/O. Nothing else talks to SQLite or the equity CSV directly.

**Databases managed:**

| File | Tables | Purpose |
|---|---|---|
| `ziidi_positions.db` | `positions` | Open positions with entry price, shares, MAE/MFE, event context |
| `ziidi_prices.db` | `prices` | Timestamped price history per symbol |
| `ziidi_equity.csv` | — | Equity curve snapshots (one row per scan) |

**`positions` table schema:**

| Column | Type | Meaning |
|---|---|---|
| `symbol` | TEXT PK | Ticker |
| `trade_id` | TEXT | Links an open position to the canonical `trades` table |
| `entry_price` | REAL | Price at which the limit order was placed |
| `shares` | INTEGER | Target quantity |
| `filled_shares` | INTEGER | Actual filled quantity (equals shares in paper mode) |
| `entry_time` | TEXT | ISO timestamp of entry |
| `highest_price` | REAL | Highest price seen since entry (trailing stop anchor) |
| `phase` | TEXT | Market phase at entry |
| `mae` | REAL | Maximum adverse excursion (fraction relative to entry) |
| `mfe` | REAL | Maximum favourable excursion (fraction relative to entry) |
| `target_price` | REAL | Explicit profit target for deterministic exits |
| `stop_price` | REAL | Explicit stop loss for deterministic exits |
| `score` | REAL | Fused signal score at entry |
| `lowest_price` | REAL | Absolute lowest price seen since entry |
| `event_flag` | INTEGER | 1 if a corporate action boosted this trade's score |
| `event_type` | TEXT | e.g. `"DIVIDEND"`, empty if no event |

**Key functions:**

- `init_dbs()` — creates tables on first run and calls `migrate_positions_schema()` for safe upgrades.
### `learning/journal.py` — Trade Journal + lifecycle controller

The journal is now backed by a canonical SQLite `trades` table (the system’s “brain”).

**Trade lifecycle controller:**

- `open_trade(...)` — called on entry to create a trade record
- `update_trade(symbol, current_price)` — real-time MAE/MFE updates (via `update_excursions`)
- `close_trade(symbol, exit_price, reason)` — closes the trade, writes exit fields, and appends the legacy CSV journal for compatibility

**Canonical SQLite table:**

`ziidi_positions.db` now includes:

- `positions` — open positions
- `trades` — closed trades with full lifecycle fields (entry/exit, target/stop, mae/mfe, pnl_pct, duration)

Analytics and calibration now read from the SQLite `trades` table.

- `fetch_nse_data()` — scrapes [afx.kwayisi.org/nse](https://afx.kwayisi.org/nse/). Identifies the correct listings table by requiring all three headers: `ticker`, `price`, `volume`. Uses `io.StringIO` for pandas 2.x compatibility. Returns a DataFrame with columns `[symbol, price, chg_pct, volume]` or `None` on failure.

**Fetch failure diagnosis (actionable):**

When the scraper fails, `FETCH_ERROR`/`FETCH_NO_TABLE` include lightweight diagnostics so you can tell *why* the system is pausing:

- `status`: HTTP status code (when available)
- `resp_bytes`: response size in bytes (helps distinguish empty/blocked pages)
- `table_found`: whether a ticker/price/volume table was detected
- `reason`: simple string reason (e.g., `"no table with ticker/price/volume columns"`)
- `save_position()` — inserts/replaces an open position including event context fields.
- `load_equity()` — reads the last row of the equity CSV; falls back to `INITIAL_CAPITAL` if missing.

---

### `core/signals.py` — Signal Engine + Score Fusion

Stateless functions. No I/O, no side effects.

**`entry_time_ok(now, params) → tuple[bool, str]`**

Returns `(allowed, reason)`. Entry window is read from params so it can be changed via `system_params.json` without touching code. Defaults: weekdays 10:30–14:45 EAT.

Boundary cases (verified):

| Time | Result | Reason |
|---|---|---|
| 10:29 | `False` | `EARLY_SESSION` |
| 10:30 | `True` | — |
| 14:44 | `True` | — |
| 14:45 | `False` | `LATE_SESSION` |
| Saturday | `False` | `WEEKEND` |

**`vol_ok(current_vol, avg_vol, now, params) → bool`**

Returns `False` immediately when `avg_vol <= 0` — prevents `ZeroDivisionError` on cold start before the price history fills. The threshold is time-adaptive: `vol_mult_morning` (1.2) before noon, `vol_mult_afternoon` (1.1) after noon. Both are stored in params so the calibrator can tune them.

**`detect_phase(recent, price, chg, params) → str`**

Data arrives `ORDER BY timestamp DESC` so `iloc[0]` is always the most recent bar. Volume series is reversed to chronological order before `diff()` so `vol_trend < 0` correctly means falling volume.

| Phase | Condition | Entry action |
|---|---|---|
| `EXPANSION` | Price > prior range max × 1.001, volume ≥ 1.3× avg | Track for pullback |
| `COMPRESSION` | Range ≤ 2.5%, falling volume | Watch for breakout |
| `EXHAUSTION` | `chg < 0` and falling volume | Low conviction |
| `CAPITULATION` | `chg < −2%`, volume ≥ 1.2× avg | Avoid (Gate 3 blocks) |
| `NEUTRAL` | No clear pattern | Skip (Gate 3 blocks) |
| `INSUFFICIENT_DATA` | < 15 bars or no volume column | Skip (Gate 3 blocks) |

**`score_signal(phase, chg) → float`**

| Phase + direction | Score | Notes |
|---|---|---|
| EXPANSION + positive | 90 | Strong breakout |
| COMPRESSION + positive | 75 | Building pressure |
| EXHAUSTION + positive | 50 | Weak recovery |
| COMPRESSION + negative | 40 | |
| EXPANSION + negative | 35 | Breakout candle pulling back = weak signal |
| EXHAUSTION + negative | 20 | |
| CAPITULATION | −80 | Avoid |

**`fuse_score(phase, chg, event_info) → (final_score, event_flag, event_type, confidence_boost)`**

Score Fusion layer. Calls `score_signal()` then `apply_event_boost()`. Returns the single score the rest of the system acts on.

**`validate_entry(sym, price, breakout_price, ...) → dict | None`**

Gate 8 check. Returns `None` if expected net after `ROUND_TRIP_FEE_PCT` (3.874%) doesn't clear `min_net_target_pct`. Pullback threshold is 0.5% from the breakout high.

---

### `core/events.py` — Event Engine

Detects time-sensitive corporate actions for watchlist symbols. Only events from the current calendar year are considered.

**Phase-conditional multipliers:**

| Phase | Multiplier |
|---|---|
| `COMPRESSION` | 1.0 |
| `EXHAUSTION` | 0.6 |
| `EXPANSION` | 0.5 |
| `NEUTRAL` / `CAPITULATION` | 0.0 |

**Raw boosts** (before multiplier and +10 pt hard cap): DIVIDEND +8, BONUS +6, SPLIT +5, RIGHTS +4.

**`fetch_active_events(watchlist) → dict`** — empty dict on any network failure.

**`apply_event_boost(base_score, phase, event_info) → (final_score, event_flag, event_type, confidence_boost)`**

---

### `core/risk.py` — Risk Governor

Returns booleans only. Never logs, never pauses, never adjusts parameters.

| Function | Trigger | Returns |
|---|---|---|
| `equity_drawdown_ok(equity, peak)` | Equity < 90% of peak | `False` → caller fires `KILL_SWITCH` |
| `portfolio_risk_ok(n_positions, params)` | Total open risk ≥ `max_total_risk_pct` | `False` → no new entries |
| `trailing_stop_hit(price, highest, params)` | Price < highest × (1 − `trailing_stop_pct`) | `True` → caller emits sell alert |

---

### `core/execution.py` — Execution Layer

Converts validated signals into formatted manual trade alerts. No risk checks, no state, no calibration logic.

**`emit_buy_alert(...)` → dict** — writes `ZIIDI_BUY_ALERT` JSONL entry including event context when corporate action boosted the signal.

**`emit_sell_alert(sym, price, reason)` → dict** — writes `ZIIDI_SELL_ALERT` JSONL entry.

---

### `core/notifications.py` — Notification Layer

Severity-based routing of all system events to Telegram and the JSONL log.

**Severity levels:**

| Level | Events | Behaviour |
|---|---|---|
| `CRITICAL` | `KILL_SWITCH`, `DRAWDOWN_KILL`, `EXPECTANCY_KILL`, `SYSTEM_PAUSED` | Always sent to Telegram |
| `WARNING` | `FETCH_ERROR`, `FETCH_NO_TABLE`, `SCAN_ABORT`, `ZIIDI_BUY_ALERT`, `ZIIDI_SELL_ALERT`, `SYSTEM_START`, `POSITION_ERROR`, `SYMBOL_ERROR`, `CALIBRATION_ERROR` | Always sent to Telegram |
| `WARNING` | `DAILY_SUMMARY` | Always sent to Telegram (end-of-session recap) |
| `INFO` | `HEALTH_PING`, `SCAN_GATED`, `SIGNAL_REJECTED`, `SIGNAL_DECISION`, `BREAKOUT_EXPIRED`, `POSITION_CLOSED`, `EVENT_DETECTED`, `CALIBRATION`, `TRADE_CLOSED`, `FETCH_OK`, ... | Sent only if `NOTIFY_INFO = True` |

> `SYSTEM_START` is `WARNING` — always sends to Telegram when the system comes online.
>
> `SCAN_ABORT` is `WARNING` — sent when fetch fails and no data is available for the cycle.

**Credential lookup order:** env vars `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` → `config/params.py` constants.

**Delivery confirmation:** `_send_telegram()` returns `True`/`False`. Failed CRITICAL/WARNING deliveries print to console and write to `logs/system.log` with the failure reason.

**Message templates:** trade alerts, kill-switch events, health pings, and system-start each have dedicated formatters.

**`test_telegram() → bool`** — one-shot credential verification:
```
python -c "from core.notifications import test_telegram; test_telegram()"
```

---

### `core/log_forwarder.py` — Log Forwarder

Daemon thread that tails `logs/system.log` and forwards `ERROR`/`CRITICAL` lines to Telegram. Safety net for anything that bypasses `log_event()`.

| Source | Via `dispatch_notification`? | Via forwarder? |
|---|---|---|
| `log_event("KILL_SWITCH", ...)` | Yes | Yes *(intentional — never miss a kill switch)* |
| `logging.error("...")` in any module | No | **Yes** |
| Uncaught exception logged by Python | No | **Yes** |

Seeks to EOF on startup (no history replay). Detects file rotation automatically. Started by `main.py` at import time via `start_log_forwarder()`.

---

### `learning/journal.py` — Trade Journal

Persistent record of every closed trade, including excursion data and cooldown helpers.

**`update_excursions(sym, current_price)`** — called by **legacy** `manage_positions()` on every scan tick. Updates `mae`, `mfe`, `lowest_price`. Every day this is skipped is a day of lost calibration data.

**`mvp_tick(sym, high, low, close, now)`** — used when **`use_mvp_engine`** is true. Loads the open position, applies `core/mvp_engine` excursion + `compute_stop`, persists `mae` / `mfe` / `highest_price` / `lowest_price` / `stop_price`, and returns an exit reason + fill price when `should_exit` fires.

**`log_trade(sym, entry_price, exit_price, shares, entry_time, reason) → float`** — closes a trade, reads MAE/MFE/phase/score/event from the positions table, writes a complete record to `ziidi_trades.csv`, returns realised P&L in KES.

**`get_last_exit_reason(sym) → str`** — reads `ziidi_trades.csv`, returns the `reason` field of the most recent closed trade for sym. Returns `"NONE"` if no prior trade exists. Used by Gate 6 (cooldown).

**`get_cooldown_hours(sym, params) → float`** — calls `get_last_exit_reason()` and looks up the duration in `params["cooldown_by_reason"]`. Default fallback: 3 hours for any unrecognised reason string.

**`migrate_positions_schema()`** — idempotent; run on every startup.

**Full trade record schema:**
```
timestamp, symbol, entry_price, exit_price, shares,
gross_return_pct, net_return_pct,
mae, mfe, mfe_vs_exit_gap,
phase, score,
event_flag, event_type,
entry_time, exit_time, holding_hours,
result, reason
```

---

### `learning/analytics.py` — Analytics Engine

Pure functions — DataFrame in, dict/DataFrame out, no side effects.

| Function | Answers |
|---|---|
| `core_metrics(df)` | Win rate, expectancy, profit factor, max drawdown |
| `by_phase(df)` | Which market conditions produce edge? |
| `by_hold_time(df)` | Are you cutting winners too early or holding losers too long? |
| `by_score(df)` | Does the signal score actually predict outcome? |
| `full_report(df)` | All four analyses combined |

---

### `learning/calibrator.py` — Calibration Engine

Returns `(new_params, should_pause, reason)`. Never writes to disk or mutates state.

**Decision hierarchy (first match wins):**

| Priority | Condition | Action |
|---|---|---|
| 1 | `expectancy < ROUND_TRIP_FEE_PCT` | Full pause — no edge after fees |
| 2 | Equity < 90% of peak | Full pause — drawdown breached |
| 3 | Win rate < 40% | Reduce `max_risk_pct` 15%, enter drawdown mode |
| 4 | Win rate > 60% + expectancy > 3% | Scale `max_risk_pct` up 10% (cap 2.5%) |
| 5 | Realised win data available | Set `min_net_target_pct` to 75% of avg winning trade |
| 6 | `mfe_vs_exit_gap > 3%` | Log: widen trailing stop |
| 7 | `avg_win_mae < 50% of stop_floor` | Log: tighten stop floor |

Requires `min_trades_to_calibrate` (default: 20) trades before running.

---

## Data Files

| File | Format | Purpose |
|---|---|---|
| `ziidi_positions.db` | SQLite | Open positions with real-time excursion tracking |
| `ziidi_positions.db` (`trades` table) | SQLite | Canonical trade brain — closed trades with MAE/MFE, targets/stops, durations |
| `ziidi_prices.db` | SQLite | Price history (30 bars per symbol) |
| `ziidi_trades.csv` | CSV | Closed trade journal — source for analytics and calibration |
| `ziidi_equity.csv` | CSV | Equity curve — one row per scan cycle |
| `ziidi_alert_log.jsonl` | JSONL | Every system event with timestamp, severity, and data |
| `system_params.json` | JSON | Live calibrated parameters (auto-updated by calibrator) |
| `ziidi_breakouts.json` | JSON | Persisted breakout state — survives restarts |
| `training/nse_candles.parquet` | Parquet | Canonical candle contract (OHLCV + ATR + rolling 20 high/low) for learning/backtests |
| `training/mvp_backtest/mvp_sweep_metrics.csv` | CSV | MVP threshold sweep metrics (Module 6) |
| `training/mvp_backtest/mvp_validation.json` | JSON | Per-threshold results + Module 7 validation lines |
| `training/mvp_backtest/trades.csv` | CSV | All MVP sweep trades (multi-threshold); use for gross vs MFE audits |
| `logs/system.log` | Text | File logger output — grep/tail friendly, persists across sessions |

---

## Telegram Setup

**Step 1 — Create a bot:**
1. Open Telegram → search `@BotFather` → `/newbot`
2. Copy the API token

**Step 2 — Get your chat ID:**
1. Send any message to your bot
2. Visit `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. Copy the `"id"` inside `"chat"`

**Step 3 — Add credentials to `config/params.py`:**
```python
TELEGRAM_TOKEN   = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
TELEGRAM_CHAT_ID = "123456789"
```

**Step 4 — Test:**
```
python -c "from core.notifications import test_telegram; test_telegram()"
```

**Optional — receive INFO events** (health pings, signal decisions, scan gate logs):
```python
NOTIFY_INFO = True
```

---

## Running the System

```bash
pip install -r requirements.txt
python main.py
```

On startup the system will:
1. Create `logs/` and initialise `logs/system.log`
2. Start the `LogForwarder` daemon thread
3. Initialise both SQLite databases and migrate schemas
4. Load the last equity value from `ziidi_equity.csv`
5. Load breakout state from `ziidi_breakouts.json`
6. Send `SYSTEM_START` to Telegram (WARNING level — always delivered)
7. Schedule scan / calibration / health-ping jobs
8. Run the first scan immediately, then follow the schedule

Set **`"use_mvp_engine": true`** in `system_params.json` to run the **frozen MVP** entry and exit path (`core/mvp_engine.py` + `mvp_tick`). Leave **`false`** for the full 8-gate + trailing/target behaviour.

To stop: `Ctrl+C`. Breakout state is saved after every scan — the system resumes tracking on next start.

**Monitoring during a session:**
```bash
tail -f logs/system.log                                    # live feed of all events
grep "CRITICAL\|WARNING" logs/system.log                   # only actionable events
grep "SIGNAL_REJECTED" ziidi_alert_log.jsonl | head -20    # why symbols were blocked
grep "SIGNAL_DECISION" ziidi_alert_log.jsonl               # what passed all gates
```

**Gate analysis after paper trading:**
```python
import pandas as pd
df = pd.read_json("ziidi_alert_log.jsonl", lines=True)
df[df["event"] == "SIGNAL_REJECTED"].groupby("gate").size().sort_values(ascending=False)
# → tells you which gate is the binding constraint
```

---

## Key Concepts

**Expectancy** is the number that determines whether the system should be running at all:

```
E = (win_rate × avg_win) − (loss_rate × avg_loss)
```

If `E < 3.874%` (NSE round-trip fee), the system has negative expectancy after costs. The calibrator fires a full pause. The same constant is used in both `validate_entry` and the calibrator — they are always in sync.

**MAE/MFE** calibrate your exits, not your entries. `mfe_vs_exit_gap` averaged across 20 trades tells you how much money your exit timing leaves on the table. `avg_win_mae` tells you how much winning trades move against you before recovering. These two numbers are more valuable than any entry signal — but only if `update_excursions()` is called on every tick (legacy) or **`mvp_tick()`** runs every scan (MVP mode).

**MVP mode** treats the score **threshold as a stability probe**, not a single “best” setting: run **`training/mvp_sweep.py`** over the fixed grid and look for a **contiguous plateau** (similar expectancy and enough trades across adjacent thresholds). Validation **does not** require every threshold to clear a bar—only that **some stable band** clears `trades >= 30` and `expectancy >=` fee floor. Default live **`score_threshold` = 55** matches the sweep grid floor and reduces calibration starvation versus a 60 ceiling on typical score ranges; entry uses **strict** `ml_score > threshold`.

**Score Fusion** is the explicit bridge where phase score and event score are combined into a single number before any risk or execution logic sees it:
```python
df.groupby("event_type")["net_return_pct"].mean()   # did events actually help?
```

**The 8-gate decision chain** means every rejection is named. After one week you can query which gate is blocking the most candidates and tune exactly that parameter. If `HISTORY` blocks 80% of rejections, your price data backfill is inadequate. If `SCORE` blocks 80%, the threshold may be too high for current market conditions.

**Reason-based cooldowns** replace the old flat `cooldown_hours`. A trailing-stop exit (stop-out) waits 6 hours before re-entry on the same symbol. A clean target-hit waits only 2 hours. This prevents the system from immediately re-entering a stock that just stopped you out, while allowing faster re-entry after a profitable close.

**Bootstrap mode** refers to the current parameter configuration (`min_net_target_pct = 0.045`, watchlist = `['SCOM', 'EQTY']`, relaxed thresholds). The goal is 15–20 paper trades to fill the journal so the calibrator activates. Once it does, it will adjust `min_net_target_pct` toward 75% of your actual average winning trade automatically.
