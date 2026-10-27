# Driver Fatigue Monitor · backend

> This folder is the Python half of the full-stack project (see the repository root README). `src/fatigue_api/` wraps the package below with a FastAPI server: `pip install -e ".[api,dev]"` then `python -m fatigue_api`. The commands below are the standalone CLI.

Phone/laptop-deployable fatigue monitoring for vehicles that have **no OEM driver-monitoring
system**: goods trucks, ambulances, buses, cabs. CPU only, no GPU, no data collection.

It covers a whole shift in three in-vehicle stages plus an offline stage:

| Stage | What it does | Entry point |
|---|---|---|
| Offline | Videos → features → leave-one-person-out evaluation → trained model | `fatigue extract / evaluate / train` |
| Before the shift | 60 s pre-trip check against the driver's **own** history (advisory, never a lockout) | `fatigue pretrip` |
| During the drive | Live monitoring, graded alerts, cabin-aware routing | `fatigue live` |
| After the shift | Driver report + aggregate depot report | `fatigue report` |

> **Honest status.** This is an engineering-and-evaluation project, not a new neural architecture.
> The code is unit- and end-to-end-tested on **synthetic** data. It has **not** been run on real
> UTA-RLDD footage or a real webcam from inside this repository's build environment (no dataset,
> no camera, no model download there). See [What is and is not verified](#what-is-and-is-not-verified).

## Quick start (no dataset, no webcam)

```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.10+
pip install -e ".[dev,tts]"
pytest                                                    # ~110 tests, about a minute
fatigue demo --outdir demo                                # full pipeline on synthetic data
open demo/driver_report.html demo/depot_report.html       # (or double-click them)
```

`fatigue demo` trains on synthetic drivers, runs a pre-trip check (a normal day, then a tired one),
monitors a compressed 20-minute shift under the `ambulance_patient` cabin profile, and renders both
report pages. Every output is stamped **SYNTHETIC**.

`fatigue profiles` prints the escalation table for every vehicle profile.

## Real-data workflow (UTA-RLDD)

1. Download UTA-RLDD (Fold1 is enough). Videos must sit at `<anything>/<subject>/<label>.<ext>`,
   with label `0` = alert, `10` = drowsy. **This layout is an assumption; check it against your download.**
   Label `5` is skipped, which makes the task binary.
2. Extract per-frame features (resumable; one part-file per video):
   ```bash
   fatigue extract --videos /path/to/RLDD --out data/features.csv --jobs 2
   ```
   First run downloads two MediaPipe models (~15 MB) into `models/`. If your network blocks that,
   put `face_landmarker.task` and `pose_landmarker_lite.task` there by hand.
3. Fill in the objective subgroup file (glasses / facial hair, 0 or 1, from the videos; **no gender or
   ethnicity inference**):
   ```bash
   fatigue subjects-template --features data/features.csv --out data/subjects.csv
   ```
4. Run every experiment (leave-one-person-out, baseline, ablations, wrong-split demo, earliness,
   false alarms/hour, fairness slice, figures, `RESULTS.md`):
   ```bash
   fatigue evaluate --features data/features.csv --subjects data/subjects.csv --outdir results --jobs 2
   ```
5. Train the deployable model on all subjects and use it:
   ```bash
   fatigue train --features data/features.csv --out models/bundle.joblib
   fatigue camera-check                                   # does the signal exist on your camera?
   fatigue pretrip --driver D07 --source webcam
   fatigue live --model models/bundle.joblib --source webcam --profile solo_goods_truck \
                --driver D07 --route North --log runs/session_log.csv --audio --show
   fatigue report --log runs/session_log.csv --outdir report \
                  --depot-glob 'runs/*/session_log.csv'     # optional aggregate depot page
   ```
   `fatigue live` also takes `--source video:PATH` and `--source synthetic:prog`. In `--show` mode,
   press `a` to acknowledge an alert and `q` to quit.

## Design in one page

* **One causal processor.** `StreamProcessor` turns frames into 3 s windows (1.5 s stride). Offline
  training and live monitoring run the *same class*, so there is no train/serve skew. A test proves
  nothing in the future can change a past window.
* **Signals.** Late: PERCLOS, longest closure, head drop. Early: blink **re-opening slowdown**,
  slow closing, left-right eye divergence, and **compensatory behaviours** (eye rubbing, head shake,
  postural shift, forced hard blinks). Plus yawns, head and posture. Blink timing is measured at
  amplitude crossings on 30 FPS video.
* **Per-driver adaptive reference.** Open-eye level = a running high percentile of that driver's
  own history; neutral head pitch = running median. No assumption the shift starts alert. Early
  signals can also be z-scored (median/IQR, not mean/std) against the driver's history.
  Enrolment-based calibration is kept as an ablation, including "enrol while already tired".
* **Evaluation.** Leave-one-person-out through one shared split function; a deliberately wrong
  random split is run once to show the inflation. Headline metric is **earliness**: seconds of
  warning before the first microsleep. Also per-person accuracy and its spread, TPR/FPR, alert
  rate, false alarms per hour with and without persistence, FPS and latency.
* **Alerts.** Three levels from two evidence streams (early only → MILD; early + late → MODERATE;
  late and fused-high → SEVERE), thresholds calibrated from out-of-fold training scores, persistence
  and hysteresis, reason strings from the top contributing features, CRITICAL for unacknowledged
  SEVERE where someone can be told. If the eyes cannot be read (glare, glasses, occlusion) it drops
  to a **degraded** head-and-posture model with widened thresholds and says so; if no face is
  visible it announces that instead of guessing.
* **Cabin context.** Three booleans per vehicle (others present / second driver / overhearing is
  harmful) pick the channel: a loud alarm for a solo truck, a private low-volume voice plus crew
  notification for an ambulance with a patient. **SEVERE is never suppressed, only redirected**; if
  nobody can be told it falls back to cabin audio and announces that. Tested for every profile.
* **Pre-trip check.** Quality gate, 60 s capture (first 10 s dropped, extended to 90 s if fewer than 8
  blinks), four measures compared to the driver's last 30 sessions (median/IQR, time-of-day matched
  when ≥3 sessions exist). Verdicts FIT / CAUTION / NOT_FIT are **advisory**. Anything unmeasurable is
  INCONCLUSIVE: not stored, never treated as NOT_FIT, never blocks. Identity is manual profile
  selection; there is no face recognition or biometric template.
* **Shift report.** Driver page (trend, worst stretch, alerts, lead time, coverage flag below 60% full
  mode) and a depot page that is aggregate only, with cells under *k* drivers hidden. No leaderboard.

Diagrams: `docs/diagrams/*.svg` (A3 sheets). Deeper notes: `docs/ARCHITECTURE.md`, `docs/VIVA_NOTES.md`.

## Layout

```
src/fatigue/
  config.py  schema.py          typed config (unknown keys rejected); data contracts and feature groups
  geometry.py                   EAR, MAR, head pose (round-trip tested), posture, wrist-to-face
  extractor.py  extract.py      MediaPipe wrapper; resumable batch extraction
  blink.py  behaviour.py        blink timing; compensatory-behaviour detectors
  reference.py  stream.py       adaptive/fixed/enrolled references; the causal StreamProcessor
  windows.py  models.py         frames→windows; fused logistic model with degraded mode and reasons
  evaluation.py  fairness.py    LOSO, baseline, earliness, false alarms/hour; subgroup audit
  experiments.py  plots.py      the full experiment suite, RESULTS.md, figures
  alerts.py  cabin.py  sinks.py policy state machine; cabin routing; output sinks
  pretrip.py  report.py         pre-trip check; driver/depot reports
  live.py  sources.py  cli.py   live session; webcam/video/feature sources; CLI
  synthetic.py  demo.py         simulator and the end-to-end demo
  data/default.yaml  data/vehicle_profiles.yaml
tests/                          ~110 tests
docs/                           architecture, viva notes, A3 diagrams
examples/synthetic/             sample outputs (SYNTHETIC)
```

Data contracts: `features.csv` (one row per sampled frame; columns in `schema.FRAME_COLUMNS`),
`windows.csv`, `session_log.csv` (+ `.meta.json`), per-driver `history/<driver>.json`, the model bundle.

Configuration: everything is in `src/fatigue/data/default.yaml`; pass `--config my.yaml` (placed
**before** the subcommand) to override any key. Vehicle presets are in `vehicle_profiles.yaml`.

## What is and is not verified

Verified here (automated tests, synthetic data): geometry (head-pose recovery of known angles), blink
timing recovery, streaming causality, alert persistence/hysteresis/acknowledge/repeat, cabin routing
invariants for every profile, pre-trip verdict logic and INCONCLUSIVE handling, report suppression and
synthetic stamping, fairness-gap arithmetic, leave-one-person-out hygiene, the MediaPipe wrapper's
conversion logic against fake landmarkers, a live session end to end, and the CLI.

**Not** verified, and you must check before relying on any number:

* Never run on real RLDD videos or a real camera. The RLDD folder layout, file extensions and frame
  rates are assumptions. Real MediaPipe output (especially with glasses) may need threshold tuning.
* Synthetic results (including those in `examples/`) only show the code runs. They are **not** evidence
  about real drivers, and synthetic data cannot reproduce the real-world inflation from a wrong split.
* RLDD drowsiness is **acted**, not real sleep deprivation. About 12 usable people means wide
  uncertainty: report the per-person spread, not just a pooled figure.
* The compensatory-behaviour detectors are rule-based with no ground-truth labels; they are justified
  only by the ablation (or by hand-annotating a subset).
* The microsleep threshold for the earliness metric (`eyes.microsleep_s`, default 1.0 s) is an assumption.
* Within one 10-minute RLDD clip a driver is either alert or drowsy, so a per-driver history cannot
  show *change* there; that benefit shows only on shift-style data. Early signals therefore enter
  the model raw by default (`model.use_z: false`) and z-scored variants are reported as ablations.
* The fairness slice at this sample size is a methodology demonstration, not a certification.
* Depot-page findings from synthetic logs are built into the simulator to demonstrate the page.
* Crew/dispatch notification is a stub or generic webhook; headrest speakers, earpieces and seat
  vibration are future hardware, behind the `AlertSink` interface.

## Ethics and scope

No data collection. No face recognition or biometrics (identity = the profile the driver selects).
Fatigue scores are driving-safety aids, not a medical or disciplinary instrument; the depot view
cannot rank individuals. `data/`, videos and features are git-ignored: do not commit the dataset.

## Future scope

Bone-conduction headset / Bluetooth earpiece / seat or steering vibration (an ESP32 + vibration
motor is about ₹500) as new `AlertSink`s; trajectory fusion; automatic occupancy detection;
conformal prediction for alert confidence; drowsiness-vs-distraction separation; a tamper-evident
log; validation on real fleet data.
