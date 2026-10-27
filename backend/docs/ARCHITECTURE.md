# Architecture

```
                         OFFLINE (once)                                 IN THE VEHICLE
 videos ──extract──► features.csv ──StreamProcessor──► windows ─► LOSO eval ─► model.joblib
 (RLDD)   MediaPipe   (per frame)    (causal, shared)                              │
                                                                                   ▼
 camera ─► LandmarkExtractor ─► StreamProcessor ─► FusionModel ─► raw_level ─► AlertPolicy ─► cabin.route ─► sinks
 (live)    (geometry.py)         (same class)      (full│degraded│blind)       (persist+hyst)  (profile)     (voice/chime/notify)
                                                                                   │
        pre-trip: 60 s ─► PretripSession ─► verdict (advisory)          session_log.csv ─► report.py (driver + depot)
```

## Data flow and contracts

| Stage | Input | Output | Notes |
|---|---|---|---|
| `extractor` | BGR frame, t | feature dict | all maths in `geometry.py`; pose every k-th frame, held between |
| `StreamProcessor.push` | feature dict | window dict every 1.5 s | causal; reference + blink + behaviour state inside |
| `FusionModel.score` | window rows | `p, pe, pl, ph, mode` | `mode` = full / degraded / blind from face and eye validity |
| `raw_level` | `Scores` + thresholds | 0..3 | two evidence streams, see below |
| `AlertPolicy.update` | raw level | level 0..4 | persistence to rise, hysteresis to fall, acknowledge, repeat, CRITICAL |
| `cabin.route` | level, reason, profile | `Action` list | pure function, tested for every profile |
| `SinkRouter.emit_all` | actions | audio / console / notify | never raises |

## Decisions worth defending

1. **Causal single processor.** Offline windows and live windows come from one class. `test_stream.py`
   truncates a stream and checks the earlier windows are bit-identical, so a future frame cannot leak backwards.
2. **Closed eyes are valid data.** An early build marked EAR < 0.03 as "invalid"; closed eyes then vanished
   from blink and PERCLOS statistics. Only non-finite or absurd (> 0.6) values are rejected.
3. **Head pose conventions.** The original scripts swapped pitch and yaw (rotation-matrix decomposition) and used
   a y-up face model against y-down image coordinates. `geometry.py` uses a camera-convention model and is tested
   by rendering known angles and recovering them. Positive pitch = nodding forward.
4. **Blink timing at amplitude crossings.** Closing / closed / re-opening are measured between 80% and 20% of the
   blink's own depth with linear interpolation. Blink-shape features use a trailing 30 s context because a 3 s window
   often contains no blink.
5. **Adaptive reference vs z-scores.** The adaptive reference rescales open-eye level and neutral pitch (robust to a
   driver who is drowsy from the start). Z-scores against own history can only show *change*, and a 10-minute RLDD
   clip does not change, so they are off by default for the model and reported as ablations; they matter on shifts.
6. **Median/IQR, not mean/std**, with a scale floor; ablation included.
7. **Thresholds from out-of-fold scores.** Alert thresholds are quantiles of alert-class scores from an inner
   subject-grouped CV on the training subjects only, so each fold's thresholds never see its test person. Binary
   decisions use the Youden threshold from the same scores.
8. **Fairness correction is leakage-free.** Per-subgroup thresholds are fit on training subjects' out-of-fold scores
   to match the pooled false-positive rate, then applied to the held-out person's group.
9. **Degraded and blind are first-class.** Unreadable eyes → head/posture/mouth model with narrower FPR targets
   (widened thresholds), announced. No face → no alert, a "driver not visible" notice after 10 s, and the policy holds
   its level instead of treating blindness as recovery.
10. **SEVERE is never suppressed.** `cabin.route` redirects it (private voice, crew, dispatch); when nobody else can
    be told it falls back to cabin audio with an announcement.
11. **Pre-trip never blocks.** `PretripResult.blocking` is always False; INCONCLUSIVE sessions are not stored, so a
    bad camera setup cannot corrupt a driver's baseline.
12. **Reports cannot rank drivers.** The depot table has no driver column and hides cells with fewer than *k* drivers.

## Alert levels

```
early_on = pe >= tau_early         late_on = pl >= tau_late        p = fused probability
MILD      early_on
MODERATE  (early_on and late_on) or p >= tau_mod
SEVERE    late_on and p >= tau_sev          (needs 3 consecutive windows; MILD/MODERATE need 2)
CRITICAL  SEVERE unacknowledged for critical_after_s, only if the profile can notify dispatch
```
Targets (`alerts.target_fpr`) are false-positive rates on alert-class training windows: 10% / 3% / 0.5%.

## Extending

* New output device → subclass `AlertSink`, register in `SinkRouter` under a channel.
* New vehicle → add a block to `vehicle_profiles.yaml` (three booleans plus dispatch/private-output flags).
* New feature → add it in `StreamProcessor._emit`, list it in `schema.GROUPS`, and `fatigue evaluate` ablates it.
* Different classifier → `model.kind: hgb` is built in; explanations then come from a companion logistic model.
