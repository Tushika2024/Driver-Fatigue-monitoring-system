# Viva notes

## One-sentence position
An engineering-and-evaluation contribution: a fatigue monitor that runs on a laptop/phone CPU for vehicles with no
OEM driver monitoring, evaluated with a protocol that does not flatter itself, and whose alerts adapt to who is in
the cabin.

## Say these first (limits before they are found)
* RLDD drowsiness is **acted**, not real sleep loss.
* About 12 usable people: wide uncertainty; we report per-person spread and bootstrap intervals over people.
* The compensatory-behaviour detectors are rule-based and have **no ground-truth labels**.
* The microsleep definition (1.0 s continuous closure) is an assumption and is configurable.
* We have not run on real fleet data. Hardware outputs are future scope.
* Synthetic-data results prove the code runs, nothing more.

## "Tesla and Mercedes already do per-driver calibration"
Yes, and we do not claim that. Per-driver baselines exist in OEM steering-based systems; camera-based OEM methods are
proprietary and unpublished, so there is nothing public to compare against. The contribution is the *measurement*
on a public dataset: early signals (blink re-opening slowdown, compensatory behaviour) rather than only PERCLOS,
**earliness** as the headline metric, an adaptive baseline that does not assume the driver starts alert, a leakage
demonstration, a subgroup slice, and cabin-aware intervention.

## Expected questions
**Why leave-one-person-out?** Random window splits put the same person (and overlapping windows) in train and test.
`fatigue evaluate` runs that wrong split once next to the correct one. On real data it should inflate; on our synthetic
data it does not, because synthetic people are independent draws.

**Why logistic regression?** About 12 people. A simple model is explainable (reason strings come from its
coefficients) and harder to overfit. A boosted model is one config line (`model.kind: hgb`) and is reported honestly.

**Why is the adaptive z-score off by default?** In a 10-minute clip the driver does not change state, so a history of
the same clip normalises the drowsiness away. We say so, test it (rows D to G of the main table), and note the effect
needs shift-length data.

**What if the driver is tired when calibrating?** Row G of the main table enrols on the person's drowsy clip. The
adaptive reference needs no alert start by design (high percentile of open-eye level).

**Is a loud alarm wrong?** In an ambulance with a patient or a school bus, yes. Cabin context chooses the channel;
SEVERE is redirected, never silenced.

**False alarms?** Reported per hour with and without persistence, because persistence is what a driver actually experiences.

**Fairness?** Glasses and facial hair only: objective and observable. No gender or ethnicity inference. The cells are
tiny, so it is a methodology demonstration. The correction (per-subgroup thresholds, reliability-weighted routing to the
head/posture model) is fit without the test person.

**Privacy?** No face recognition and no template. Identity is the profile the driver picks. The pre-trip history stores
four numbers per session. The depot view cannot show an individual.

**What would you do next?** Validate on real fleet video, add an LED or vibration sink, conformal confidence on the score,
separate drowsiness from distraction, and tamper-evident logging.
