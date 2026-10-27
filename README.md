# Night Watch · Driver Fatigue Monitoring (full stack)

Fatigue monitoring for vehicles that have **no OEM driver-monitoring system** (goods trucks, ambulances, buses, cabs).
Two independent parts:

```
backend/    Python. The `fatigue` package (perception, models, alert policy, cabin routing, reports,
            offline evaluation, CLI) + `fatigue_api` (FastAPI: REST + WebSockets). Own README, tests, docs.
frontend/   React + Vite. "Night Watch" cockpit UI. Talks to the backend only through /api and /ws.
```

## Run it (two terminals)

```bash
# 1) backend  (Python 3.10+)
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[api,dev]"
python -m fatigue_api --port 8000        # first start trains a SYNTHETIC demo model (~1 min)

# 2) frontend (Node 18+)
cd frontend
npm install
npm run dev                              # http://localhost:5173
```

Or `make setup`, then `make backend` and `make frontend`. To use your own model:
`FATIGUE_MODEL=path/to/bundle.joblib python -m fatigue_api` (see `backend/README.md` for training on UTA-RLDD).
For a production build served from another origin, set `VITE_API_URL` and `FATIGUE_CORS` (see `frontend/.env.example`).

## Pages

| Page | What it shows |
|---|---|
| **Drive** | Live monitor: fatigue dial, the eye the system is reading, "what the cabin hears" diagram, shift timeline, alert log. Simulated driver (5 scenarios) or your browser camera. |
| **Pre-trip** | Guided 60 s check against the driver's own history; advisory verdict (never a lock-out). |
| **Shift report** | Per-shift score chart, episodes, coverage, lead time. |
| **Depot** | Aggregate heatmap (time slot × route), small groups hidden (k-anonymity), no per-driver data. |
| **Cabin** | Explorer: who hears what for every vehicle profile and level, or build your own cabin. |
| **Evidence** | Offline leave-one-person-out results, ablations, earliness, false alarms, fairness slice. |
| **About** | Verified / not verified, ethics, system status. |

## Honest status

* Everything runs and is automated-tested on **synthetic** data (backend ~130 tests incl. API and both WebSockets; a few frontend unit tests; the UI was checked in a headless browser).
* **Not tested:** real UTA-RLDD videos, a real webcam, real MediaPipe output, the browser-camera path end to end (the server side is tested with a *fake* landmark extractor), long-running production load, auth/TLS (there is none: put it behind your own gateway).
* The demo model and every number in the app are synthetic and are labelled so. They show that the system works, not how accurate it is on real drivers.
* No face recognition; only the score log is written, camera frames are analysed and discarded.

## API (summary)

`GET /api/health · /api/config · /api/profiles · /api/scenarios · /api/sessions · /api/sessions/{id}/report · /api/depot · /api/evaluation`,
`POST /api/cabin/route · /api/sessions/demo`, `GET|DELETE /api/pretrip/history/{driver}`,
`WS /ws/live` (start message, then `tick`/`update`/`acked`/`ended`; client sends `ack`, `stop`, `speed` or JPEG bytes),
`WS /ws/pretrip`. Interactive docs at `http://localhost:8000/docs`.
