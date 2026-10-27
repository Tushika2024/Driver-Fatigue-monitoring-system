.PHONY: setup backend frontend test build extract evaluate train

setup:            ## install both halves
	cd backend && pip install -e ".[api,dev]"
	cd frontend && npm install

backend:          ## API on :8000 (auto-detects models/bundle.joblib)
	cd backend && python -m fatigue_api --port 8000

frontend:         ## UI on :5173 (proxies /api and /ws to :8000)
	cd frontend && npm run dev

test:
	cd backend && pytest
	cd frontend && npm test

build:
	cd frontend && npm run build

extract:          ## extract features from dataset: make extract VIDEOS=path/to/videos
	cd backend && python -m fatigue extract --videos $(VIDEOS) --out data/features.csv --jobs 4

evaluate:         ## run LOSO evaluation and save figures/tables to results/
	cd backend && python -m fatigue evaluate --features data/features.csv --subjects data/subjects.csv --outdir results --jobs 4

train:            ## train production model: saves to models/bundle.joblib
	cd backend && python -m fatigue train --features data/features.csv --out models/bundle.joblib --jobs 4
