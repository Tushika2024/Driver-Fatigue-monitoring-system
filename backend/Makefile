.PHONY: install test extract evaluate train demo

install:
	pip install -e ".[api,dev,tts]"

test:
	pytest

extract:
	python -m fatigue extract --videos $(VIDEOS) --out data/features.csv --jobs 4

evaluate:
	python -m fatigue evaluate --features data/features.csv --subjects data/subjects.csv --outdir results --jobs 4

train:
	python -m fatigue train --features data/features.csv --out models/bundle.joblib --jobs 4

demo:
	fatigue demo --outdir demo
