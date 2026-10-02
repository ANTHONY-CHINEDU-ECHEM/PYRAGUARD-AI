.PHONY: install install_ml index demo test lint benchmark ragcheck assets serve dashboard audit train validate export docker clean

PYTHON ?= python3

install:        ## core package with API, dashboard and development tools
	$(PYTHON) -m pip install -e ".[api,dashboard,dev]"

install_ml:     ## optional heavy stack: YOLO, neural embeddings, LLM clients
	$(PYTHON) -m pip install -r requirements_ml.txt

index:          ## build the knowledge index and the reference image library
	pyraguard index

demo:           ## run a generated ignition clip through the whole pipeline
	pyraguard demo ignition cam_kitchen_01 7

test:
	$(PYTHON) -m pytest

lint:
	ruff check src tests scripts

benchmark:      ## 1500 stills and 60 clips, writes reports/benchmark.json
	pyraguard benchmark 1500 60

ragcheck:       ## retrieval, answer and scenario evaluation, writes reports/rag_eval.json
	pyraguard ragcheck

assets:         ## regenerate every README figure from real pipeline runs
	$(PYTHON) scripts/generate_readme_assets.py

serve:
	pyraguard serve

dashboard:
	pyraguard dashboard

audit:          ## audit data/dfire and write reports/dfire_audit.json
	pyraguard audit data/dfire

train:          ## fine tune YOLO on DFire (needs install_ml and a GPU)
	pyraguard train data/dfire 100 16

validate:
	pyraguard validate test

export:
	pyraguard export onnx

docker:
	docker compose up --build

clean:
	rm -rf storage runs .pytest_cache .ruff_cache reports/demo reports/detect
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
