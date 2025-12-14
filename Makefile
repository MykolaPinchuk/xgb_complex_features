PYTHON ?= python
SMOKE_CONFIG := configs/smoke.yaml
EXP_CONFIG := configs/exp_default.yaml

.PHONY: install smoke exp_default run_smoke run_exp_default aggregate_smoke aggregate_exp_default report_smoke report_exp_default

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

smoke:
	$(PYTHON) -m xgb_complex_features workflow --config $(SMOKE_CONFIG) --aggregate-output runs/smoke/aggregate --report runs/smoke/report.md

exp_default:
	$(PYTHON) -m xgb_complex_features workflow --config $(EXP_CONFIG) --aggregate-output runs/exp_default/aggregate --report runs/exp_default/report.md

run_smoke:
	$(PYTHON) -m xgb_complex_features run --config $(SMOKE_CONFIG)

run_exp_default:
	$(PYTHON) -m xgb_complex_features run --config $(EXP_CONFIG)

aggregate_smoke:
	$(PYTHON) -m xgb_complex_features aggregate --input runs/smoke --output runs/smoke/aggregate

aggregate_exp_default:
	$(PYTHON) -m xgb_complex_features aggregate --input runs/exp_default --output runs/exp_default/aggregate

report_smoke:
	$(PYTHON) -m xgb_complex_features report --input runs/smoke/aggregate --output runs/smoke/report.md

report_exp_default:
	$(PYTHON) -m xgb_complex_features report --input runs/exp_default/aggregate --output runs/exp_default/report.md
