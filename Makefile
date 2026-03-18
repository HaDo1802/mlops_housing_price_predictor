SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

SRC := src serving pipelines conf
QUALITY_TARGETS := src data tests serving pipelines dags
TEST_DIR := tests

.PHONY: help install lint format test clean all vercel-preview vercel-prod api

help:
	@printf "Targets:\\n"
	@printf "  install  Install Python dependencies\\n"
	@printf "  lint     Run flake8\\n"
	@printf "  format   Run black + isort\\n"
	@printf "  test     Run pytest if a test dir exists\\n"
	@printf "  api      Run FastAPI locally\\n"
	@printf "  ui       Run Streamlit locally\\n"
	@printf "  vercel-preview Deploy FastAPI to Vercel preview\\n"
	@printf "  vercel-prod    Deploy FastAPI to Vercel production\\n"
	
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lint:
	$(PYTHON) -m flake8 $(QUALITY_TARGETS) --max-line-length 120 --extend-ignore=E501,E402,F401,F541,W291,W293
	

format:
	$(PYTHON) -m black $(QUALITY_TARGETS)

test:
	@if [ -n "$(TEST_DIR)" ]; then $(PYTHON) -m pytest $(TEST_DIR) -v; else echo "No test/ or tests/ directory found."; fi

api:
	$(PYTHON) -m uvicorn serving.api.main:app --host 0.0.0.0 --port 8000

ui:
	$(PYTHON) -m streamlit run serving/app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0


all: install format lint test

vercel-preview:
	vercel --yes

vercel-prod:
	vercel --prod --yes
