.PHONY: help install data eda train test lint format api mlflow docker-build docker-run compose-up k8s-deploy clean

help:
	@echo "Targets: install data eda train test lint format api mlflow docker-build docker-run compose-up k8s-deploy clean"

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt -r requirements-dev.txt
	pip install -e .

data:
	python -m heart_mlops.data_download

eda:
	python -m heart_mlops.eda

train:
	python -m heart_mlops.train

test:
	pytest

lint:
	flake8 src api tests
	black --check src api tests
	isort --check-only src api tests

format:
	black src api tests
	isort src api tests

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	mlflow ui --port 5000

docker-build:
	docker build -t heart-disease-api:latest .

docker-run:
	docker run --rm -p 8000:8000 heart-disease-api:latest

compose-up:
	docker compose up --build

k8s-deploy:
	kubectl apply -f k8s/

clean:
	rm -rf __pycache__ .pytest_cache mlruns *.egg-info build dist coverage.xml
