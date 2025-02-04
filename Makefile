.PHONY: venv test prepare clean activate install help


help:
	@echo "Available commands:"
	@echo "  venv      - Create a virtual environment"
	@echo "  test      - Run tests"
	@echo "  prepare   - Prepare the environment"
	@echo "  clean     - Clean the environment"
	@echo "  activate  - Activate the virtual environment"
	@echo "  install   - Install dependencies"
	@echo "  build     - Build the Docker image"
	@echo "  up        - Run docker-compose u"
	@echo "  help      - Show this help message"

venv:
	bash scripts/create_venv.sh

test:
	bash scripts/run_tests.sh

prepare: venv
	sleep 5
	bash scripts/activate.sh

clean:
	bash scripts/clean.sh

activate:
	source venv/bin/activate

install:
	pip install -r requirements.txt

build:
	docker-compose build
up:
	docker-compose up