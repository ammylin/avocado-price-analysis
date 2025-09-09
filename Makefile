# Makefile for Avocado Price Analysis Project

# Install dependencies
install:
	pip install --upgrade pip
	pip install -r requirements.txt

# Format code with Black
format:
	black avocado_analysis.py test_avocado_analysis.py

# Lint code with Flake8
lint:
	flake8 avocado_analysis.py test_avocado_analysis.py

# Run tests with coverage
test:
	python -m pytest -vv --cov=avocado_analysis test_avocado_analysis.py

# Run the main analysis
run:
	python avocado_analysis.py

# Clean cache files
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -f .coverage

# Default target to run all tasks
all: install format lint test run
