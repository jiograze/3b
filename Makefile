.PHONY: install test clean

install:
	python -m pip install --upgrade pip
	pip install -e .

install-dev: install
	pip install -r requirements-dev.txt

test:
	pytest tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete