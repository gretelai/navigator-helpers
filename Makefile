PYTHON?=python

.PHONY: create_venv
create_venv:
	$(PYTHON) -m venv venv

.PHONY: pip_install
pip_install:
	$(PYTHON) -m pip install -e "."

.PHONY: pip_install_dev
pip_install_dev:
	$(PYTHON) -m pip install -e ".[dev]"

.PHONY: style
style:
	$(PYTHON) -m isort .
	$(PYTHON) -m black .

.PHONY: test
test:
	$(PYTHON) -m pytest