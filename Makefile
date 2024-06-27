.DEFAULT_GOAL := help
.PHONY: test dev-deps deps style lint install help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

target_dirs := bin examples tests trainer

test_all:	## run tests and don't stop on an error.
	coverage run -m pytest trainer tests

test:	## run tests.
	coverage run -m pytest -x trainer tests

test_failed:  ## only run tests failed the last time.
	coverage run -m pytest --ff trainer tests

style:	## update code style.
	ruff format ${target_dirs}

lint:	## run linter.
	ruff check ${target_dirs}

dev-deps:  ## install development deps
	pip install -r requirements.dev.txt

install:	## install 🐸 Trainer for development.
	pip install -e .[dev,test]
	pre-commit install
