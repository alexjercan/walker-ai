### TEST
# ¯¯¯¯¯¯¯¯

.PHONY: test
test: ## Launch tests
	python -m pytest test/

.PHONY: test.coverage
test.coverage: ## Generate test coverage
	python -m pytest --cov-report term --cov-report html:coverage --cov-config setup.cfg --cov=src/ test/

test.lint: ## Lint python files with flake8
	python -m flake8 ./src ./test

test.safety: ## Check for dependencies security breach with safety
	python -m safety check
