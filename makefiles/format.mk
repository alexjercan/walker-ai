### FORMAT
# ¯¯¯¯¯¯¯¯

format: format.isort format.black

format.black: ## Run black on every file
	python -m black src/ test/ --exclude .venv/

format.isort: ## Sort imports
	python -m isort src/ test/ --skip .venv/
