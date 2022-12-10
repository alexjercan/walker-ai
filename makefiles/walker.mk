### Walker
# ¯¯¯¯¯¯¯¯¯¯¯

walker.install: ## Install dependencies
	pip install -r requirements-dev.txt --upgrade --no-warn-script-location

walker.evolve: ## Start train
	python src/evolve.py
