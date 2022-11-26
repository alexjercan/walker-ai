### Walker
# ¯¯¯¯¯¯¯¯¯¯¯

walker.install: ## Install dependencies
	pip install -r requirements-dev.txt --upgrade --no-warn-script-location

walker.train: ## Start train
	python src/train.py
