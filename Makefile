
train:
	poetry run python -m src.cli train

format:
	poetry run black . --line-length=100
