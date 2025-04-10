
train:
	poetry run python -m src.cli train

train-overfit:
	poetry run python -m src.cli train_overfit --num-samples=10 --epochs=100

predict-deterministic:
	poetry run python -m src.cli predict_from_dataset_index --split=valid --index=0 --deterministic=True

predict-stochastic:
	poetry run python -m src.cli predict_from_dataset_index --split=valid --index=0 --deterministic=False

# utilities

clear-checkpoints-and-samples:
	rm -rf checkpoints/*
	rm -rf generated_samples/*

clear-wandb-logs:
	rm -rf logs/*

format:
	poetry run black . --line-length=100
