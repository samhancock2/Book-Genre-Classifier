import yaml
from train import train_and_validate
from evaluate import evaluate

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

model, label_encoder, loss_history = train_and_validate(config)
evaluate(model, config, label_encoder)
