# src/run.py
import os
import yaml
from datetime import datetime

from train import train_and_validate
from evaluate import evaluate


def main():
    # ---- Load config ----
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # ---- Create experiment folder ----
    base_name = config.get("experiment_name", "exp")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{base_name}_{timestamp}"  # optional timestamp for uniqueness
    save_dir = os.path.join("results", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 Results will be saved in {save_dir}")

    # ---- Train ----
    model, label_encoder, loss_history = train_and_validate(config, save_dir)

    # ---- Evaluate ----
    evaluate(model, config, label_encoder, save_dir)


if __name__ == "__main__":
    main()
