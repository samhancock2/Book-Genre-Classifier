# src/grid_search.py
import os
import yaml
import itertools
import csv
import shutil
from datetime import datetime
from train import train_and_validate
from evaluate import evaluate

# ---- Load base config ----
with open("configs/config.yaml") as f:
    base_config = yaml.safe_load(f)

# ---- Define hyperparameter grid ----
param_grid = {
    "lr": [0.001, 0.0005, 0.002],
    "batch_size": [32, 64],
    "hidden_dim": [128, 256],
    "epochs": [50],
    "warmup_ratio": [0.05, 0.1, 0.2] 
}

# ---- Build all combinations ----
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
print(f"🔍 Running grid search with {len(param_combinations)} combinations")

# ---- Project root ----
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_root = os.path.join(project_root, "results")

# ---- Prepare CSV summary file ----
summary_file = os.path.join(results_root, f"grid_search_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
with open(summary_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["run_id", "embedding_type", "embedding_dim", "lr", "batch_size", "hidden_dim", "epochs", "accuracy", "save_dir"])

best_acc = -1.0
best_run = None

for i, params in enumerate(param_combinations, start=1):
    print(f"\n🚀 Training model {i}/{len(param_combinations)} with params: {params}")

    # ---- Clone base config ----
    config = yaml.safe_load(open("configs/config.yaml"))  # reload fresh
    config["training"]["lr"] = params["lr"]
    config["training"]["batch_size"] = params["batch_size"]
    config["training"]["epochs"] = params["epochs"]
    config["model"]["hidden_dim"] = params["hidden_dim"]
    config["training"]["warmup_ratio"] = params["warmup_ratio"]

    # ---- Create unique save dir ----
    embedding_type = config['dataset'].get('embedding_type', 'glove')
    embedding_dim = config['embedding']['dim']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if embedding_type == 'glove':
        save_dir_name = f"glove_{embedding_dim}_e{params['epochs']}bs{params['batch_size']}lr{params['lr']}hd{params['hidden_dim']}_{timestamp}"
    elif embedding_type == 'sentence_transformers':
        model_name = config['dataset'].get('transformer_model', 'unknown_model').replace("/", "_")
        save_dir_name = f"transformer_{model_name}_e{params['epochs']}bs{params['batch_size']}lr{params['lr']}hd{params['hidden_dim']}_{timestamp}"
    else:
        save_dir_name = f"experiment_e{params['epochs']}bs{params['batch_size']}lr{params['lr']}hd{params['hidden_dim']}_{timestamp}"

    save_dir = os.path.join(results_root, save_dir_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"✅ Saving results to: {save_dir}")

    # ---- Train and validate ----
    model, label_encoder, loss_history = train_and_validate(config, save_dir)

    # ---- Evaluate ----
    acc, report, cm = evaluate(model, config, label_encoder, save_dir)

    # ---- Save metrics ----
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nHyperparameters:\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")


    # ---- Append to CSV summary ----
    with open(summary_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            i,
            embedding_type,
            embedding_dim,
            params["lr"],
            params["batch_size"],
            params["hidden_dim"],
            params["epochs"],
            f"{acc:.4f}",
            save_dir
        ])

    # ---- Track best run ----
    if acc > best_acc:
        best_acc = acc
        best_run = {
            "id": i,
            "acc": acc,
            "save_dir": save_dir,
            "params": params,
            "embedding_type": embedding_type,
            "embedding_dim": embedding_dim
        }

    print(f"📊 Finished model {i}/{len(param_combinations)} | Accuracy: {acc:.4f}")

# ---- Copy best run to best_model dir ----
if best_run:
    best_dir = os.path.join(results_root, "best_model")
    if os.path.exists(best_dir):
        shutil.rmtree(best_dir)
    shutil.copytree(best_run["save_dir"], best_dir)

    print(f"\n🌟 Best model (Run {best_run['id']})")
    print(f"   Accuracy: {best_run['acc']:.4f}")
    print(f"   Params: {best_run['params']}")
    print(f"   Saved to: {best_dir}")

print(f"\n✅ Grid search complete. Summary saved to: {summary_file}")
