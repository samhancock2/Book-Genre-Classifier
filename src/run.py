# src/run.py
import os
import yaml
from datetime import datetime
from train import train_and_validate
from evaluate import evaluate

# ---- Load config ----
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

# ---- Create a meaningful save directory with training params and timestamp ----
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g., 20250930_153012

embedding_type = config['dataset'].get('embedding_type', 'glove')
epochs = config['training']['epochs']
batch_size = config['training']['batch_size']
lr = config['training']['lr']
embedding_dim = config['embedding']['dim']

if embedding_type == 'glove':
    save_dir_name = f"glove_{embedding_dim}e{epochs}bs{batch_size}lr{lr}t{timestamp}"
elif embedding_type == 'sentence_transformers':
    model_name = config['dataset'].get('transformer_model', 'unknown_model').replace("/", "_")
    save_dir_name = f"transformer_{model_name}_e{epochs}bs{batch_size}lr{lr}t{timestamp}"
else:
    save_dir_name = f"experiment_e{epochs}bs{batch_size}lr{lr}t{timestamp}"

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_dir = os.path.join(project_root, "results", save_dir_name)
os.makedirs(save_dir, exist_ok=True)

print(f"✅ Saving results to: {save_dir}")

# ---- Optional: load transformer model for sentence transformers ----
transformer_model = None
if embedding_type == 'sentence_transformers':
    from sentence_transformers import SentenceTransformer
    transformer_model = SentenceTransformer(config['dataset']['transformer_model']).to("cuda")

# ---- Train and validate ----
model, label_encoder, loss_history = train_and_validate(config, save_dir)

# ---- Evaluate ----
acc, report, cm = evaluate(model, config, label_encoder, save_dir)
