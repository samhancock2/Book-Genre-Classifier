# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from data import BookDataset
from models import SimpleNN


def train_and_validate(config, save_dir):
    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Optional: Sentence Transformer model ----
    transformer_model = None
    if config['dataset']['embedding_type'] == 'sentence_transformers':
        transformer_model = SentenceTransformer(config['dataset']['transformer_model']).to(device)

    # ---- Datasets ----
    train_ds = BookDataset(
        csv_path=config['dataset']['path'],
        split="train",
        test_size=config['dataset']['test_size'],
        val_size=config['dataset']['val_size'],
        random_state=config['dataset']['random_state'],
        embedding_type=config['dataset']['embedding_type'],
        glove_dim=config['embedding']['dim'],
        transformer_model=transformer_model,
        device=device
    )

    val_ds = BookDataset(
        csv_path=config['dataset']['path'],
        split="val",
        test_size=config['dataset']['test_size'],
        val_size=config['dataset']['val_size'],
        random_state=config['dataset']['random_state'],
        embedding_type=config['dataset']['embedding_type'],
        glove_dim=config['embedding']['dim'],
        transformer_model=transformer_model,
        device=device
    )

    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'])

    # ---- Model ----
    input_dim = train_ds.vectors.shape[1]  # automatically adapt to GloVe or transformer
    model = SimpleNN(
        input_dim=input_dim,
        hidden_dim=config['model']['hidden_dim'],
        num_classes=len(train_ds.le.classes_)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

    train_loss_history, val_loss_history = [], []

    # ---- Early stopping ----
    best_val_loss = float('inf')
    patience = config['training'].get('early_stopping_patience', 5)
    epochs_no_improve = 0
    best_model_state = None

    # ---- Training loop ----
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X.float())
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # ---- Validation ----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X.float())
                loss = criterion(outputs, y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{config['training']['epochs']} "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ---- Early stopping ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹ Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break

    # ---- Save loss curves ----
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label="Train Loss")
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()
    print(f"✅ Saved loss curves to {save_dir}/loss_curve.png")

    return model, train_ds.le, (train_loss_history, val_loss_history)
