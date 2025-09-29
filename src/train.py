# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data import BookDataset
from models import SimpleNN


def train_and_validate(config, save_dir):
    # ---- Datasets ----
    train_ds = BookDataset(config['dataset']['path'], "train",
                           config['dataset']['test_size'], config['dataset']['val_size'],
                           config['dataset']['random_state'], glove_dim=config['embedding']['dim'])
    val_ds = BookDataset(config['dataset']['path'], "val",
                         config['dataset']['test_size'], config['dataset']['val_size'],
                         config['dataset']['random_state'], glove_dim=config['embedding']['dim'])

    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'])

    # ---- Model ----
    model = SimpleNN(input_dim=config['embedding']['dim'],
                     hidden_dim=config['model']['hidden_dim'],
                     num_classes=len(train_ds.le.classes_))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

    train_loss_history, val_loss_history = [], []

    # ---- Early stopping ----
    best_val_loss = float('inf')
    patience = config['training'].get('early_stopping_patience', 5)
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(config['training']['epochs']):
        # ---- Training ----
        model.train()
        total_loss = 0
        for X, y in train_loader:
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
        total_val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                outputs = model(X.float())
                loss = criterion(outputs, y)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{config['training']['epochs']} "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ---- Early stopping check ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹ Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)  # restore best weights
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
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()
    print(f"✅ Saved loss curves to {save_dir}/loss_curve.png")

    return model, train_ds.le, (train_loss_history, val_loss_history)
