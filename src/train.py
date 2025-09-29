import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import BookDataset
from models import SimpleNN

def train_and_validate(config):
    # datasets
    train_ds = BookDataset(config['dataset']['path'], "train",
                           config['dataset']['test_size'], config['dataset']['val_size'],
                           config['dataset']['random_state'], glove_dim=config['embedding']['dim'])
    val_ds = BookDataset(config['dataset']['path'], "val",
                         config['dataset']['test_size'], config['dataset']['val_size'],
                         config['dataset']['random_state'], glove_dim=config['embedding']['dim'])

    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'])

    # model
    model = SimpleNN(input_dim=config['embedding']['dim'],
                     hidden_dim=config['model']['hidden_dim'],
                     num_classes=len(train_ds.le.classes_))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

    train_loss_history, val_loss_history = [], []

    for epoch in range(config['training']['epochs']):
        # training
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

        # validation
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

    return model, train_ds.le, (train_loss_history, val_loss_history)
