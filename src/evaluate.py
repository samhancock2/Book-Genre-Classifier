# src/evaluate.py
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from data import BookDataset


def evaluate(model, config, label_encoder, save_dir):
    test_ds = BookDataset(config['dataset']['path'], "test",
                          config['dataset']['test_size'], config['dataset']['val_size'],
                          config['dataset']['random_state'], glove_dim=config['embedding']['dim'])
    test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'])

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X.float())
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # ---- Accuracy ----
    acc = accuracy_score(all_labels, all_preds)
    with open(os.path.join(save_dir, "accuracy.txt"), "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
    print(f"✅ Saved accuracy: {acc:.4f}")

    # ---- Classification report ----
    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_)
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    print("✅ Saved classification report")

    # ---- Confusion matrix ----
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"✅ Saved confusion matrix to {cm_path}")
