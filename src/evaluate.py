import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from data import BookDataset

def evaluate(model, config, label_encoder):
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

    # report
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
