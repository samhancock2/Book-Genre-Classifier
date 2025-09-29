import os
import torch
from torch.utils.data import Dataset
from torchtext.vocab import GloVe
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import clean_text


class BookDataset(Dataset):
    def __init__(self, csv_path, split, test_size, val_size, random_state, glove_dim=50):
        # ---- Resolve path relative to project root ----
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # one level above src/
        csv_path = os.path.join(project_root, csv_path) if not os.path.isabs(csv_path) else csv_path

        # ---- Load CSV ----
        df = pd.read_csv(csv_path)

        # ---- Encode labels & clean text ----
        self.le = LabelEncoder()
        df['label'] = self.le.fit_transform(df['Category'])
        df['cleaned_desc'] = df['Description'].apply(clean_text)

        # ---- Stratified split ----
        X_temp, X_test, y_temp, y_test = train_test_split(
            df['cleaned_desc'], df['label'],
            test_size=test_size, stratify=df['label'], random_state=random_state
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            stratify=y_temp, random_state=random_state
        )

        if split == "train":
            self.texts, self.labels = X_train, y_train
        elif split == "val":
            self.texts, self.labels = X_val, y_val
        else:
            self.texts, self.labels = X_test, y_test

        # ---- GloVe embeddings ----
        self.glove = GloVe(name='6B', dim=glove_dim)
        self.vectors = torch.stack([self.sentence_to_vec(t) for t in self.texts])
        self.labels = torch.tensor(self.labels.values, dtype=torch.long)

    def sentence_to_vec(self, sentence):
        words = sentence.split()
        vecs = [self.glove[word] for word in words if word in self.glove.stoi]
        if len(vecs) == 0:
            return torch.zeros(self.glove.dim)
        return torch.mean(torch.stack(vecs), dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx]
