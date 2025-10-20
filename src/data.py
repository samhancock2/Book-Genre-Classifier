import os
import torch
from torch.utils.data import Dataset
from torchtext.vocab import GloVe
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import clean_text


class BookDataset(Dataset):
    def __init__(self, 
                 csv_path, 
                 split, 
                 test_size, 
                 val_size, 
                 random_state, 
                 embedding_type='glove', 
                 glove_dim=50, 
                 transformer_model=None, 
                 device='cpu'):

        # Path relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(project_root, csv_path) if not os.path.isabs(csv_path) else csv_path

        
        df = pd.read_csv(csv_path)

        
        self.le = LabelEncoder()
        df['label'] = self.le.fit_transform(df['Category'])
        df['cleaned_desc'] = df['Description'].apply(clean_text)

        
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
        else:  # test
            self.texts, self.labels = X_test, y_test

        # Prepare embeddings 
        if embedding_type == 'glove':
            self.glove = GloVe(name='6B', dim=glove_dim)
            self.vectors = torch.stack([self.sentence_to_vec(t) for t in self.texts])
        elif embedding_type == 'sentence_transformers':
            if transformer_model is None:
                raise ValueError("Provide a transformer_model for sentence-transformers embeddings")
            self.vectors = transformer_model.encode(
                self.texts.tolist(),
                batch_size=64,
                convert_to_tensor=True,
                device=device
            )
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")

        # Convert labels to tensor 
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
