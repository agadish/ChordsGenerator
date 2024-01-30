#!/usr/bin/env python3

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import kaggle
from sklearn.model_selection import train_test_split


class ChordDataset(Dataset):
    @classmethod
    def load_dataset(cls, output_dir='./dataset', kaggle_dataset_name='eitanbentora/chords-and-lyrics-dataset'):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(kaggle_dataset_name, path=output_dir, unzip=True)


    def __init__(self, df: pd.DataFrame, split_ratio=(0.8, 0.1, 0.1)):
        self.df = df
        # self.tokenizer = CustomTokenizer.from_pretrained("your_pretrained_tokenizer")

        train_data, test_data = train_test_split(self.df, test_size=split_ratio[2], random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=split_ratio[1]/(1 - split_ratio[2]), random_state=42)

        # Choose the appropriate split
        self.data = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }

        # Set the initial split to 'train'
        self.set_split('train')

    def set_split(self, split='train'):
        self.current_split = split
        self.current_data = self.data[split]

    def __len__(self):
        return len(self.current_data)

    def __getitem__(self, idx):
        lyrics = self.current_data.iloc[idx]['lyrics']
        chords = self.current_data.iloc[idx]['chords']

        return {'lyrics': lyrics, 'chords': chords}

        # # Tokenize lyrics and chords using your custom tokenizer
        # input_ids = self.tokenizer.encode(lyrics, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        # labels = self.tokenizer.encode(chords, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

        # return {
        #     'input_ids': input_ids.flatten(),
        #     'labels': labels.flatten()
        # }

dataset = ChordDataset(df)
print(dataset[40])
# print(dataset.columns)
print(dataset)  