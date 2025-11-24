import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import config


class TranslationDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = pd.read_json(data_path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.data[idx]['zh'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[idx]['en'], dtype=torch.long)
        return input_tensor, target_tensor


def get_dataloader(is_train=True):
    data_path = config.PROCESSED_DATA_DIR / ("train_dataset.jsonl" if is_train else "test_dataset.jsonl")
    train_dataset = TranslationDataset(data_path)
    return DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    train_dataloader = get_dataloader()
    print(f'train dataloader 个数: {len(train_dataloader)}')

    for inputs, target in train_dataloader:
        print(inputs.shape, target.shape)
        break
