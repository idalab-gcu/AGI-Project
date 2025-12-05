import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class JsonCaptionDataset(Dataset):
    def __init__(self, json_path, tokenizer, transform=None, max_len=100):
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len
        with open(json_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = item.get("image_path", item.get("img"))
        report = item.get("report", item.get("text", ""))
        sid = str(item.get("id", ""))

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f" Image Load Error: {img_path}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # Tokenize
        ids = self.tokenizer(report)
        # Pad or Truncate
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            pad_idx = self.tokenizer.word2idx[self.tokenizer.pad_token]
            ids = ids + [pad_idx] * (self.max_len - len(ids))
        
        return image, torch.tensor(ids, dtype=torch.long), report, img_path, sid

class TestDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = sample["image_path"]
        sid = sample.get("id", f"test_{idx}")
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new('RGB', (256, 256))
            
        if self.transform:
            img = self.transform(img)
        return img, img_path, sid