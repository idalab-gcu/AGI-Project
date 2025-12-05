import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm

# 모듈 import
from modules.tokenizer import Tokenizer
from modules.dataset import JsonCaptionDataset
from modules.model import R2GenLike
from modules.metrics import compute_metrics

# --- Config ---
CONFIG = {
    "train_json": "/content/drive/MyDrive/Final_AGI/test_AGI/annotation/train.json",
    "val_json": "/content/drive/MyDrive/Final_AGI/test_AGI/annotation/val.json",
    "save_dir": "./checkpoints",
    "vocab_path": "vocab.pkl",
    "batch_size": 16,
    "epochs": 30,
    "lr": 3e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
os.makedirs(CONFIG["save_dir"], exist_ok=True)

def train():
    # 1. Tokenizer
    tokenizer = Tokenizer({"ann_path": CONFIG["train_json"], "vocab_path": CONFIG["vocab_path"]})
    if os.path.exists(CONFIG["vocab_path"]):
        tokenizer.load_vocab()
    else:
        tokenizer.build_vocab()
        tokenizer.save_vocab()

    # 2. Dataset & Loader
    tf = T.Compose([T.Resize((256,256)), T.RandomCrop((224,224)), T.ToTensor()])
    
    train_set = JsonCaptionDataset(CONFIG["train_json"], tokenizer, transform=tf)
    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
    
    # 3. Model
    model = R2GenLike(vocab_size=len(tokenizer.word2idx)).to(CONFIG["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx["<pad>"])

    print(" Training Start!")
    for ep in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}")
        for imgs, tgt_ids, _, _, _ in pbar:
            imgs, tgt_ids = imgs.to(CONFIG["device"]), tgt_ids.to(CONFIG["device"])
            
            # Forward
            logits = model(imgs, tgt_ids[:-1, :]) 
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_ids[1:, :].reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss/len(train_loader))
        
        # Save Checkpoint
        torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "latest.pth"))
        print(f" Saved epoch {ep+1}")

if __name__ == "__main__":
    train()
