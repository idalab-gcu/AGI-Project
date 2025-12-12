import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm

# 모듈 import
from modules.tokenizer import Tokenizer
from modules.dataset import TestDataset
from modules.model import R2GenLike

# 설정
TEST_JSON = "/content/drive/MyDrive/Final_AGI/test_AGI/annotation/test.json"
MODEL_PATH = "./checkpoints/latest.pth" # 학습된 모델 경로
VOCAB_PATH = "vocab.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test():
    # 1. Load Tokenizer & Vocab
    tokenizer = Tokenizer({"vocab_path": VOCAB_PATH})
    tokenizer.load_vocab()
    
    # 2. Data Loader
    tf = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_set = TestDataset(TEST_JSON, transform=tf)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    
    # 3. Load Model
    model = R2GenLike(vocab_size=len(tokenizer.word2idx)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    print(" Generating Reports...")
    results = []
    
    with torch.no_grad():
        for imgs, path, sid in tqdm(test_loader):
            imgs = imgs.to(DEVICE)
            
            # Beam Search Generation
            pred_ids = model.generate_beam(
                imgs, 
                sos_idx=tokenizer.word2idx["<sos>"], 
                eos_idx=tokenizer.word2idx["<eos>"],
                beam_size=5
            )
            
            # Decoding
            report = tokenizer.decode(pred_ids[0])
            
            print(f"\n[ID: {sid[0]}]")
            print(f"Report: {report}")
            results.append({"id": sid[0], "report": report})

if __name__ == "__main__":
    test()
