import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import resnet101, ResNet101_Weights

class ImageEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        m = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(m.children())[:-2])
        self.conv_proj = nn.Conv2d(2048, d_model, kernel_size=1)
    def forward(self, x):
        feat = self.backbone(x)
        feat = self.conv_proj(feat)
        B,C,H,W = feat.shape
        seq = feat.flatten(2).permute(2,0,1)
        return seq, (H,W)

class MeshedMemory(nn.Module):
    def __init__(self, num_slots, d_model):
        super().__init__()
        self.mem = nn.Parameter(torch.randn(num_slots, 1, d_model) * 0.02)
        self.gate = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(),
                                  nn.Linear(d_model,d_model), nn.Sigmoid())
    def forward(self, B):
        mem = self.mem.repeat(1,B,1)
        return mem * self.gate(mem)

class R2GenLike(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_slots=40):
        super().__init__()
        self.encoder = ImageEncoder(d_model)
        self.mem = MeshedMemory(num_slots, d_model)
        self.pos_enc = self._init_pos_enc(d_model)
        
        layer = nn.TransformerDecoderLayer(d_model, nhead=8, dim_feedforward=2048, dropout=0.1, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=3)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def _init_pos_enc(self, d_model, max_len=4096):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0,max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        return nn.Parameter(pe.unsqueeze(1), requires_grad=False)

    def encode(self, imgs):
        seq, _ = self.encoder(imgs)
        # Memory 결합
        enc = torch.cat([self.mem(seq.size(1)), seq], dim=0)
        return enc + self.pos_enc[:enc.size(0)]

    def forward(self, imgs, tgt_ids):
        enc = self.encode(imgs)
        tgt_emb = self.tok_emb(tgt_ids) + self.pos_enc[:tgt_ids.size(0)]
        
        T = tgt_ids.size(0)
        mask = torch.triu(torch.ones(T,T, device=imgs.device), diagonal=1).bool()
        
        out = self.decoder(tgt=tgt_emb, memory=enc, tgt_mask=mask)
        return self.lm_head(out)

    @torch.no_grad()
    def generate_beam(self, imgs, sos_idx, eos_idx, max_len=100, beam_size=3):
        # Beam Search 
        self.eval()
        enc = self.encode(imgs)
        B = imgs.size(0)
        
        # [seq, score]
        beams = [[([sos_idx], 0.0)] for _ in range(B)]
        
        for _ in range(max_len):
            new_beams = []
            for b in range(B):
                candidates = []
                for seq, score in beams[b]:
                    if seq[-1] == eos_idx:
                        candidates.append((seq, score))
                        continue
                    
                    inp = torch.tensor([seq], device=imgs.device).permute(1,0) # (Seq, 1)
                    tgt = self.tok_emb(inp) + self.pos_enc[:inp.size(0)]
                    out = self.decoder(tgt, enc[:,b:b+1])
                    log_prob = F.log_softmax(self.lm_head(out)[-1, 0, :], dim=-1)
                    
                    vals, idxs = log_prob.topk(beam_size)
                    for v, i in zip(vals, idxs):
                        candidates.append((seq + [i.item()], score + v.item()))
                
                # 정렬 후 상위 k개 선택
                ordered = sorted(candidates, key=lambda x: x[1], reverse=True)
                new_beams.append(ordered[:beam_size])
            beams = new_beams
            
        # 가장 높은 점수의 시퀀스 반환
        final_preds = []
        for b in range(B):
            final_preds.append(beams[b][0][0]) # Best seq
        return final_preds
