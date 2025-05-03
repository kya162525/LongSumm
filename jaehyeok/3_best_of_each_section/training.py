#!/usr/bin/env python
"""
keysent_train.py
End‑to‑end pipeline: Baseline → SimCSE warm start → Hard‑Neg CL → BERTSUM‑Ext + mixed loss
"""

import os, json, random, math
from typing import List, Dict, Tuple
from dataclasses import dataclass

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------- 1. 데이터 ------------------------- #
def sent_split(paragraph: str) -> List[str]:
    """간단 KSS fallback; 실제론 kss.split_sentences 사용 권장"""
    return [s.strip() for s in paragraph.split(" . ") if s.strip()]

class KeySentDataset(Dataset):
    """
    expects each item: {"paragraph": str, "label_sent": str}
    """

    def __init__(self, data: List[Dict], tokenizer, max_len=1024):
        self.data = data
        self.tok = tokenizer
        self.max_len = max_len

    def _encode(self, sentences):
        # 각 문장 앞에 [CLS] 삽입해 문장 벡터 분리
        tokens, cls_positions = [], []
        for sent in sentences:
            cls_positions.append(len(tokens))
            tokens += [self.tok.cls_token] + self.tok.tokenize(sent)
        tokens = tokens[: self.max_len - 1] + [self.tok.sep_token]

        enc = self.tok.convert_tokens_to_ids(tokens)
        attn = [1] * len(enc)
        pad_len = self.max_len - len(enc)
        enc += [self.tok.pad_token_id] * pad_len
        attn += [0] * pad_len
        return torch.tensor(enc), torch.tensor(attn), torch.tensor(cls_positions)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sents = sent_split(sample["paragraph"])
        label_idx = sents.index(sample["label_sent"])
        enc, attn, cls_positions = self._encode(sents)
        return {
            "input_ids": enc,
            "attention_mask": attn,
            "cls_pos": cls_positions,
            "label": torch.tensor(label_idx),
            "sents": sents,  # Phase 2 hard‑neg 생성용
        }

    def __len__(self):
        return len(self.data)

# ------------------------- 2. 손실 ------------------------- #
class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al.)"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, features, labels):
        """
        features: [N, d]  labels: [N]
        같은 label(=문단 내 핵심/양성) 끼리 pull, 다른 label push
        """
        device = features.device
        sim = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1) / self.t
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        # log_prob
        exp = torch.exp(sim) * (1 - torch.eye(len(labels), device=device))
        log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
        loss = -(mask * log_prob).sum(1) / mask.sum(1)
        return loss.mean()

# ------------------------- 3. 모델 ------------------------- #
class SentenceEncoder(nn.Module):
    def __init__(self, plm_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(plm_name)
        self.hidden = self.backbone.config.hidden_size
        # 문장 분류 head
        self.cls_head = nn.Linear(self.hidden, 1)  # score per sentence
        # projection head (SupCon)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
        )

    def forward(self, input_ids, attention_mask, cls_pos):
        out = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state
        sent_vecs = out[torch.arange(len(cls_pos)), cls_pos]  # [total_sents, hidden]
        logits = self.cls_head(sent_vecs).squeeze(-1)
        proj = F.normalize(self.proj(sent_vecs), dim=-1)
        return logits, proj

class BertSumExt(nn.Module):
    """
    Liu & Lapata 2019 구조: 단어 BERT → 문장 CLS vectors → inter‑sentence Transformer
    """
    def __init__(self, plm_name, n_layers=2, n_heads=8):
        super().__init__()
        self.enc = SentenceEncoder(plm_name)
        hidden = self.enc.hidden
        self.inter = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden, nhead=n_heads, dim_feedforward=4*hidden, activation="gelu"),
            num_layers=n_layers,
        )
        self.score = nn.Linear(hidden, 1)
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, input_ids, attention_mask, cls_pos):
        _, sent_vecs = self.enc.backbone(input_ids, attention_mask=attention_mask).last_hidden_state, None
        sent_vecs = sent_vecs = self.enc.backbone(input_ids, attention_mask=attention_mask).last_hidden_state[
            torch.arange(len(cls_pos)), cls_pos
        ]
        sent_vecs = self.inter(sent_vecs.unsqueeze(1)).squeeze(1)  # [S, H]
        logits = self.score(sent_vecs).squeeze(-1)
        proj = F.normalize(self.proj(sent_vecs), dim=-1)
        return logits, proj

# ------------------------- 4. 학습 헬퍼 ------------------------- #
@dataclass
class TrainConfig:
    plm: str
    lr: float = 2e-5
    epochs: int = 3
    warmup_ratio: float = 0.1
    batch: int = 4
    max_len: int = 1024
    temperature: float = 0.07
    lambda_cl: float = 0.0  # Phase 3에서만 0.2

def hard_negative_indices(sents: List[str], label_idx: int, k=3) -> List[int]:
    """가장 유사하지만 정답이 아닌 문장 k개 선택(TF‑IDF cosine)."""
    tfidf = TfidfVectorizer().fit_transform(sents).toarray()
    sim = tfidf @ tfidf[label_idx]
    neg_candidates = [(i, sim) for i, sim in enumerate(sim) if i != label_idx]
    neg_candidates.sort(key=lambda x: -x[1])
    return [i for i, _ in neg_candidates[:k]]

def train_phase(model, train_dl, cfg: TrainConfig, save_dir: str):
    model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    total = cfg.epochs * len(train_dl)
    sch = get_linear_schedule_with_warmup(opt, int(total * cfg.warmup_ratio), total)
    ce_loss_fn = nn.CrossEntropyLoss()
    scl_loss_fn = SupConLoss(cfg.temperature)

    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_dl:
            ids, attn, pos, label = [b.cuda() for b in batch.values() if torch.is_tensor(b)]
            logits, proj = model(ids, attn, pos)
            # ----- CE -----
            ce = ce_loss_fn(logits.view(1, -1), label)  # label 하나
            # ----- Contrastive -----
            if cfg.lambda_cl > 0:
                # 양성=label, 음성=hard neg
                neg_indices = hard_negative_indices(batch["sents"], label.item(), k=3)
                sup_labels = torch.zeros(len(batch["sents"]), dtype=torch.long, device=ids.device)
                sup_labels[label] = 1  # 같은 클래스 = 1, 나머지 0
                scl = scl_loss_fn(proj, sup_labels)
                loss = ce + cfg.lambda_cl * scl
            else:
                loss = ce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()

        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch{epoch}.pt"))

# ------------------------- 5. 메인 ------------------------- #
def main():
    # ---- 0. 데이터 ----
    raw = load_dataset("json", data_files={"train":"train.json"})["train"]  # {"paragraph", "label_sent"}
    tok = AutoTokenizer.from_pretrained("klue/roberta-large")
    train_ds = KeySentDataset(raw, tok)
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=lambda x: x)

    # ---- Phase 0 Baseline ----
    cfg0 = TrainConfig(plm="klue/roberta-large")
    m0 = SentenceEncoder(cfg0.plm)
    train_phase(m0, train_dl, cfg0, "phase0")

    # ---- Phase 1 SimCSE warm ----
    cfg1 = TrainConfig(plm="BM-K/KoSimCSE-bert")
    m1 = SentenceEncoder(cfg1.plm)
    train_phase(m1, train_dl, cfg1, "phase1")

    # ---- Phase 2 Hard‑Neg + SupCon ----
    cfg2 = TrainConfig(plm="BM-K/KoSimCSE-bert", lambda_cl=0.5, lr=1e-5, epochs=4)
    m2 = SentenceEncoder(cfg2.plm)
    m2.load_state_dict(torch.load("phase1/epoch2.pt"))
    train_phase(m2, train_dl, cfg2, "phase2")

    # ---- Phase 3 BERTSUM‑Ext + mixed loss ----
    cfg3 = TrainConfig(plm="BM-K/KoSimCSE-bert", lambda_cl=0.2, lr=1e-5, epochs=4)
    m3 = BertSumExt(cfg3.plm)
    # backbone 가중치 warm start
    m3.enc.backbone.load_state_dict(m2.backbone.state_dict())
    train_phase(m3, train_dl, cfg3, "phase3")

if __name__ == "__main__":
    main()