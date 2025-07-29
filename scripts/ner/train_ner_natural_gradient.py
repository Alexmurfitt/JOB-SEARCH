#!/usr/bin/env python
# train_ner_natural_gradient.py

import os
import argparse
import pickle
import json
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForTokenClassification, BertTokenizerFast, AdamW

# Intentamos importar NaturalGradient de Pyro
try:
    from pyro.optim import NaturalGradient
    NG_AVAILABLE = True
    ng_name = "NaturalGradient(Pyro)"
except (ImportError, ModuleNotFoundError):
    NG_AVAILABLE = False
    ng_name = "AdamW (fallback)"
    print("⚠️  NaturalGradient no disponible, usando AdamW en su lugar.")

class NERDataset(Dataset):
    def __init__(self, texts: List[str], tags: List[List[int]], tokenizer, max_len: int = 128):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tags = self.tags[idx]
        encoding = self.tokenizer(
            text.split(),
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        labels = torch.full((self.max_len,), -100, dtype=torch.long)
        word_ids = encoding.word_ids(batch_index=0)
        for i, word_id in enumerate(word_ids):
            if word_id is not None and word_id < len(tags):
                labels[i] = tags[word_id]
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding['labels'] = labels
        return encoding

def train_ng_ner(
    model_name: str,
    train_texts, train_tags,
    val_texts, val_tags,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 5e-5,
    device: str = 'cuda'
):
    # 1) Detectar y mapear etiquetas automáticamente
    unique_labels = sorted({lbl for seq in train_tags + val_tags for lbl in seq})
    label2id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    print("🔖 Etiquetas detectadas y mapeadas automáticamente:")
    for lbl, idx in label2id.items():
        print(f"  {idx}: {lbl}")

    train_tags_ids = [[label2id[lbl] for lbl in seq] for seq in train_tags]
    val_tags_ids   = [[label2id[lbl] for lbl in seq] for seq in val_tags]

    # 2) Inicializar modelo y tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label={i: lbl for lbl, i in label2id.items()},
        label2id=label2id
    )
    model.to(device)

    # 3) Preparar datasets y loaders
    train_ds = NERDataset(train_texts, train_tags_ids, tokenizer)
    val_ds   = NERDataset(val_texts,   val_tags_ids,   tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    # 4) Configurar optimizador
    if NG_AVAILABLE:
        base_opt = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = NaturalGradient(base_opt)
    else:
        optimizer = AdamW(model.parameters(), lr=lr)

    print(f"📊 Usando optimizador: {ng_name}")
    # 5) Loop de entrenamiento
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if NG_AVAILABLE:
                optimizer.step(lambda: loss)
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} — Train loss: {avg_train:.4f}")

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                total_val += outputs.loss.item()
        avg_val = total_val / len(val_loader)
        print(f"Epoch {epoch}/{epochs} — Val   loss: {avg_val:.4f}")

    # 6) Guardar modelo, tokenizer y mapping de etiquetas
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Guardar labels.json
    with open(os.path.join(output_dir, "labels.json"), "w", encoding="utf8") as fp:
        json.dump(label2id, fp, ensure_ascii=False, indent=2)
    print(f"✅ Modelo, tokenizer y labels.json guardados en: {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT-NER (Natural Gradient fallback a AdamW)"
    )
    parser.add_argument('--model_name',  type=str, required=True)
    parser.add_argument('--train_texts', type=str, required=True)
    parser.add_argument('--train_tags',  type=str, required=True)
    parser.add_argument('--val_texts',   type=str, required=True)
    parser.add_argument('--val_tags',    type=str, required=True)
    parser.add_argument('--output_dir',  type=str, default='./ng_ner_model')
    parser.add_argument('--epochs',      type=int, default=3)
    parser.add_argument('--batch_size',  type=int, default=16)
    parser.add_argument('--lr',          type=float, default=5e-5)
    parser.add_argument('--device',      type=str, default='cuda')
    args = parser.parse_args()

    # Verificar existencia de ficheros
    for path in [args.train_texts, args.train_tags, args.val_texts, args.val_tags]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No existe el fichero de datos: {path}")

    # Cargar datos desde pickle
    train_texts = pickle.load(open(args.train_texts, 'rb'))
    train_tags  = pickle.load(open(args.train_tags,  'rb'))
    val_texts   = pickle.load(open(args.val_texts,   'rb'))
    val_tags    = pickle.load(open(args.val_tags,    'rb'))

    train_ng_ner(
        model_name=args.model_name,
        train_texts=train_texts,
        train_tags=train_tags,
        val_texts=val_texts,
        val_tags=val_tags,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )

if __name__ == '__main__':
    main()
