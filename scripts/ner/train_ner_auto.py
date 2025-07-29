import argparse
import pickle
import os
import time
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW

try:
    from pyro.optim import NaturalGradient
    pyro_available = True
except ImportError:
    pyro_available = False

class NERDataset(Dataset):
    def __init__(self, texts, tags):
        self.texts = texts
        self.tags = tags

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.tags[idx]

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def build_label_map(tags_list):
    flat_tags = [tag for seq in tags_list for tag in seq]
    unique_tags = sorted(set(flat_tags))
    label2id = {label: i for i, label in enumerate(unique_tags)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label

def convert_tags_to_ids(tags_list, label2id):
    return [[label2id[label] for label in seq] for seq in tags_list]

def align_labels(texts, tags, tokenizer):
    tokenized_inputs = tokenizer(texts, padding=True, truncation=True, is_split_into_words=True, return_tensors="pt")
    labels = []
    for i, label in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    return tokenized_inputs, torch.tensor(labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_texts', required=True)
    parser.add_argument('--train_tags', required=True)
    parser.add_argument('--val_texts', required=True)
    parser.add_argument('--val_tags', required=True)
    parser.add_argument('--model_name', default='bert-base-multilingual-cased')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output_dir', default='./models/ner_auto')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("🔁 Cargando datos...")
    train_texts = load_pickle(args.train_texts)
    train_tags = load_pickle(args.train_tags)
    val_texts = load_pickle(args.val_texts)
    val_tags = load_pickle(args.val_tags)

    label2id, id2label = build_label_map(train_tags + val_tags)
    num_labels = len(label2id)
    print(f"🔢 Número de etiquetas detectadas automáticamente: {num_labels} ({list(label2id.keys())})")

    train_tags_ids = convert_tags_to_ids(train_tags, label2id)
    val_tags_ids = convert_tags_to_ids(val_tags, label2id)

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    model = BertForTokenClassification.from_pretrained(args.model_name, num_labels=num_labels)
    model.to(args.device)

    train_inputs, train_labels = align_labels(train_texts, train_tags_ids, tokenizer)
    val_inputs, val_labels = align_labels(val_texts, val_tags_ids, tokenizer)

    train_inputs['labels'] = train_labels
    val_inputs['labels'] = val_labels

    train_dataset = torch.utils.data.TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_inputs['input_ids'], val_inputs['attention_mask'], val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    print("🚀 Entrenamiento iniciado...")
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = [b.to(args.device) for b in batch]
            outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"🟢 Epoch {epoch + 1}/{args.epochs} — Loss: {total_loss:.4f}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, 'label2id.pkl'), 'wb') as f:
        pickle.dump(label2id, f)
    print(f"✅ Modelo y mapeo guardados en: {args.output_dir}")
    print(f"⏱️ Tiempo total: {time.time() - start_time:.2f} segundos")

if __name__ == '__main__':
    main()
