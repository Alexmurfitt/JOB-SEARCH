# scripts/ner.py

import os
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments
)

# 1. Definición de etiquetas y mapeos
LABEL_LIST = [
    "O",
    "B-Skill", "I-Skill",
    "B-Tool", "I-Tool",
    "B-Certification", "I-Certification",
    "B-Education", "I-Education",
    "B-Language", "I-Language",
    "B-Seniority", "I-Seniority"
]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS = len(LABEL_LIST)

# 2. Carga del dataset anotado
dataset = load_dataset(
    "json",
    data_files={"train": "data/annotated/ner_dataset.jsonl"}
)

# 3. División en train/test (90% / 10%)
split = dataset["train"].train_test_split(test_size=0.1, seed=42)
split["validation"] = split.pop("test")

# 4. Preparar tokenizer y modelo
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

# 5. Función de tokenización y alineación de etiquetas
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length"
    )
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                aligned_labels.append(LABEL2ID[label[word_id]])
            else:
                current_label = label[word_id]
                if current_label.startswith("B-"):
                    aligned_labels.append(LABEL2ID[current_label.replace("B-", "I-")])
                else:
                    aligned_labels.append(LABEL2ID[current_label])
            prev_word_id = word_id
        labels.append(aligned_labels)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 6. Aplicar tokenización al dataset
tokenized_datasets = {
    split_name: ds.map(tokenize_and_align_labels, batched=True)
    for split_name, ds in split.items()
}

# 7. Métrica de evaluación
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=2)
    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for l in label if l != -100]
        for label in labels
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"]
    }

# 8. Configuración de entrenamiento (sin evaluation_strategy)
training_args = TrainingArguments(
    output_dir="models/ner_model",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=50,
    eval_steps=50,
    save_steps=50,
    save_total_limit=2
)

# 9. Entrenamiento y guardado
def main():
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model("models/ner_model")
    tokenizer.save_pretrained("models/ner_model")
    print("✅ Fine-tuning completado y modelo guardado en models/ner_model")

if __name__ == "__main__":
    main()