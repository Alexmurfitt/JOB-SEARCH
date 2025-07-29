import argparse
import pickle
import torch
from transformers import BertTokenizerFast, BertForTokenClassification


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_model_and_tokenizer(model_dir):
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForTokenClassification.from_pretrained(model_dir)
    model.eval()
    with open(f"{model_dir}/label2id.pkl", "rb") as f:
        label2id = pickle.load(f)
    id2label = {v: k for k, v in label2id.items()}
    return model, tokenizer, id2label

def predict_sequences(texts, model, tokenizer):
    pred_tags = []
    for tokens in texts:
        if isinstance(tokens, str):
            tokens = tokens.split()
        elif not isinstance(tokens, list):
            raise ValueError(f"Formato no válido de entrada: {tokens}")

        inputs = tokenizer(tokens, return_tensors="pt", is_split_into_words=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)[0].tolist()
        word_ids = inputs.word_ids()

        seq = []
        seen = set()
        for i, word_id in enumerate(word_ids):
            if word_id is not None and word_id not in seen:
                seq.append(predictions[i])
                seen.add(word_id)
        pred_tags.append(seq)
    return pred_tags

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--val_texts', required=True)
    parser.add_argument('--output_pred', required=True)
    args = parser.parse_args()

    model, tokenizer, id2label = load_model_and_tokenizer(args.model_dir)
    val_texts = load_pickle(args.val_texts)

    print("🔍 Realizando predicciones...")
    pred_tags = predict_sequences(val_texts, model, tokenizer)
    save_pickle(pred_tags, args.output_pred)
    print(f"✅ Etiquetas predichas guardadas en: {args.output_pred}")

if __name__ == '__main__':
    main()
