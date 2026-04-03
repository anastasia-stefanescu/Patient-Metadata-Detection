import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import BertTokenizer

# Assuming these are in your bert_base.py
from bert_base import BERTModel, CustomDataset, CustomCollator


def train(train_df, target_col, num_classes):
    # 1. Setup Data
    #train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    collator = CustomCollator(tokenizer, max_seq_len=64)  # Reduced for speed

    train_loader = DataLoader(
        CustomDataset(train_df, target_col),
        batch_size=16,
        shuffle=True,
        collate_fn=collator,
    )

    # 2. Setup Model
    model = BERTModel("bert-base-uncased", num_classes=num_classes)

    # 3. Setup Silent Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=3,
        logger=False,  # No log folders
        enable_checkpointing=False,  # No weight files saved
        enable_model_summary=False,  # No big architecture printout
        enable_progress_bar=True,  # Keep this if you want to see if it's moving
    )

    # 4. Train
    trainer.fit(model, train_loader)
    return model, tokenizer

def predict(model, tokenizer, sentence, device):
    """Predicts a single class for a single sentence."""

    model.eval()

    inputs = tokenizer(
        sentence, return_tensors="pt", truncation=True, padding=True, max_length=64
    ).to(device)

    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])

    return torch.argmax(logits, dim=1).item()


def evaluate(model, test_df, tokenizer):
    """Quickly loops through a test dataframe and prints the final accuracy."""
    model.eval()
    correct = 0
    total = len(test_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Evaluating on {total} samples...")
    for _, row in test_df.iterrows():
        pred = predict(model, tokenizer, str(row["sentence"]), device)
        if int(pred) == int(row["target"]):
            correct += 1

    accuracy = (correct / total) * 100
    print(f"Final Test Accuracy: {accuracy:.2f}%")
