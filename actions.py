import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import f1_score
# Assuming these are in your bert_base.py
from bert_base import BERTModel, CustomDataset, CustomCollator


def train(train_df, num_classes, num_epochs=5, id_col="pmcid", content_col="text", target_col="label"):
    # 1. Setup Data
    #train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    collator = CustomCollator(tokenizer, max_seq_len=64)  # Reduced for speed

    train_loader = DataLoader(
        CustomDataset(train_df, id_col, content_col, target_col),
        batch_size=16,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,  # Adjust based on your CPU cores
        persistent_workers=True,
    )

    # 2. Setup Model
    model = BERTModel("bert-base-uncased", num_classes=num_classes)

    # 3. Setup Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=num_epochs,
        num_nodes=1,
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
    """Loops through a test dataframe and prints the final accuracy."""

    model.eval()

    true_labels = test_df["label"].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions = []
    for _, row in test_df.iterrows():
        pred = predict(model, tokenizer, str(row["sentence"]), device)
        predictions.append(pred)

    f1 = f1_score(true_labels, predictions, average="weighted")
    print(f"F1 Score: {f1:.4f}")

    return f1
