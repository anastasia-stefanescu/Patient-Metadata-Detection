import os
import torch
from datetime import datetime
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from pytorch_lightning.callbacks import TQDMProgressBar


class CustomDataset(Dataset):
    def __init__(self, df, id_col, content_col, target_col):
        self.df = df
        self.id_col = id_col
        self.content_col = content_col
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return {
            "pmcid": row[self.id_col],
            "content": str(row[self.content_col]),
            "class": row[self.target_col],
        }


class CustomCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, input_batch: list[dict]) -> dict:
        sentences = [instance["content"] for instance in input_batch]
        targets = [instance["class"] for instance in input_batch]

        tokenized_batch = self.tokenizer(
            sentences,
            padding=True,  # Dynamic padding to longest in batch
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized_batch["input_ids"],
            "attention_mask": tokenized_batch["attention_mask"],
            "targets": torch.tensor(targets, dtype=torch.float),
        }


# Model for classification
class BERTModel(pl.LightningModule):
    def __init__(
        self, model_name: str, num_classes: int, lr: float = 2e-5, class_weights=None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModel.from_pretrained(model_name)
        self.output_layer = nn.Linear(self.model.config.hidden_size, num_classes)
        self.register_buffer("class_weights", class_weights)
        self.lr = lr

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0, :]
        return self.output_layer(cls_output)

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(
            logits, batch["targets"].long()
        )
        preds = torch.argmax(logits, dim=1)

        acc = (preds == batch["targets"].long()).float().mean()
        f1 = f1_score(batch["targets"].cpu(), preds.cpu(), average="weighted")

        self.log("train_f1", f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(
            logits, batch["targets"].long()
        )
        preds = torch.argmax(logits, dim=1)

        acc = (preds == batch["targets"].long()).float().mean()
        f1 = f1_score(batch["targets"].cpu(), preds.cpu(), average="weighted")

        self.log("val_f1", f1, prog_bar=True, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)


def train(
    train_df,
    model_name="bert-base-uncased",
    num_classes=2,
    num_epochs=3,
    id_col="pmcid",
    content_col="text",
    target_col="label",
):
    # 1. Setup Data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collator = CustomCollator(tokenizer, max_seq_len=64)  # Reduced for speed

    train_loader = DataLoader(
        CustomDataset(train_df, id_col, content_col, target_col),
        batch_size=16,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # Must be 0 in notebooks (multiprocessing can't pickle notebook-defined classes)
        persistent_workers=False,
    )

    # 2. Setup Model
    weights = compute_class_weight(
        "balanced", classes=np.unique(train_df[target_col]), y=train_df[target_col]
    )
    class_weights = torch.tensor(weights, dtype=torch.float)
    model = BERTModel(model_name, num_classes=num_classes, class_weights=class_weights)

    # 3. Setup Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=num_epochs,
        num_nodes=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        callbacks=[TQDMProgressBar(refresh_rate=1)],
    )

    # 4. Train
    trainer.fit(model, train_loader)

    save_dir = "saved_models"
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    tokenizer.save_pretrained(save_dir)

    return (model, tokenizer)


def predict(model, tokenizer, sentence, device):
    """Predicts a single class for a single sentence."""

    model.eval()

    inputs = {
        k: v.to(device)
        for k, v in tokenizer(
            sentence, return_tensors="pt", truncation=True, padding=True, max_length=64
        ).items()
    }

    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])

    return torch.argmax(logits, dim=1).item()


def evaluate(model, test_df, tokenizer):
    """Loops through a test dataframe, saves predictions and prints the
    final F1 score."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    true_labels = test_df["label"].tolist()

    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"bert_predictions_{current_date}.txt"

    predictions = []
    with open(f"predictions/{file_name}", "w") as f:
        f.write("pmcid\tlabel")

        for _, row in test_df.iterrows():
            pred = predict(model, tokenizer, str(row["text"]), device)
            predictions.append(pred)

            f.write(f"\n{row['pmcid']}\t{pred}")

    f1 = f1_score(true_labels, predictions, average="weighted")
    print(f"F1 Score: {f1:.4f}")

    return f1


def train_and_eval_on_train_set(train_data):
    train_df, test_df = train_test_split(train_data, test_size=0.2, random_state=42)

    model, tokenizer = train(train_df, target_col="label", num_classes=2, num_epochs=3)

    evaluate(model, test_df, tokenizer)


def train_and_eval():
    model, tokenizer = train(
        train_data, target_col="label", num_classes=2, num_epochs=3
    )

    evaluate(model, val_data, tokenizer)


def load_model_and_eval():
    model_name = "bert-base-uncased"
    num_classes = 2
    save_dir = "saved_models"

    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = BERTModel(model_name, num_classes=num_classes)
    model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")), strict=False)

    evaluate(model, val_data, tokenizer)


train_path = "data/train.tsv"
val_path = "data/val.tsv"

train_data = pd.read_csv(train_path, sep="\t")
val_data = pd.read_csv(val_path, sep="\t")
