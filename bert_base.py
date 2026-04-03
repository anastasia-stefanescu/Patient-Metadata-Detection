import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel


class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return {
            "pmcid": row["pmcid"],
            "content": str(row["text"]),
            "class": row["label"],
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
    def __init__(self, model_name: str, num_classes: int, lr: float = 2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModel.from_pretrained(model_name)
        self.output_layer = nn.Linear(self.model.config.hidden_size, num_classes)
        self.loss_fct = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0, :]
        return self.output_layer(cls_output)

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fct(logits, batch["targets"].long())
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch["targets"]).float().mean()
        # We don't log to file, just return for the trainer to track in memory
        return {
            "val_loss": self.loss_fct(logits, batch["targets"].long()),
            "val_acc": acc,
        }

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)
