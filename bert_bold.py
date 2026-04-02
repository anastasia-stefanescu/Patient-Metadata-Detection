import pandas as pd
import csv
import os
import shutil
import multiprocessing
from scipy.stats import pearsonr

import logging
import torch

import numpy as np

import torch.nn as nn
import pytorch_lightning as pl

from transformers import BertTokenizer

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.ao.quantization.qconfig import float_qparams_weight_only_qconfig
from transformers import AutoTokenizer, AutoModel, AutoConfig


MAX_SEQ_LEN = 256  # Adjust the maximum length of the sequence to your needs
BATCH_SIZE = 16  # Adjust the batch size to your requirements

#################################

import warnings

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure reproducibility
seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)

##########################################


def scalare(x, minimum, maximum):
    return (x - minimum) / (maximum - minimum)


class CustomDataset(Dataset):
    def __init__(self, df):
        self.samples = [
            {"content": row["sentence"], "class": scalare(row[col], minimum, maximum)}
            for _, row in df.iterrows()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class CustomCollator:
    """
    A custom data collator that prepares batches of single sentences for BERT model training or evaluation.
    It handles tokenization and ensures that sequences are padded to a uniform length.
    """

    def __init__(self, tokenizer: AutoTokenizer, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def __call__(self, input_batch: list[dict]) -> dict:
        """
        Processes a batch of input data, tokenizing the sentences and padding them to the same length.
        Args: input_batch: A list of dictionaries, where each dictionary contains 'content' (sentence) and 'class' (regression target).
        Returns: dict: A dictionary with tokenized and padded sentences, and associated regression targets.
        """
        sentences = [instance["content"] for instance in input_batch]
        targets = [instance["class"] for instance in input_batch]

        # Tokenize the sentences with padding and truncation
        tokenized_batch = self.tokenizer(
            sentences,
            padding=True,  # Pad to the longest sentence in the batch
            max_length=self.max_seq_len,  # Truncate if the sentence exceeds max length
            truncation=True,
            return_tensors="pt",  # Return the tokenized inputs as PyTorch tensors
        )

        # Convert the regression targets to a tensor
        targets_tensor = torch.tensor(targets, dtype=torch.float)

        return {
            "input_ids": tokenized_batch["input_ids"],
            "attention_mask": tokenized_batch["attention_mask"],
            "targets": targets_tensor,
        }


class BERTModel(pl.LightningModule):
    def __init__(
        self, model_name: str, lr: float = 2e-5, sequence_max_length: int = MAX_SEQ_LEN
    ):
        super().__init__()

        self.val_preds = []
        self.val_labels = []

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load the model configuration and set output_hidden_states=True
        self.model_config = AutoConfig.from_pretrained(
            model_name, output_hidden_states=True
        )

        self.model = AutoModel.from_pretrained(model_name, config=self.model_config)
        self.output_layer = nn.Linear(
            self.model.config.hidden_size, 1
        )  # One output unit for regression
        self.loss_fct = nn.MSELoss()
        self.lr = lr
        self.save_hyperparameters()

        if use_static_quantized_model == 1:
            #     # Set up quantization configuration
            #     self.qconfig = torch.quantization.get_default_qconfig('qnnpack')
            #     # Apply a specific quantization configuration for the embedding layers
            #     self.model.embeddings.qconfig = float_qparams_weight_only_qconfig
            #     # Insert quantization stubs in the model
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
            # # Prepare the model for quantization
            # torch.quantization.prepare(self, inplace=True)

        if using_only_layer5 == 1:
            # Fixed weights - pt toate
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.encoder.layer[4].parameters():
                param.requires_grad = True

            # Ensure the regression output layer is trainable
            for param in self.output_layer.parameters():
                param.requires_grad = True

        # self.gpu_available = torch.cuda.is_available()
        # if self.gpu_available:
        #     print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        # else:
        #     print("GPU is not available, using CPU.")

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        :param input_ids: Tensor of input IDs.
        :param attention_mask: Tensor of attention masks.
        :return: Predictions for the input batch.
        """
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        if using_only_layer5 == 1:
            # Extract the hidden states from the 5th transformer block
            hidden_states = output.hidden_states
            layer_5_output = hidden_states[4]  # 5th block's output

            if use_static_quantized_model == 1:
                # Quantize the layer_5_output before using it for regression
                quantized_layer_5_output = self.quant(layer_5_output)

                # Use the [CLS] token's representation from the 5th block for regression
                cls_embedding = quantized_layer_5_output[
                    :, 0, :
                ]  # [CLS] token embedding
            else:
                # Use the [CLS] token's representation from the 5th block for regression
                cls_embedding = layer_5_output[:, 0, :]  # [CLS] token embedding

            # Pass through the output layer (regression head)
            prediction = self.output_layer(cls_embedding).flatten()

            return prediction

        else:
            if use_static_quantized_model == 1:
                # Quantize the pooler output before applying the output layer
                quantized_pooler_output = self.quant(output.pooler_output)

                # Apply the output layer to the quantized pooled output
                prediction = self.output_layer(quantized_pooler_output).flatten()
            else:
                # Apply the output layer to the pooled output of the model
                prediction = self.output_layer(output.pooler_output).flatten()

            return prediction

    def prepare_custom_quantization(self):
        # Set the default qconfig for the entire model
        self.qconfig = torch.quantization.get_default_qconfig("qnnpack")

        # Manually set qconfig=None for normalization layers to skip quantization
        for i in range(len(self.model.encoder.layer)):
            self.model.encoder.layer[i].attention.output.LayerNorm.qconfig = None
            self.model.encoder.layer[i].output.LayerNorm.qconfig = None

        # Set the correct quantization configuration for embeddings
        self.model.embeddings.qconfig = (
            torch.quantization.float_qparams_weight_only_qconfig
        )

        # Now prepare the model for quantization
        torch.quantization.prepare(self.model, inplace=True)

    def convert_custom_quantization(self):
        # Perform the conversion to quantized model
        torch.quantization.convert(self.model, inplace=True)

    def training_step(self, batch, batch_idx):
        """
        :param batch: The batch of data.
        :param batch_idx: Index of the batch.
        :return: A dictionary containing the loss.
        """
        # Prepare the tokenized batch
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        ground_truth = batch["targets"].float()

        # Forward pass
        prediction = self(input_ids=input_ids, attention_mask=attention_mask)

        loss = self.loss_fct(prediction, ground_truth)
        mae = torch.mean(torch.abs(prediction - ground_truth))

        self.log(
            "train_loss",
            loss.detach().cpu().item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train_mae", mae, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """
        :param batch: The batch of data.
        :param batch_idx: Index of the batch.
        :return: A dictionary containing the loss.
        """
        # Prepare the tokenized batch
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        ground_truth = batch["targets"].float()

        # Forward pass
        prediction = self(input_ids=input_ids, attention_mask=attention_mask)

        loss = self.loss_fct(prediction, ground_truth)
        mae = torch.mean(torch.abs(prediction - ground_truth))

        self.log(
            "val_loss",
            loss.detach().cpu().item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=True)

        # Save predictions and ground truth for later correlation calculation
        self.val_preds.extend(prediction.detach().cpu().numpy())
        self.val_labels.extend(ground_truth.detach().cpu().numpy())

        return {"loss": loss}

    def on_validation_epoch_end(self):
        """
        This function is called at the end of the validation epoch.
        It calculates the Pearson correlation between predictions and ground truth.
        """
        # Convert stored lists of predictions and labels to numpy arrays
        val_preds = np.array(self.val_preds)
        val_labels = np.array(self.val_labels)

        # Calculate Pearson correlation
        if len(val_preds) > 0 and len(val_labels) > 0:
            pearson_correlation, _ = pearsonr(val_preds, val_labels)
        else:
            pearson_correlation = 0.0

        # Log the Pearson correlation
        self.log("val_pearson_corr", pearson_correlation, prog_bar=True)

        # Clear the stored predictions and labels for the next epoch
        self.val_preds = []
        self.val_labels = []

    def quantize_model(self):  # dynamically!
        # Apply dynamic quantization to the entire model
        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )

        # Also quantize the output layer (if necessary)
        self.output_layer = torch.quantization.quantize_dynamic(
            self.output_layer, {torch.nn.Linear}, dtype=torch.qint8
        )

        # Return quantized model for any further use
        return self

    def configure_optimizers(self):
        return AdamW(
            [p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08
        )


def predict(model, tokenizer, sentence):
    # Tokenize the single sentence input
    tokenized_input = tokenizer(
        sentence,
        padding=True,
        max_length=10,  # You can reduce the max length for a single sentence
        truncation=True,
        return_tensors="pt",
    )

    # Move tokenized input to the same device as the model
    tokenized_input = {
        key: value.to(next(model.parameters()).device)
        for key, value in tokenized_input.items()
    }

    with torch.no_grad():  # Turn off gradient computation for inference
        predictions = model(
            input_ids=tokenized_input["input_ids"],
            attention_mask=tokenized_input["attention_mask"],
        )

    prediction_score = predictions.mean().item()  # Handle batch dimension if needed

    return prediction_score


def evaluate_model_on_tests(model, lista, output_file, column):
    model.eval()  # Set the model to evaluation mode.

    device = next(
        model.parameters()
    ).device  # Get the device of the model's parameters.

    with (
        torch.no_grad(),
        open(output_file, "a", newline="") as csvfile,
    ):  # Inference mode, no gradients needed.
        lung = len(lista)

        wrt = csv.writer(csvfile)
        # wrt.writerow(['column', 'correlation'])
        abs_mean_error = 0
        preds = []
        reals = []
        for i in range(lung):
            sentence = lista[i][1]
            # sent_id = lista[i][0]
            actual_value = float(lista[i][2])

            rez = predict(model, tokenizer, sentence)
            original_value = rez * (maximum - minimum) + minimum
            preds.append(original_value)
            reals.append(actual_value)

            # rez2 = np.array([rez])
            # scaled_prediction = rez2.reshape(-1, 1)
            # original_range_prediction = scaler.inverse_transform(scaled_prediction)
            # original_value = original_range_prediction.flatten()[0]

            # error = (original_value - actual_value)
            # #mean_error += error
            abs_mean_error += abs(original_value - actual_value)
            # med_pred += original_value
            # med_real += actual_value
            # scalare
            # wrt.writerow([sent_id, sentence, original_value,actual_value])
        val_preds = np.array(preds)
        val_labels = np.array(reals)
        (pearson_corr, _) = pearsonr(val_preds, val_labels)
        wrt.writerow([column, pearson_corr])
        print(f"{column} -> corr: {pearson_corr}")


def train(train_dataloader, validation_dataloader):
    MODEL_PATH = "bert-base-uncased"  # or your custom model path
    model = BERTModel(model_name=MODEL_PATH)

    trainer = pl.Trainer(
        devices=-1,  # Comment this when training on cpu
        accelerator="gpu",
        max_epochs=3,  # Set this to -1 when training fully
        limit_train_batches=10,  # Uncomment this when training fully
        limit_val_batches=5,  # Uncomment this when training fully
        gradient_clip_val=1.0,
        enable_checkpointing=False,
        # aici am adaugat
        log_every_n_steps=50,
    )

    # Training
    trainer.fit(model, train_dataloader, validation_dataloader)

    pth = inc + "lightning_logs/"
    if os.path.exists(pth):
        shutil.rmtree(pth)  # Recursively delete the log directory
        print(f"Deleted log directory: {pth}")

    return (model, trainer)


def get_results(model, testdf, output_file, column):

    data_test = list(testdf.to_records(index=False))
    data_test_str = [(str(elem[0]), str(elem[1]), str(elem[2])) for elem in data_test]

    evaluate_model_on_tests(model, data_test_str, output_file, column)


def prepare(df, y, column):
    train_df, val_df, y_train, y_val = train_test_split(
        df, y, test_size=0.3, random_state=42
    )
    val_df, test_df, y_val, y_test = train_test_split(
        val_df, y_val, test_size=0.5, random_state=42
    )

    train_dataset = CustomDataset(train_df)
    val_dataset = CustomDataset(val_df)
    # test_dataset = CustomDataset(test_df)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collator,
        num_workers=2,
        pin_memory=True,
    )
    validation_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collator,
        num_workers=2,
        pin_memory=True,
    )
    # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collator, num_workers = 2)

    (model, _) = train(train_dataloader, validation_dataloader)
    get_results(model, test_df, "rez_testing_3ep.csv", column)


def quantize_model(model):
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,  # Quantizing only Linear layers
    )
    return quantized_model


def calibrate_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            model(input_ids=input_ids, attention_mask=attention_mask)


if __name__ == "__main__":
    multiprocessing.set_start_method(
        "spawn", force=True
    )  # Ensures safe multiprocessing on MacOS

    # Enable the quantization backend for ARM
    torch.backends.quantized.engine = "qnnpack"

    # torch.device("mps") # for Unsloth

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    scaler = MinMaxScaler()

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    custom_collator = CustomCollator(tokenizer, MAX_SEQ_LEN)

    #################################
    inc = "/Users/anastasiastefanescu/Documents/dataseturi eyetracking/"

    columns = [
        "lang_LH_AntTemp",
        "lang_LH_IFG",
        "lang_LH_IFGorb",
        "lang_LH_MFG",
        "lang_LH_PostTemp",
        "lang_LH_netw",
    ]

    using_train_test_split = 1
    using_only_layer5 = 0
    testing_eyetracking = 1
    use_dynamically_quantized_model = 0
    use_static_quantized_model = 1

    if testing_eyetracking == 0:
        if using_train_test_split == 1:  ############train - test - split
            for col in columns:
                df = pd.read_csv(inc + "bold_response_LH.csv")
                maximum = df[col].max()
                minimum = df[col].min()
                y = df[col]
                prepare(df, y, col)
        else:
            for col in columns:
                df = pd.read_csv(inc + "bold_response_LH.csv")
                maximum = df[col].max()
                minimum = df[col].min()
                y = df[col]

                val_losses = []
                val_maes = []
                correlations = []
                output_file = "5fold_correlations.csv"

                with open(output_file, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Fold", "Validation MAE", "Validation Corr"])
                    writer.writerow([col])

                    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
                        print(f" ------- Fold {fold + 1}")

                        df_train = df.iloc[train_idx]
                        df_val = df.iloc[val_idx]

                        train_dataset = CustomDataset(df_train)
                        val_dataset = CustomDataset(df_val)

                        train_dataloader = DataLoader(
                            train_dataset,
                            batch_size=16,
                            shuffle=True,
                            collate_fn=custom_collator,
                        )
                        validation_dataloader = DataLoader(
                            val_dataset,
                            batch_size=16,
                            shuffle=False,
                            collate_fn=custom_collator,
                        )

                        (model, trainer) = train(
                            train_dataloader, validation_dataloader
                        )

                        val_metrics = trainer.validate(model, validation_dataloader)
                        val_loss = val_metrics[0]["val_loss"]
                        val_mae = val_metrics[0]["val_mae"]

                        pearson_correlation = trainer.callback_metrics[
                            "val_pearson_corr"
                        ].item()

                        val_losses.append(val_loss)
                        val_maes.append(val_mae)
                        correlations.append(pearson_correlation)
                        print(f"Validation MAE for Fold {fold + 1}: {val_mae:.4f}")
                        print(
                            f"Validation Correlation for Fold {fold + 1}: {pearson_correlation:.4f}"
                        )
                        writer.writerow([fold + 1, val_mae, pearson_correlation])

                    # After all folds
                    mean_val_loss = np.mean(val_losses)
                    mean_val_mae = np.mean(val_maes)
                    mean_pearson_correlation = np.mean(correlations)

                    print(f"Average Validation MAE: {mean_val_mae:.4f}")
                    print(f"Average Validation Loss: {mean_val_loss:.4f}")
                    print(f"Average correlations: {mean_pearson_correlation:.4f}")
                    writer.writerow(["Average", mean_val_mae, mean_pearson_correlation])

    else:
        for col in columns:
            df = pd.read_csv(inc + "bold_response_LH.csv")
            maximum = df[col].max()
            minimum = df[col].min()
            y = df[col]

            train_df, val_df, y_train, y_val = train_test_split(
                df, y, test_size=0.2, random_state=42
            )

            train_dataset = CustomDataset(train_df)
            val_dataset = CustomDataset(val_df)

            train_dataloader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                collate_fn=custom_collator,
                num_workers=2,
                pin_memory=True,
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=custom_collator,
                num_workers=2,
            )

            MODEL_PATH = "bert-base-uncased"  # or your custom model path
            model = BERTModel(model_name=MODEL_PATH)

            if use_dynamically_quantized_model == 1:
                # Quantize the model after training
                quantized_model = model.quantize_model()

                # Now you can save or use the quantized model for efficient inference
                torch.save(quantized_model.state_dict(), "quantized_model.pt")
            else:
                if use_static_quantized_model == 1:
                    model.prepare_custom_quantization()
                    calibrate_model(model, val_dataloader)
                    # quantize the model
                    model.convert_custom_quantization()

            trainer = pl.Trainer(
                devices=-1,  # Comment this when training on cpu
                accelerator="gpu",
                max_epochs=1,  # Set this to -1 when training fully
                limit_train_batches=6,  # Uncomment this when training fully
                limit_val_batches=5,  # Uncomment this when training fully
                gradient_clip_val=1.0,
                enable_checkpointing=False,
                # aici am adaugat
                log_every_n_steps=50,
            )

            # Training
            if use_dynamically_quantized_model == 1:
                trainer.fit(quantized_model, train_dataloader, val_dataloader)
            else:
                trainer.fit(model, train_dataloader, val_dataloader)

            pth = inc + "lightning_logs/"
            if os.path.exists(pth):
                shutil.rmtree(pth)  # Recursively delete the log directory
                print(f"Deleted log directory: {pth}")

            df_test = pd.read_csv(inc + "datasets/zuco/train_sent.csv")

            if use_dynamically_quantized_model == 1:
                get_results(quantized_model, df_test, "quant_bold_6ep.csv", col)
            else:
                if use_static_quantized_model == 1:
                    model.convert_custom_quantization()
                    get_results(model, df_test, "static_quant_zuco_bold_1ep.csv", col)
                else:
                    get_results(model, df_test, "corr_zuco_bold_10ep.csv", col)
