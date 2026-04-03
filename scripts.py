from actions import train, evaluate
import pandas as pd
from sklearn.model_selection import train_test_split

train_path = "data/train.tsv"
val_path = "data/val.tsv"

train_data = pd.read_csv(train_path, sep="\t")
val_data = pd.read_csv(val_path, sep="\t")

def train_and_eval_on_train_set():
    train_df, test_df = train_test_split(train_data, test_size=0.2, random_state=42)

    model, tokenizer = train(train_df, target_col="label", num_classes=2, num_epochs=3)

    evaluate(model, test_df, tokenizer)


