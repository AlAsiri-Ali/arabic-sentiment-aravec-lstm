from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from preprocess_arabic import clean_arabic_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/arabic_tweets.csv"))
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    text_col = "text"
    label_col = "sentiment"

    texts = df[text_col].apply(clean_arabic_text)
    labels = LabelEncoder().fit_transform(df[label_col])
    y = to_categorical(labels)

    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    X = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=50, padding="post", truncating="post")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=labels)

    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=50),
        LSTM(64),
        Dropout(0.5),
        Dense(2, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=64, validation_split=0.2)
    print(model.evaluate(X_test, y_test, verbose=0))


if __name__ == "__main__":
    main()
