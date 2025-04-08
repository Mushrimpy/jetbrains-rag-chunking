import json
import pandas as pd


def load_corpus(corpus_path: str):
    with open(corpus_path, "r") as file:
        return file.read()


def load_questions(questions_path: str):
    df = pd.read_csv(questions_path)
    df["references"] = df["references"].apply(json.loads)
    return df
