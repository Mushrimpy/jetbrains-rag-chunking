import json
import pandas as pd
import numpy as np
from fixed_token_chunker import FixedTokenChunker
from metrics import calculate_metrics
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_corpus, load_questions


def run_evaluation(
    chunker,
    embedding_fn,
    num_retrievals,
    corpus_path="data/state_of_the_union.md",
    questions_path="data/questions_df.csv",
):

    corpus = load_corpus(corpus_path)
    questions_df = load_questions(questions_path)

    chunks = chunker.split_text(corpus)
    chunk_objects = [{"text": chunk, "id": i} for i, chunk in enumerate(chunks)]
    chunk_embeddings = np.array([embedding_fn(chunk) for chunk in chunks])

    results = {"precision": [], "recall": []}

    for _, row in questions_df.iterrows():
        query = row["question"]
        golden_references = row["references"]
        query_embedding = embedding_fn(query)

        # Retrival by cosine similarity
        similarities = cosine_similarity(
            chunk_embeddings, query_embedding.reshape(1, -1)
        )
        top_indices = np.argsort(similarities).flatten()[-num_retrievals:]
        retrieved_chunks = [chunk_objects[i] for i in top_indices]

        metrics = calculate_metrics(retrieved_chunks, golden_references, corpus)
        results["precision"].append(metrics["precision"])
        results["recall"].append(metrics["recall"])

    return {
        "avg_precision": np.mean(results["precision"]),
        "avg_recall": np.mean(results["recall"]),
    }
