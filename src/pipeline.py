import numpy as np
from metrics import calculate_metrics
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_corpus, load_questions


def run_evaluation(
    chunker,
    embedding_func,
    retrieval_count,
    corpus_path="data/state_of_the_union.md",
    questions_path="data/questions_df.csv",
):

    corpus = load_corpus(corpus_path)
    questions_df = load_questions(questions_path)

    chunks = chunker.split_text(corpus)
    chunk_objects = [{"text": chunk, "id": i} for i, chunk in enumerate(chunks)]
    chunk_embeddings = np.array([embedding_func(chunk) for chunk in chunks])

    results = {"precision": [], "recall": []}

    for _, row in questions_df.iterrows():
        query = row["question"]
        golden_references = row["references"]
        query_embedding = embedding_func(query)

        # Retrival by cosine similarity
        similarities = np.dot(chunk_embeddings, query_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_indices = np.argsort(similarities).flatten()[-retrieval_count:]
        retrieved_chunks = [chunk_objects[i] for i in top_indices]

        metrics = calculate_metrics(retrieved_chunks, golden_references, corpus)
        results["precision"].append(metrics["precision"])
        results["recall"].append(metrics["recall"])

    return {
        "avg_precision": np.mean(results["precision"]),
        "avg_recall": np.mean(results["recall"]),
    }
