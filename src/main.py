import os
import pandas as pd
from fixed_token_chunker import FixedTokenChunker
from pipeline import run_evaluation
from sentence_transformers import SentenceTransformer


def create_embedder(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"):
    model = SentenceTransformer(model_name)

    def embed_text(text):
        return model.encode(str(text), convert_to_numpy=True)

    return embed_text


def main():
    os.makedirs("results", exist_ok=True)

    chunk_sizes = [200, 400, 600]
    chunk_overlaps = [0, 50, 100]
    retrieval_counts = [2, 5, 10]
    embedding_func = create_embedder()

    results_data = []

    for chunk_size in chunk_sizes:
        for chunk_overlap in chunk_overlaps:
            for retrieval_count in retrieval_counts:
                print(
                    f"\nRunning Evaluation with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, retrieval_count={retrieval_count}:"
                )
                chunker = FixedTokenChunker(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                results = run_evaluation(chunker, embedding_func, retrieval_count)
                results_data.append(
                    {
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "num_retrievals": retrieval_count,
                        "precision": f"{results['avg_precision']:.4f}",
                        "recall": f"{results['avg_recall']:.4f}",
                    }
                )

    results_df = pd.DataFrame(results_data)
    results_df.to_csv("results/summary.csv")


if __name__ == "__main__":
    main()
