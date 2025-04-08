from fixed_token_chunker import FixedTokenChunker
from pipeline import run_evaluation
from sentence_transformers import SentenceTransformer


def create_embedder(model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)

    def embed_text(text):
        return model.encode(str(text), convert_to_numpy=True)

    return embed_text


def main():

    print("\nEnter the following hyperparameters:")

    chunk_size = int(input("Chunk size: "))
    chunk_overlap = int(input("Chunk overlap: "))
    num_retrievals = int(input("Number of retrievals: "))

    print("\nRunning Evaluation:")
    embedding_fn = create_embedder()
    chunker = FixedTokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    results = run_evaluation(chunker, embedding_fn, num_retrievals)
    print(f"Average Precision: {results['avg_precision']:.4f}")
    print(f"Average Recall: {results['avg_recall']:.4f}")


if __name__ == "__main__":
    main()
