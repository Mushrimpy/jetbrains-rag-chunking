# JetBrains RAG Chunking

## Overview
This report presents a brief investigation into a chunking strategy for RAG systems. We implement the **FixedTokenChunker** algorithm and evaluated its performance using precision and recall metrics, whilst tuning chunk sizes and retrieval hyperparameters.

## Methodology
### Dataset
We used the State of the Union address (~10,000 tokens) as our corpus, paired with 76 reference questions with golden excerpts. 

###  Evaluation Pipeline
Our evaluation pipeline consists of: Loading and preprocessing the corpus and questions -> Applying the chunking algorithm -> Generating embeddings for chunks and queries -> Retrieving the most similar chunks for each query -> Calculating precision and recall metrics.

### Metrics
We implemente a range-based retrival evaluation approach that calculates:
- Precision: Proportion of retrieved content that overlaps with golden references
- Recall: Proportion of golden reference content that is found in retrieved chunks

Reference: https://research.trychroma.com/evaluating-chunking#metrics

## Getting Started

### Setting Up Your Environment

1. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Script**
   ```bash
   python src/main.py
   ```

## Analysis

The aggregated results table can be found in ```results/summary.csv```.

#### Effect of Number of Retrievals:

As ```num_retrievals``` increases from 2 → 5 → 10:
- Precision consistently decreases
- Recall consistently increases

This is expected: retrieving more chunks leads to a greater chance of finding relevant information but dilutes precision.


#### Effect of Chunk Size:

As ```chunk size``` increases 200 → 400 → 600:

- Precision generally decreases
- Recall varies without a clear pattern

This is likely because larger chunks contain more "irrelevant" text alongside the relevant content.


#### Effect of Chunk Overlap:

The impact of chunk overlap is less consistent, though we note that overlapping is particularly beneficial for smaller chunk sizes.

## Conclusion
Our investigation of the **FixedTokenChunker** chunking strategy showed that this evaluation effectively captures variations in retrieval performance caused by varying hyperparameters as desired.
