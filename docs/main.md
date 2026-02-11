# RAG Evaluation Framework

A framework for evaluating RAG (Retrieval-Augmented Generation) pipelines using Langsmith SDK.

## Overview

The framework helps you systematically evaluate different RAG configurations by:
1. Loading and chunking knowledge base documents
2. Embedding documents into a vector store
3. Running retrieval evaluations against Langsmith datasets
4. Calculating metrics (recall, precision, MRR) across different configurations

## Quick Start

```python
from rag_evaluation_framework import Evaluation

evaluator = Evaluation(
    langsmith_dataset_name="my-dataset",
    kb_data_path="./knowledge_base"
)

results = evaluator.run(
    chunker=my_chunker,           # optional, uses default if None
    embedder=my_embedder,         # optional, uses OpenAI default if None
    vector_store=my_vector_store, # optional, uses Chroma default if None
    k=5,
    reranker=my_reranker,         # optional
)
```

## Components

- **[Evaluation](evaluation.md)** - Core evaluation pipeline, single runs and sweeps
- **[Visualization](visualization.md)** - Compare sweep results with bar, line and heatmap charts
- **[Synthetic Data Generation](synthetic_datagen.md)** - Coming soon
- **[Chunker](chunker.md)** - Document chunking strategies
- **[Embedder](embedder.md)** - Embedding model integration
- **[Vector Store](vector_store.md)** - Vector database abstraction

## Key Features

- **Flexible Components**: Use custom chunkers, embedders, and rerankers
- **Hyperparameter Sweeps**: Evaluate multiple configurations automatically
- **Langsmith Integration**: Seamless integration with Langsmith for tracking and visualization
- **Default Implementations**: Zero-config defaults for quick start

## Project Structure

```
rag_evaluation_framework/
├── evaluation/
│   ├── base_eval.py          # Main Evaluation class
│   ├── config.py             # Configuration & defaults
│   ├── metrics.py            # Recall, Precision, MRR calculators
│   ├── chunker/
│   │   ├── base.py           # Abstract base
│   │   └── default.py        # Default chunker implementation
│   ├── embedder/
│   │   ├── base.py           # Abstract base
│   │   └── default.py        # Default embedder
│   └── reranker/
│       ├── base.py           # Abstract base
│       └── default.py        # Default reranker (Cohere)
└── visualization/
    └── comparison.py         # ComparisonGraph class
```

