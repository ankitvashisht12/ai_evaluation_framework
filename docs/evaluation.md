# Evaluation

Core evaluation pipeline for RAG systems using Langsmith SDK.

## Single Evaluation

Run a single evaluation with specific configuration:

```python
from rag_evaluation_framework import Evaluation

evaluator = Evaluation(
    langsmith_dataset_name="my-dataset",
    kb_data_path="./knowledge_base"
)

results = evaluator.run(
    chunker=my_chunker,           # Optional, see [Chunker](chunker.md)
    embedder=my_embedder,         # Optional, see [Embedder](embedder.md)
    vector_store=my_vector_store, # Optional, see [Vector Store](vector_store.md)
    k=5,
    reranker=my_reranker,         # Optional
)
```

### Process

1. Load knowledge base documents from `kb_data_path`
2. Chunk documents using provided chunker
3. Embed chunks using embedder and store in vector database
4. Fetch evaluation dataset from Langsmith
5. Run retrieval for each query in dataset
6. Calculate metrics (see [Metrics](metrics.md)): recall@k, precision@k, MRR@k
7. Return results with Langsmith trace URLs

### Return Value

```python
{
    "metrics": {"token_level_recall@5": 0.85, "token_level_precision@5": 0.72, ...},
    "langsmith_experiment_url": "https://smith.langchain.com/...",
    "raw_results": ExperimentResults  # Raw Langsmith object
}
```

## Hyperparameter Sweep

Evaluate multiple configurations automatically:

```python
from rag_evaluation_framework import Evaluation, SweepConfig
from rag_evaluation_framework.evaluation.chunker import RecursiveCharTextSplitter
from rag_evaluation_framework.evaluation.embedder.openai_embedder import OpenAIEmbedder

evaluator = Evaluation(
    langsmith_dataset_name="my-dataset",
    kb_data_path="./knowledge_base"
)

sweep_results = evaluator.sweep(
    sweep_config=SweepConfig(
        chunkers=[
            RecursiveCharTextSplitter(chunk_size=500, chunk_overlap=50),
            RecursiveCharTextSplitter(chunk_size=1000, chunk_overlap=100),
        ],
        embedders=[
            OpenAIEmbedder(model_name="text-embedding-3-small"),
            OpenAIEmbedder(model_name="text-embedding-3-large"),
        ],
        k_values=[5, 10, 20],
        rerankers=[None],  # None = no reranking
    )
)
```

### SweepConfig Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chunkers` | `List[Chunker]` | Framework default | Chunker instances to evaluate |
| `embedders` | `List[Embedder]` | Framework default | Embedder instances to evaluate |
| `k_values` | `List[int]` | `[5]` | Top-k retrieval values |
| `rerankers` | `List[Optional[Reranker]]` | `[None]` | Reranker instances (include `None` for no reranking) |
| `metrics` | `Dict[str, Metrics]` | All token-level metrics | Metrics to compute for every experiment |
| `max_concurrency` | `int` | `4` | Max concurrent queries within each Langsmith evaluation run |

### How It Works

1. Generates the Cartesian product of all provided parameters
2. Groups combinations by `(chunker, embedder)` pair — chunks and embeds the KB once per group to avoid redundant (and costly) embedding API calls
3. Runs each combination as a separate Langsmith experiment
4. Returns a list of result dicts, one per combination

### Cost Optimisation

The sweep is structured so that the expensive steps (chunking + embedding the full KB) happen once per unique `(chunker, embedder)` pair.  Variations in `k` and `reranker` only trigger the lightweight retrieval + evaluation step.

```
For each (chunker, embedder):     # chunk + embed once
    For each k:                   # lightweight: just retrieve + evaluate
        For each reranker:
            → run evaluation
```

### Partial Sweeps

Any parameter you omit uses the framework default:

```python
# Only vary k — uses default chunker, embedder, no reranker
sweep_results = evaluator.sweep(
    sweep_config=SweepConfig(k_values=[5, 10, 20])
)
```

### Return Value

A `List[Dict]` where each dict contains:

```python
{
    "config": {
        "chunker": "RecursiveCharTextSplitter(1000,100)",
        "embedder": "OpenAIEmbedder(text-embedding-3-small)",
        "k": 10,
        "reranker": None
    },
    "metrics": {"token_level_recall@10": 0.85, ...},
    "langsmith_experiment_url": "https://smith.langchain.com/...",
    "raw_results": ExperimentResults
}
```

### Visualising Sweep Results

Pass sweep results to `ComparisonGraph` for visual comparison. See [Visualization](visualization.md).

```python
from rag_evaluation_framework import ComparisonGraph

graph = ComparisonGraph(sweep_results)
graph.bar()           # grouped bar chart
graph.line(x="k")     # line chart varying k
graph.heatmap()       # colour-coded grid
```

## Components

- **[Chunker](chunker.md)** - Document chunking strategies
- **[Embedder](embedder.md)** - Embedding model integration
- **[Metrics](metrics.md)** - Evaluation metrics (recall, precision, MRR)
- **[Vector Store](vector_store.md)** - Vector database abstraction
- **[Visualization](visualization.md)** - Comparing sweep results graphically
