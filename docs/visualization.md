# Visualization

Compare sweep results graphically using `ComparisonGraph`.

## Quick Start

```python
from rag_evaluation_framework import ComparisonGraph

# sweep_results is the list returned by evaluator.sweep()
graph = ComparisonGraph(sweep_results)
graph.bar()
```

## Views

### Bar Chart

Grouped bar chart — every experiment config on the x-axis, one bar per metric.

```python
graph.bar()

# Show only specific metrics
graph.bar(metrics=["token_level_recall", "token_level_precision"])
```

Best for: getting a quick overview of all experiments side by side.

### Line Chart

Line chart varying a single parameter on the x-axis.  Each line represents a unique combination of the *other* parameters.

```python
graph.line(x="k")           # x-axis = k values (numeric)
graph.line(x="chunker")     # x-axis = chunker variants

# Filter metrics
graph.line(x="k", metrics=["token_level_recall"])
```

Supported `x` values: `"k"`, `"chunker"`, `"embedder"`, `"reranker"`.

Best for: understanding the effect of a single parameter (especially `k`, which is numeric).

### Heatmap

Colour-coded grid — configs as rows, metrics as columns, colour intensity = score.

```python
graph.heatmap()

# Filter metrics
graph.heatmap(metrics=["token_level_recall", "token_level_mrr"])
```

Best for: spotting the best config at a glance when you have many experiments.

## Metric Name Normalisation

Metric names returned by Langsmith include an `@k` suffix (e.g. `token_level_recall@10`). `ComparisonGraph` automatically strips this suffix so that metrics are comparable across different `k` values.

## Full Example

```python
from rag_evaluation_framework import Evaluation, SweepConfig, ComparisonGraph
from rag_evaluation_framework.evaluation.chunker import RecursiveCharTextSplitter

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
        k_values=[5, 10, 20],
    )
)

graph = ComparisonGraph(sweep_results)

# Overview
graph.bar()

# How does recall change with k?
graph.line(x="k", metrics=["token_level_recall"])

# Compact comparison
graph.heatmap()
```

## Related

- **[Evaluation](evaluation.md)** – `sweep()` method that produces the results
- **[Metrics](metrics.md)** – Available evaluation metrics
