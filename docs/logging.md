# Logging

The RAG Evaluation Framework uses Python's standard `logging` module with sensible defaults for library usage.

## Default Behavior

**By default, no logs are produced.** This follows Python's best practices for libraries — users opt-in to logging when they need it.

## Enabling Logging

### Quick Start

```python
import logging

# Enable INFO level logs (recommended for most users)
logging.basicConfig(level=logging.INFO)

from rag_evaluation_framework import Evaluation
# Now you'll see high-level progress messages
```

### Detailed Debug Logs

```python
import logging

# Enable DEBUG level for detailed output
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from rag_evaluation_framework import Evaluation
# Now you'll see all debug information
```

### Package-Specific Logging

If you only want logs from this package (not other libraries):

```python
import logging

# Configure only rag_evaluation_framework logger
logging.getLogger("rag_evaluation_framework").setLevel(logging.DEBUG)

# Add a handler to see the output
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logging.getLogger("rag_evaluation_framework").addHandler(handler)
```

## Log Levels

| Level | What You'll See |
|-------|-----------------|
| `DEBUG` | Everything: file processing, chunk counts, retrieval details |
| `INFO` | Progress milestones: start/end of runs, KB indexing stats |
| `WARNING` | Potential issues: empty queries, extraction fallbacks |
| `ERROR` | Failures: missing files, invalid configurations |

## Example Output

### INFO Level

```
2025-01-14 10:30:15 - rag_evaluation_framework.evaluation.base_eval - INFO - Starting evaluation run with k=5
2025-01-14 10:30:15 - rag_evaluation_framework.evaluation.base_eval - INFO - Processing knowledge base...
2025-01-14 10:30:16 - rag_evaluation_framework.evaluation.base_eval - INFO - Knowledge base indexed: 150 total chunks from 3 files
2025-01-14 10:30:16 - rag_evaluation_framework.evaluation.base_eval - INFO - Running LangSmith evaluation on dataset 'my-dataset' with 1 evaluators
2025-01-14 10:30:20 - rag_evaluation_framework.evaluation.base_eval - INFO - Evaluation complete. Metrics: {'chunk_level_recall@5': 0.85}
```

### DEBUG Level

```
2025-01-14 10:30:15 - rag_evaluation_framework.evaluation.base_eval - DEBUG - Initialized Evaluation with dataset='my-dataset', kb_path='./kb', query_field='question'
2025-01-14 10:30:15 - rag_evaluation_framework.evaluation.base_eval - INFO - Starting evaluation run with k=5
2025-01-14 10:30:15 - rag_evaluation_framework.evaluation.base_eval - DEBUG - Using default chunker: RecursiveCharTextSplitter
2025-01-14 10:30:15 - rag_evaluation_framework.evaluation.base_eval - DEBUG - Using default embedder: OpenAIEmbedder
2025-01-14 10:30:15 - rag_evaluation_framework.evaluation.base_eval - DEBUG - Using default vector store: ChromaVectorStore
2025-01-14 10:30:15 - rag_evaluation_framework.evaluation.vector_store.chroma - DEBUG - Initializing ChromaDB in-memory client
2025-01-14 10:30:15 - rag_evaluation_framework.evaluation.vector_store.chroma - DEBUG - Using collection 'default' with cosine similarity
2025-01-14 10:30:15 - rag_evaluation_framework.evaluation.base_eval - INFO - Processing knowledge base...
2025-01-14 10:30:15 - rag_evaluation_framework.evaluation.base_eval - DEBUG - Found 3 markdown files in knowledge base
2025-01-14 10:30:15 - rag_evaluation_framework.evaluation.base_eval - DEBUG - Processing file: doc1.md
2025-01-14 10:30:15 - rag_evaluation_framework.evaluation.base_eval - DEBUG - Created 50 chunks from doc1.md
2025-01-14 10:30:15 - rag_evaluation_framework.evaluation.vector_store.chroma - DEBUG - Added 50 documents to collection 'default'
...
```

## Using with Jupyter Notebooks

```python
import logging
import sys

# Set up logging for notebooks
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    stream=sys.stdout
)

from rag_evaluation_framework import Evaluation

evaluator = Evaluation(
    langsmith_dataset_name="my-dataset",
    kb_data_path="./knowledge_base"
)

results = evaluator.run(k=5)
```

## Disabling Logs

If you've enabled logging but want to silence this package:

```python
logging.getLogger("rag_evaluation_framework").setLevel(logging.CRITICAL)
```

## Writing Logs to a File

```python
import logging

# Log to file
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="evaluation.log",
    filemode="w"  # 'w' to overwrite, 'a' to append
)
```

## Integration with Other Logging

The framework's logging integrates seamlessly with your application's logging configuration. If you've already set up logging for your application, the framework will use those settings automatically.

```python
import logging

# Your application's logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Framework logs will follow your configuration
from rag_evaluation_framework import Evaluation
```
