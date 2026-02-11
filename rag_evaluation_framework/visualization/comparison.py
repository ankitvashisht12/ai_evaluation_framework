import re
import logging
from typing import List, Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class ComparisonGraph:
    """Visualise results from a hyperparameter sweep.

    Accepts the list of result dicts returned by ``Evaluation.sweep()`` and
    provides three views for comparing experiments:

    - ``bar()``     – grouped bar chart (configs on x-axis, one bar per metric)
    - ``line(x=…)`` – line chart varying a single parameter on the x-axis
    - ``heatmap()`` – colour-coded grid (configs × metrics)

    Example::

        from rag_evaluation_framework.visualization import ComparisonGraph

        graph = ComparisonGraph(sweep_results)
        graph.bar()                          # overview
        graph.line(x="k")                    # effect of k
        graph.heatmap(metrics=["token_level_recall", "token_level_precision"])
    """

    def __init__(self, sweep_results: List[Dict[str, Any]]):
        if not sweep_results:
            raise ValueError("sweep_results must be a non-empty list")
        self.results = sweep_results

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_metric_name(name: str) -> str:
        """Strip the ``@k`` suffix so metrics are comparable across k values."""
        return re.sub(r"@\d+$", "", name)

    def _get_normalized_metrics(
        self, result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Return metrics dict with @k suffixes removed."""
        return {
            self._normalize_metric_name(k): v
            for k, v in result.get("metrics", {}).items()
        }

    def _all_metric_names(self, filter_metrics: Optional[List[str]] = None) -> List[str]:
        """Collect every unique (normalised) metric name across results."""
        names: set[str] = set()
        for r in self.results:
            names.update(self._get_normalized_metrics(r).keys())
        ordered = sorted(names)
        if filter_metrics:
            ordered = [m for m in ordered if m in filter_metrics]
        return ordered

    @staticmethod
    def _config_label(config: Dict[str, Any]) -> str:
        """Build a short, readable label from a config dict."""
        parts: list[str] = []
        if config.get("chunker"):
            parts.append(config["chunker"])
        if config.get("embedder"):
            parts.append(config["embedder"])
        if config.get("k") is not None:
            parts.append(f"k={config['k']}")
        if config.get("reranker"):
            parts.append(config["reranker"])
        return " | ".join(parts)

    # ------------------------------------------------------------------ #
    #  Bar chart                                                           #
    # ------------------------------------------------------------------ #

    def bar(self, metrics: Optional[List[str]] = None) -> None:
        """Grouped bar chart comparing all experiment configs.

        Args:
            metrics: Metric names to include. ``None`` means all metrics.
        """
        metric_names = self._all_metric_names(metrics)
        if not metric_names:
            logger.warning("No metrics to plot")
            return

        labels = [self._config_label(r["config"]) for r in self.results]
        n_configs = len(labels)
        n_metrics = len(metric_names)

        x = np.arange(n_configs)
        width = 0.8 / n_metrics

        fig, ax = plt.subplots(figsize=(max(8, n_configs * 1.5), 6))

        for i, metric in enumerate(metric_names):
            values = [
                self._get_normalized_metrics(r).get(metric, 0.0)
                for r in self.results
            ]
            offset = (i - n_metrics / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=metric)
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(
                        f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

        ax.set_ylabel("Score")
        ax.set_title("Experiment Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize="small")
        fig.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    #  Line chart                                                          #
    # ------------------------------------------------------------------ #

    def line(self, x: str, metrics: Optional[List[str]] = None) -> None:
        """Line chart varying one parameter on the x-axis.

        Best used with a numeric parameter like ``k``.

        Args:
            x: Config key to place on the x-axis (e.g. ``"k"``, ``"chunker"``).
            metrics: Metric names to include. ``None`` means all metrics.
        """
        valid_keys = {"k", "chunker", "embedder", "reranker"}
        if x not in valid_keys:
            raise ValueError(f"x must be one of {sorted(valid_keys)}")

        metric_names = self._all_metric_names(metrics)
        if not metric_names:
            logger.warning("No metrics to plot")
            return

        # Group results by all config keys *except* x
        other_keys = [k for k in ["chunker", "embedder", "k", "reranker"] if k != x]

        groups: Dict[tuple, Dict[str, Any]] = {}
        for r in self.results:
            cfg = r["config"]
            group_key = tuple(str(cfg.get(k)) for k in other_keys)
            group_label = " | ".join(
                f"{k}={cfg.get(k)}"
                for k in other_keys
                if cfg.get(k) is not None
            )
            if group_key not in groups:
                groups[group_key] = {"label": group_label, "results": []}
            groups[group_key]["results"].append(r)

        n_metrics = len(metric_names)
        fig, axes = plt.subplots(
            1, n_metrics, figsize=(6 * n_metrics, 5), squeeze=False
        )
        axes = axes[0]

        for ax, metric_name in zip(axes, metric_names):
            for group_data in groups.values():
                x_values = [r["config"][x] for r in group_data["results"]]
                y_values = [
                    self._get_normalized_metrics(r).get(metric_name, 0.0)
                    for r in group_data["results"]
                ]

                # Sort by x value
                pairs = sorted(zip(x_values, y_values), key=lambda p: (isinstance(p[0], str), p[0]))
                x_sorted = [p[0] for p in pairs]
                y_sorted = [p[1] for p in pairs]

                ax.plot(x_sorted, y_sorted, marker="o", label=group_data["label"])

            ax.set_xlabel(x)
            ax.set_ylabel("Score")
            ax.set_title(metric_name)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize="small")

        fig.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    #  Heatmap                                                             #
    # ------------------------------------------------------------------ #

    def heatmap(self, metrics: Optional[List[str]] = None) -> None:
        """Heatmap of configs (rows) vs metrics (columns).

        Args:
            metrics: Metric names to include. ``None`` means all metrics.
        """
        metric_names = self._all_metric_names(metrics)
        if not metric_names:
            logger.warning("No metrics to plot")
            return

        labels = [self._config_label(r["config"]) for r in self.results]
        data = np.array(
            [
                [
                    self._get_normalized_metrics(r).get(m, 0.0)
                    for m in metric_names
                ]
                for r in self.results
            ]
        )

        fig, ax = plt.subplots(
            figsize=(max(8, len(metric_names) * 2), max(4, len(labels) * 0.6))
        )
        im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(metric_names)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(metric_names, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=8)

        # Annotate cells with values
        for i in range(len(labels)):
            for j in range(len(metric_names)):
                val = data[i, j]
                color = "white" if val > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

        ax.set_title("Sweep Results Heatmap")
        fig.colorbar(im, ax=ax, label="Score")
        fig.tight_layout()
        plt.show()
