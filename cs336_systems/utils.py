import json
import os
import statistics as stats
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields

import torch
import torch.cuda.nvtx as nvtx


@contextmanager
def nvtx_range(name: str, enabled: bool):
    if enabled and torch.cuda.is_available():
        nvtx.range_push(name)
        try:
            yield
        finally:
            nvtx.range_pop()
    else:
        yield


def stats_ms(times_ms):
    mean = stats.mean(times_ms)
    std = stats.pstdev(times_ms) if len(times_ms) > 1 else 0.0
    return mean, std


@dataclass
class BenchmarkRow:
    model_size: str
    batch_size: int
    context_length: int
    vocab_size: int
    amp: str
    device: str
    phase: str          # e.g. "forward", "backward", "optimizer"
    mean_ms: float
    std_ms: float
    tok_s: float
    num_warmups: int
    num_measures: int


class BenchmarkReporter:
    """Appends benchmark rows to a JSONL file and (optionally) a Markdown table.

    Both files are opened in append mode, so repeated runs (e.g. a sweep over
    model sizes) accumulate into one table. The Markdown title and header are
    only written when the file is new or empty.
    """

    def __init__(self, jsonl_path: str, md_path: str | None = None, title: str | None = None):
        self.jsonl_path = jsonl_path
        self.md_path = md_path
        self.title = title
        for path in (jsonl_path, md_path):
            if path:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def add(self, row: BenchmarkRow) -> None:
        record = asdict(row)
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.md_path:
            self._append_md(row, record)

    def _append_md(self, row: BenchmarkRow, record: dict) -> None:
        cols = [f.name for f in fields(row)]
        new_file = not os.path.exists(self.md_path) or os.path.getsize(self.md_path) == 0
        with open(self.md_path, "a") as f:
            if new_file:
                if self.title:
                    f.write(f"{self.title}\n\n")
                f.write("| " + " | ".join(cols) + " |\n")
                f.write("|" + "|".join("---" for _ in cols) + "|\n")
            f.write("| " + " | ".join(_fmt(record[c]) for c in cols) + " |\n")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 1e4 else f"{v:.1f}"
    return str(v)
