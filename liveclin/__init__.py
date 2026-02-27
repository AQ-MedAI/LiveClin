"""LiveClin - A live clinical benchmark evaluation framework."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalConfig:
    """All configuration for a single evaluation run."""

    model: str
    api_base: str
    api_key: str
    image_mode: str  # "url" | "local"
    dataset: str = "2025_H1"
    concurrency: int = 100
    max_retries: int = 5
    retry_delay: float = 1.0
    timeout: float = 120.0
    temperature: float = 0.0
    max_tokens: int = 16384
    output: Optional[str] = None
    resume: bool = False
    data_dir: str = "data"
    jsonl_path: Optional[str] = None
    image_root: Optional[str] = None
    verbose: bool = False

    @property
    def use_url(self) -> bool:
        return self.image_mode == "url"

    @property
    def output_path(self) -> str:
        if self.output:
            return self.output
        safe_model = self.model.replace("/", "_").replace(" ", "_")
        return f"results/{safe_model}_{self.dataset}.json"


__version__ = "1.0.0"
