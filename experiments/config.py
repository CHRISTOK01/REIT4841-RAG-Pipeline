"""Load and assemble a sweep configuration from configs/*.yaml.

A sweep config ties together the KB-variant config (and its config_hash --
how index artifacts on disk are located/versioned), the generation settings
held fixed across the whole sweep, and which retrieval methods / KB variants
to cross in the grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from data.kb_variants import KBVariantConfig
from data.kb_variants import config_hash as compute_kb_config_hash
from data.schemas import ChunkingConfig


@dataclass(frozen=True)
class HybridConfig:
    k_per_retriever: int = 50


@dataclass(frozen=True)
class GenerationMetricsConfig:
    enabled: bool = True
    answer_relevance_num_questions: int = 3


@dataclass(frozen=True)
class RobustnessConfig:
    enabled: bool = True
    seed: int = 123
    retrieval_method: str = "dense"  # which sweep retrieval method to run perturbations against


@dataclass(frozen=True)
class GenerationSettings:
    model: str
    embedding_model: str
    prompt_template_id: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class SweepConfig:
    retrieval_methods: list[str]
    kb_variant_names: list[str]
    top_k: int
    hybrid: HybridConfig
    generation_metrics: GenerationMetricsConfig
    robustness: RobustnessConfig
    kb_variant_config: KBVariantConfig
    chunking_config: ChunkingConfig
    generation: GenerationSettings
    query_set_path: Path
    processed_data_dir: Path
    artifacts_root: Path
    embedding_cache_path: Path
    results_path: Path

    def __post_init__(self) -> None:
        valid_methods = {"lexical", "dense", "hybrid"}
        if not set(self.retrieval_methods) <= valid_methods:
            raise ValueError(f"Unknown retrieval method(s) in {self.retrieval_methods}; expected subset of {valid_methods}")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

    @property
    def kb_config_hash(self) -> str:
        return compute_kb_config_hash(self.kb_variant_config, self.chunking_config)

    def kb_variant_chunks_path(self, kb_variant: str) -> Path:
        return self.processed_data_dir / f"kb_{kb_variant}" / "chunks.jsonl"


def load_sweep_config(path: Path) -> SweepConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    kb_raw = yaml.safe_load(Path(raw["kb_variants_config"]).read_text(encoding="utf-8"))
    kb_variant_config = KBVariantConfig(**kb_raw["kb_variant"])
    chunking_config = ChunkingConfig(**kb_raw["chunking"])

    gen_raw = yaml.safe_load(Path(raw["generation_config"]).read_text(encoding="utf-8"))
    generation = GenerationSettings(**gen_raw)

    paths = raw["paths"]
    return SweepConfig(
        retrieval_methods=list(raw["retrieval_methods"]),
        kb_variant_names=list(raw["kb_variants"]),
        top_k=raw["top_k"],
        hybrid=HybridConfig(**raw.get("hybrid", {})),
        generation_metrics=GenerationMetricsConfig(**raw.get("generation_metrics", {})),
        robustness=RobustnessConfig(**raw.get("robustness", {})),
        kb_variant_config=kb_variant_config,
        chunking_config=chunking_config,
        generation=generation,
        query_set_path=Path(raw["query_set"]),
        processed_data_dir=Path(paths.get("processed_data_dir", "datasets/processed")),
        artifacts_root=Path(paths["artifacts_root"]),
        embedding_cache_path=Path(paths["embedding_cache"]),
        results_path=Path(paths["results"]),
    )
