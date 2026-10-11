"""Load and assemble a sweep configuration from configs/*.yaml.

A sweep config ties together the KB-variant config (and its config_hash --
how index artifacts on disk are located/versioned), the generation settings
held fixed across the whole sweep, and which retrieval methods / KB variants
to cross in the grid.

`SweepConfig.run_fingerprint()` hashes every setting that can change a result
row. It is written into each row, and the runner refuses to append to a
results file whose rows carry a different fingerprint, so one file can never
silently mix two experimental setups.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from data.kb_variants import KBVariantConfig
from data.kb_variants import config_hash as compute_kb_config_hash
from data.schemas import ChunkingConfig
from evaluation.generation_metrics import METRICS_VERSION, judge_prompts_hash
from evaluation.robustness import PERTURBATIONS
from generation.prompts import CLOSED_BOOK_TEMPLATE_ID, PROMPT_TEMPLATE_ID, prompt_hash
from retrieval.hybrid import DEFAULT_RRF_K

RETRIEVAL_METHODS = ("lexical", "dense", "hybrid")
BASELINES = ("oracle", "random", "none")
NO_KB = "none"  # kb_variant value on rows of the no-retrieval baseline


@dataclass(frozen=True)
class HybridConfig:
    k_per_retriever: int = 50
    rrf_k: int = DEFAULT_RRF_K


@dataclass(frozen=True)
class GenerationMetricsConfig:
    enabled: bool = True
    answer_relevance_num_questions: int = 3
    reference_agreement: bool = True


@dataclass(frozen=True)
class RobustnessConfig:
    enabled: bool = True
    retrieval_method: str = "dense"  # which sweep retrieval method to run perturbations against
    perturbations: tuple[str, ...] = PERTURBATIONS

    def __post_init__(self) -> None:
        unknown = set(self.perturbations) - set(PERTURBATIONS)
        if unknown:
            raise ValueError(f"Unknown perturbation(s) {sorted(unknown)}; expected subset of {PERTURBATIONS}")


@dataclass(frozen=True)
class GenerationSettings:
    model: str
    embedding_model: str
    prompt_template_id: str
    temperature: float
    max_tokens: int
    judge_model: str | None = None  # None -> same as `model`
    max_attempts: int = 3

    def __post_init__(self) -> None:
        # The template text lives in generation/prompts.py; the config names which
        # one the experiment is supposed to be using. A mismatch is a setup error.
        if self.prompt_template_id != PROMPT_TEMPLATE_ID:
            raise ValueError(
                f"config asks for prompt template {self.prompt_template_id!r} but the code "
                f"implements {PROMPT_TEMPLATE_ID!r}"
            )

    @property
    def effective_judge_model(self) -> str:
        return self.judge_model or self.model


@dataclass(frozen=True)
class BudgetConfig:
    max_usd: float = 10.0
    ledger_path: Path = Path("artifacts/api_spend.json")


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
    baselines: list[str] = field(default_factory=list)
    seed: int = 123
    repeat: int = 0  # >0 for repeat runs of the same setup (run-to-run variance)
    max_workers: int = 1
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    def __post_init__(self) -> None:
        if not set(self.retrieval_methods) <= set(RETRIEVAL_METHODS):
            raise ValueError(
                f"Unknown retrieval method(s) in {self.retrieval_methods}; expected subset of {RETRIEVAL_METHODS}"
            )
        if not set(self.baselines) <= set(BASELINES):
            raise ValueError(f"Unknown baseline(s) in {self.baselines}; expected subset of {BASELINES}")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.robustness.enabled and self.robustness.retrieval_method not in self.retrieval_methods:
            raise ValueError(
                f"robustness.retrieval_method={self.robustness.retrieval_method!r} is not one of the "
                f"sweep's retrieval_methods {self.retrieval_methods}"
            )
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")

    @property
    def kb_config_hash(self) -> str:
        return compute_kb_config_hash(self.kb_variant_config, self.chunking_config)

    def kb_variant_chunks_path(self, kb_variant: str) -> Path:
        return self.processed_data_dir / f"kb_{kb_variant}" / "chunks.jsonl"

    def settings_for_log(self) -> dict:
        """Every setting that can change a result row, as plain JSON types."""
        return {
            "kb_config_hash": self.kb_config_hash,
            "top_k": self.top_k,
            "seed": self.seed,
            "repeat": self.repeat,
            "model": self.generation.model,
            "judge_model": self.generation.effective_judge_model,
            "embedding_model": self.generation.embedding_model,
            "temperature": self.generation.temperature,
            "max_tokens": self.generation.max_tokens,
            "max_attempts": self.generation.max_attempts,
            "prompt_template_id": self.generation.prompt_template_id,
            "prompt_hash": prompt_hash(self.generation.prompt_template_id),
            "closed_book_prompt_hash": prompt_hash(CLOSED_BOOK_TEMPLATE_ID),
            "hybrid_k_per_retriever": self.hybrid.k_per_retriever,
            "rrf_k": self.hybrid.rrf_k,
            "generation_metrics": asdict(self.generation_metrics),
            "metrics_version": METRICS_VERSION,
            "judge_prompts_hash": judge_prompts_hash(),
            "robustness_method": self.robustness.retrieval_method if self.robustness.enabled else None,
        }

    def run_fingerprint(self, query_set_sha256: str) -> str:
        payload = {**self.settings_for_log(), "query_set_sha256": query_set_sha256}
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def load_sweep_config(path: Path) -> SweepConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    kb_raw = yaml.safe_load(Path(raw["kb_variants_config"]).read_text(encoding="utf-8"))
    kb_variant_config = KBVariantConfig(**kb_raw["kb_variant"])
    chunking_config = ChunkingConfig(**kb_raw["chunking"])

    gen_raw = yaml.safe_load(Path(raw["generation_config"]).read_text(encoding="utf-8"))
    generation = GenerationSettings(**gen_raw)

    robustness_raw = dict(raw.get("robustness", {}))
    if "perturbations" in robustness_raw:
        robustness_raw["perturbations"] = tuple(robustness_raw["perturbations"])

    budget_raw = raw.get("budget", {})
    budget = BudgetConfig(
        max_usd=float(budget_raw.get("max_usd", BudgetConfig.max_usd)),
        ledger_path=Path(budget_raw.get("ledger", BudgetConfig.ledger_path)),
    )

    paths = raw["paths"]
    return SweepConfig(
        retrieval_methods=list(raw["retrieval_methods"]),
        kb_variant_names=list(raw["kb_variants"]),
        baselines=list(raw.get("baselines", [])),
        top_k=raw["top_k"],
        seed=raw.get("seed", 123),
        repeat=raw.get("repeat", 0),
        max_workers=raw.get("execution", {}).get("max_workers", 1),
        hybrid=HybridConfig(**raw.get("hybrid", {})),
        generation_metrics=GenerationMetricsConfig(**raw.get("generation_metrics", {})),
        robustness=RobustnessConfig(**robustness_raw),
        budget=budget,
        kb_variant_config=kb_variant_config,
        chunking_config=chunking_config,
        generation=generation,
        query_set_path=Path(raw["query_set"]),
        processed_data_dir=Path(paths.get("processed_data_dir", "datasets/processed")),
        artifacts_root=Path(paths["artifacts_root"]),
        embedding_cache_path=Path(paths["embedding_cache"]),
        results_path=Path(paths["results"]),
    )
