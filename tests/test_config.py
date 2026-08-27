from pathlib import Path

import pytest

from experiments.config import load_sweep_config

KB_VARIANTS_YAML = """
kb_variant:
  seed: 42
  base_corpus_size: 100
  min_answer_words: 15
  mixed_noise_ratio: 0.3
  reduced_drop_topic_fraction: 0.35
chunking:
  chunk_size_words: 120
  chunk_overlap_words: 20
"""

GENERATION_YAML = """
model: gpt-4o-mini
embedding_model: text-embedding-3-small
prompt_template_id: wellness_summary_v1
temperature: 0.2
max_tokens: 800
"""

SWEEP_YAML_TEMPLATE = """
kb_variants_config: {kb_variants_path}
generation_config: {generation_path}
query_set: {query_set_path}

retrieval_methods: [lexical, dense, hybrid]
kb_variants: [curated, mixed, reduced]
top_k: 5

hybrid:
  k_per_retriever: 50

generation_metrics:
  enabled: true
  answer_relevance_num_questions: 3

robustness:
  enabled: true
  seed: 123
  retrieval_method: dense

paths:
  processed_data_dir: {processed_dir}
  artifacts_root: {artifacts_dir}
  embedding_cache: {cache_path}
  results: {results_path}
"""


def write_sweep_configs(tmp_path: Path) -> Path:
    kb_path = tmp_path / "kb_variants.yaml"
    kb_path.write_text(KB_VARIANTS_YAML, encoding="utf-8")
    gen_path = tmp_path / "generation.yaml"
    gen_path.write_text(GENERATION_YAML, encoding="utf-8")

    sweep_path = tmp_path / "full_sweep.yaml"
    sweep_path.write_text(
        SWEEP_YAML_TEMPLATE.format(
            kb_variants_path=kb_path.as_posix(),
            generation_path=gen_path.as_posix(),
            query_set_path=(tmp_path / "query_set.jsonl").as_posix(),
            processed_dir=(tmp_path / "processed").as_posix(),
            artifacts_dir=(tmp_path / "artifacts").as_posix(),
            cache_path=(tmp_path / "cache.sqlite").as_posix(),
            results_path=(tmp_path / "results.jsonl").as_posix(),
        ),
        encoding="utf-8",
    )
    return sweep_path


def test_load_sweep_config_assembles_nested_configs(tmp_path):
    sweep_path = write_sweep_configs(tmp_path)
    config = load_sweep_config(sweep_path)

    assert config.retrieval_methods == ["lexical", "dense", "hybrid"]
    assert config.kb_variant_names == ["curated", "mixed", "reduced"]
    assert config.top_k == 5
    assert config.hybrid.k_per_retriever == 50
    assert config.generation_metrics.enabled is True
    assert config.robustness.retrieval_method == "dense"
    assert config.kb_variant_config.base_corpus_size == 100
    assert config.chunking_config.chunk_size_words == 120
    assert config.generation.model == "gpt-4o-mini"
    assert config.generation.embedding_model == "text-embedding-3-small"


def test_kb_config_hash_is_deterministic(tmp_path):
    sweep_path = write_sweep_configs(tmp_path)
    config1 = load_sweep_config(sweep_path)
    config2 = load_sweep_config(sweep_path)
    assert config1.kb_config_hash == config2.kb_config_hash
    assert len(config1.kb_config_hash) == 12


def test_kb_variant_chunks_path_builds_expected_layout(tmp_path):
    sweep_path = write_sweep_configs(tmp_path)
    config = load_sweep_config(sweep_path)
    path = config.kb_variant_chunks_path("curated")
    assert path == config.processed_data_dir / "kb_curated" / "chunks.jsonl"


def test_rejects_unknown_retrieval_method(tmp_path):
    sweep_path = write_sweep_configs(tmp_path)
    text = sweep_path.read_text(encoding="utf-8").replace(
        "retrieval_methods: [lexical, dense, hybrid]", "retrieval_methods: [lexical, made_up_method]"
    )
    sweep_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown retrieval method"):
        load_sweep_config(sweep_path)


def test_rejects_non_positive_top_k(tmp_path):
    sweep_path = write_sweep_configs(tmp_path)
    text = sweep_path.read_text(encoding="utf-8").replace("top_k: 5", "top_k: 0")
    sweep_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="top_k must be positive"):
        load_sweep_config(sweep_path)


def test_real_repo_full_sweep_config_loads():
    """Catches schema drift between configs/full_sweep.yaml and experiments/config.py."""
    config = load_sweep_config(Path("configs/full_sweep.yaml"))
    assert config.top_k > 0
    assert set(config.retrieval_methods) <= {"lexical", "dense", "hybrid"}
    assert set(config.kb_variant_names) <= {"curated", "mixed", "reduced"}
