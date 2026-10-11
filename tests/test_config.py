from pathlib import Path

import pytest

from experiments.config import GenerationSettings, RobustnessConfig, load_sweep_config
from generation.prompts import prompt_hash

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
model: gpt-4o-mini-2024-07-18
judge_model: some-judge-model
embedding_model: text-embedding-3-small
prompt_template_id: wellness_summary_v1
temperature: 0.2
max_tokens: 800
max_attempts: 4
"""

SWEEP_YAML_TEMPLATE = """
kb_variants_config: {kb_variants_path}
generation_config: {generation_path}
query_set: {query_set_path}

retrieval_methods: [lexical, dense, hybrid]
kb_variants: [curated, mixed, reduced]
baselines: [oracle, random, none]
top_k: 5
seed: 77

hybrid:
  k_per_retriever: 40
  rrf_k: 30

generation_metrics:
  enabled: true
  answer_relevance_num_questions: 4
  reference_agreement: false

robustness:
  enabled: true
  retrieval_method: hybrid
  perturbations: [drop_top_relevant, control_rerun]

execution:
  max_workers: 6

budget:
  max_usd: 2.5
  ledger: {ledger_path}

paths:
  processed_data_dir: {processed_dir}
  artifacts_root: {artifacts_dir}
  embedding_cache: {cache_path}
  results: {results_path}
"""


def write_sweep_configs(tmp_path: Path, generation_yaml: str = GENERATION_YAML) -> Path:
    kb_path = tmp_path / "kb_variants.yaml"
    kb_path.write_text(KB_VARIANTS_YAML, encoding="utf-8")
    gen_path = tmp_path / "generation.yaml"
    gen_path.write_text(generation_yaml, encoding="utf-8")

    sweep_path = tmp_path / "full_sweep.yaml"
    sweep_path.write_text(
        SWEEP_YAML_TEMPLATE.format(
            kb_variants_path=kb_path.as_posix(),
            generation_path=gen_path.as_posix(),
            query_set_path=(tmp_path / "query_set.jsonl").as_posix(),
            ledger_path=(tmp_path / "ledger.json").as_posix(),
            processed_dir=(tmp_path / "processed").as_posix(),
            artifacts_dir=(tmp_path / "artifacts").as_posix(),
            cache_path=(tmp_path / "cache.sqlite").as_posix(),
            results_path=(tmp_path / "results.jsonl").as_posix(),
        ),
        encoding="utf-8",
    )
    return sweep_path


def edit(sweep_path: Path, old: str, new: str) -> None:
    text = sweep_path.read_text(encoding="utf-8")
    assert old in text
    sweep_path.write_text(text.replace(old, new), encoding="utf-8")


def test_every_yaml_setting_reaches_the_config_object(tmp_path):
    """No dead settings: each value in the YAML must show up where the code reads it."""
    config = load_sweep_config(write_sweep_configs(tmp_path))

    assert config.retrieval_methods == ["lexical", "dense", "hybrid"]
    assert config.kb_variant_names == ["curated", "mixed", "reduced"]
    assert config.baselines == ["oracle", "random", "none"]
    assert (config.top_k, config.seed, config.repeat, config.max_workers) == (5, 77, 0, 6)
    assert (config.hybrid.k_per_retriever, config.hybrid.rrf_k) == (40, 30)
    assert config.generation_metrics.answer_relevance_num_questions == 4
    assert config.generation_metrics.reference_agreement is False
    assert config.robustness.retrieval_method == "hybrid"
    assert config.robustness.perturbations == ("drop_top_relevant", "control_rerun")
    assert (config.budget.max_usd, config.budget.ledger_path) == (2.5, tmp_path / "ledger.json")
    assert config.kb_variant_config.base_corpus_size == 100
    assert config.chunking_config.chunk_size_words == 120
    assert config.generation.model == "gpt-4o-mini-2024-07-18"
    assert config.generation.effective_judge_model == "some-judge-model"
    assert (config.generation.temperature, config.generation.max_tokens, config.generation.max_attempts) == (0.2, 800, 4)
    assert config.kb_variant_chunks_path("curated") == tmp_path / "processed" / "kb_curated" / "chunks.jsonl"


def test_settings_logged_with_each_row_reflect_the_yaml(tmp_path):
    settings = load_sweep_config(write_sweep_configs(tmp_path)).settings_for_log()
    assert settings["model"] == "gpt-4o-mini-2024-07-18" and settings["judge_model"] == "some-judge-model"
    assert (settings["top_k"], settings["seed"], settings["rrf_k"], settings["hybrid_k_per_retriever"]) == (5, 77, 30, 40)
    assert settings["temperature"] == 0.2 and settings["max_tokens"] == 800
    assert settings["prompt_hash"] == prompt_hash("wellness_summary_v1")
    assert settings["generation_metrics"]["answer_relevance_num_questions"] == 4
    assert settings["metrics_version"] == "v2"


def test_judge_model_defaults_to_the_generation_model():
    settings = GenerationSettings(model="m", embedding_model="e", prompt_template_id="wellness_summary_v1", temperature=0.2, max_tokens=10)
    assert settings.effective_judge_model == "m"


def test_a_prompt_template_id_the_code_does_not_implement_is_rejected(tmp_path):
    """The template text lives in code. A config naming a different template
    would otherwise run the old prompt while the log claimed the new one."""
    sweep_path = write_sweep_configs(tmp_path, GENERATION_YAML.replace("wellness_summary_v1", "wellness_summary_v2"))
    with pytest.raises(ValueError, match="wellness_summary_v2"):
        load_sweep_config(sweep_path)


def test_kb_config_hash_is_deterministic(tmp_path):
    sweep_path = write_sweep_configs(tmp_path)
    assert load_sweep_config(sweep_path).kb_config_hash == load_sweep_config(sweep_path).kb_config_hash
    assert len(load_sweep_config(sweep_path).kb_config_hash) == 12


@pytest.mark.parametrize(
    "old,new,message",
    [
        ("retrieval_methods: [lexical, dense, hybrid]", "retrieval_methods: [lexical, made_up_method]", "Unknown retrieval method"),
        ("baselines: [oracle, random, none]", "baselines: [oracle, psychic]", "Unknown baseline"),
        ("top_k: 5", "top_k: 0", "top_k must be positive"),
        ("perturbations: [drop_top_relevant, control_rerun]", "perturbations: [shuffle]", "Unknown perturbation"),
        ("max_workers: 6", "max_workers: 0", "max_workers must be positive"),
    ],
)
def test_invalid_values_are_rejected(tmp_path, old, new, message):
    sweep_path = write_sweep_configs(tmp_path)
    edit(sweep_path, old, new)
    with pytest.raises(ValueError, match=message):
        load_sweep_config(sweep_path)


def test_robustness_method_must_be_one_the_sweep_actually_runs(tmp_path):
    sweep_path = write_sweep_configs(tmp_path)
    edit(sweep_path, "retrieval_methods: [lexical, dense, hybrid]", "retrieval_methods: [lexical, dense]")
    with pytest.raises(ValueError, match="robustness.retrieval_method"):
        load_sweep_config(sweep_path)


def test_unknown_perturbation_rejected_directly():
    with pytest.raises(ValueError):
        RobustnessConfig(perturbations=("drop_top_relevant", "typo"))


def test_the_real_full_sweep_config_is_the_documented_experiment():
    """Catches drift between configs/*.yaml and the experiment the thesis describes."""
    config = load_sweep_config(Path("configs/full_sweep.yaml"))
    assert config.retrieval_methods == ["lexical", "dense", "hybrid"]
    assert config.kb_variant_names == ["curated", "mixed", "reduced"]
    assert config.baselines == ["oracle", "random", "none"]
    assert config.top_k == 5
    assert (config.hybrid.k_per_retriever, config.hybrid.rrf_k) == (50, 60)
    assert config.generation.model == "gpt-4o-mini-2024-07-18"  # a dated snapshot, not the floating alias
    assert config.generation.embedding_model == "text-embedding-3-small"
    assert config.generation.temperature == 0.2
    assert config.robustness.retrieval_method == "dense"
    assert config.robustness.perturbations == ("drop_top_relevant", "inject_noise", "control_rerun")
    assert config.budget.max_usd == 10.0


def test_the_demo_config_loads_and_shares_the_generation_settings():
    demo = load_sweep_config(Path("configs/demo_sweep.yaml"))
    full = load_sweep_config(Path("configs/full_sweep.yaml"))
    assert demo.generation == full.generation
    assert demo.results_path != full.results_path
