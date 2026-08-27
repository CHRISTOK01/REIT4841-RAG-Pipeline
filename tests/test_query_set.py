from data.query_set import QuerySetConfig, load_query_set_jsonl, sample_query_set, save_query_set_jsonl
from data.schemas import SourceDocument


def make_docs(question_type: str, count: int, prefix: str) -> list[SourceDocument]:
    return [
        SourceDocument(
            doc_id=f"{prefix}:doc{i}:q{i}",
            subset="TestSubset",
            document_id=f"doc{i}",
            focus=f"Topic{i}",
            question_id=f"q{i}",
            question_type=question_type,
            question=f"What about {prefix}{i}?",
            answer="some answer text here that is long enough",
            url=None,
            umls_cui=None,
        )
        for i in range(count)
    ]


def test_sample_query_set_respects_num_queries():
    docs = make_docs("symptoms", 30, "sym") + make_docs("treatment", 70, "trt")
    config = QuerySetConfig(seed=1, num_queries=20)
    queries = sample_query_set(docs, config)
    assert len(queries) == 20


def test_sample_query_set_stratifies_by_question_type():
    docs = make_docs("symptoms", 30, "sym") + make_docs("treatment", 70, "trt")
    config = QuerySetConfig(seed=1, num_queries=20)
    queries = sample_query_set(docs, config)
    counts: dict[str, int] = {}
    for q in queries:
        counts[q.question_type] = counts.get(q.question_type, 0) + 1
    assert counts == {"symptoms": 6, "treatment": 14}  # exact proportional split (30/100, 70/100) * 20


def test_sample_query_set_is_deterministic():
    docs = make_docs("symptoms", 50, "sym")
    config = QuerySetConfig(seed=5, num_queries=10)
    q1 = sample_query_set(docs, config)
    q2 = sample_query_set(docs, config)
    assert [q.query_id for q in q1] == [q.query_id for q in q2]


def test_sample_query_set_caps_at_available_docs():
    docs = make_docs("symptoms", 5, "sym")
    config = QuerySetConfig(seed=1, num_queries=40)
    queries = sample_query_set(docs, config)
    assert len(queries) == 5


def test_query_id_matches_source_doc_id():
    docs = make_docs("symptoms", 5, "sym")
    config = QuerySetConfig(seed=1, num_queries=5)
    queries = sample_query_set(docs, config)
    doc_ids = {d.doc_id for d in docs}
    assert all(q.query_id in doc_ids for q in queries)


def test_query_set_jsonl_roundtrip(tmp_path):
    docs = make_docs("symptoms", 5, "sym")
    config = QuerySetConfig(seed=1, num_queries=5)
    queries = sample_query_set(docs, config)
    path = tmp_path / "queries.jsonl"
    save_query_set_jsonl(queries, path)
    loaded = load_query_set_jsonl(path)
    assert loaded == queries
