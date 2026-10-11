"""Cross-check the metrics against two independent implementations on many
random rankings: `ranx` (an IR evaluation library) for all four metrics, and
scikit-learn's `ndcg_score` for NDCG.

Random cases deliberately include: fewer results than k, k larger than the
candidate pool, relevant chunks that are never retrieved, and queries whose
relevant chunks are all missing from the ranking.
"""

import random

import numpy as np
import pytest
from ranx import Qrels, Run, evaluate
from sklearn.metrics import ndcg_score

from evaluation.retrieval_metrics import ndcg_at_k, precision_at_k, reciprocal_rank, recall_at_k

POOL = [f"c{i}" for i in range(30)]


def random_case(rng: random.Random) -> tuple[list[str], set[str]]:
    ranked = rng.sample(POOL, rng.randint(1, 12))
    relevant = set(rng.sample(POOL, rng.randint(1, 8)))
    return ranked, relevant


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("k", [1, 3, 5, 10, 50])
def test_all_four_metrics_match_ranx(seed, k):
    rng = random.Random(seed * 1000 + k)
    cases = {f"q{i:02d}": random_case(rng) for i in range(25)}  # zero-padded: ranx returns scores in sorted query-id order

    qrels = Qrels({qid: {cid: 1 for cid in relevant} for qid, (_, relevant) in cases.items()})
    # ranx orders a run by score, so give strictly decreasing scores in list order
    run = Run({qid: {cid: float(len(ranked) - pos) for pos, cid in enumerate(ranked)} for qid, (ranked, _) in cases.items()})
    names = [f"precision@{k}", f"recall@{k}", f"mrr@{k}", f"ndcg@{k}"]
    reference = evaluate(qrels, run, names, return_mean=False)

    ours = {
        f"precision@{k}": [precision_at_k(r, rel, k) for r, rel in cases.values()],
        f"recall@{k}": [recall_at_k(r, rel, k) for r, rel in cases.values()],
        f"mrr@{k}": [reciprocal_rank(r, rel, k) for r, rel in cases.values()],
        f"ndcg@{k}": [ndcg_at_k(r, rel, k) for r, rel in cases.values()],
    }
    for name in names:
        np.testing.assert_allclose(ours[name], reference[name], atol=1e-9, err_msg=name)


@pytest.mark.parametrize("seed", range(200))
def test_ndcg_matches_scikit_learn(seed):
    rng = random.Random(seed)
    ranked, relevant = random_case(rng)
    k = rng.choice([1, 3, 5, 10])

    # scikit-learn scores a fixed candidate set: every pool item gets a true
    # relevance (0/1) and a predicted score; unretrieved items score below all
    # retrieved ones. Its top-k of that ordering is exactly our ranked list
    # (when the list is at least k long, which we ensure).
    if len(ranked) < min(k, len(POOL)):
        ranked = ranked + [c for c in POOL if c not in ranked][: k - len(ranked)]
    position = {cid: pos for pos, cid in enumerate(ranked)}
    y_true = [[1.0 if cid in relevant else 0.0 for cid in POOL]]
    y_score = [[float(len(POOL) - position[cid]) if cid in position else -1.0 for cid in POOL]]
    expected = ndcg_score(y_true, y_score, k=k)

    # unretrieved items all tie at -1 in sklearn, but they sit below rank k, so they never enter DCG@k
    assert ndcg_at_k(ranked, relevant, k) == pytest.approx(expected, abs=1e-9)
