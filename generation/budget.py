"""Hard spend cap for OpenAI calls.

`SpendTracker` prices every call from the token counts the API reports, adds
it to a ledger file that persists across runs, and refuses to start another
call once the ledger has reached the cap. The cap is therefore cumulative over
everything that shares one ledger file, not per process.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import yaml

DEFAULT_PRICING_PATH = Path("configs/pricing.yaml")


class BudgetExceeded(RuntimeError):
    """The spend ledger has reached the configured cap. Never retried."""


class UnknownModelPrice(RuntimeError):
    """A budgeted call named a model with no entry in the pricing table."""


def load_pricing(path: Path = DEFAULT_PRICING_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def chat_cost_usd(pricing: dict, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    try:
        price = pricing["chat"][model]
    except KeyError as e:
        raise UnknownModelPrice(f"no chat price for model {model!r} in the pricing table") from e
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000


def embedding_cost_usd(pricing: dict, model: str, tokens: int) -> float:
    try:
        price = pricing["embedding"][model]
    except KeyError as e:
        raise UnknownModelPrice(f"no embedding price for model {model!r} in the pricing table") from e
    return tokens * price["input"] / 1_000_000


class SpendTracker:
    def __init__(self, max_usd: float, ledger_path: Path, pricing: dict | None = None):
        self.max_usd = max_usd
        self.ledger_path = ledger_path
        self.pricing = pricing if pricing is not None else load_pricing()
        self._lock = threading.Lock()
        self._state = {"spent_usd": 0.0, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "embedding_tokens": 0}
        if ledger_path.exists():
            self._state.update(json.loads(ledger_path.read_text(encoding="utf-8")))
        self._unsaved_calls = 0

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self._state["spent_usd"]

    def check(self, model: str, kind: str) -> None:
        """Call before every request: fails if the model is unpriced or the cap is reached."""
        if model not in self.pricing[kind]:
            raise UnknownModelPrice(f"no {kind} price for model {model!r} in the pricing table")
        with self._lock:
            if self._state["spent_usd"] >= self.max_usd:
                raise BudgetExceeded(
                    f"spend ledger {self.ledger_path} is at ${self._state['spent_usd']:.4f}, "
                    f"cap is ${self.max_usd:.2f}"
                )

    def record_chat(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        cost = chat_cost_usd(self.pricing, model, prompt_tokens, completion_tokens)
        self._add(cost, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    def record_embedding(self, model: str, tokens: int) -> None:
        self._add(embedding_cost_usd(self.pricing, model, tokens), embedding_tokens=tokens)

    def _add(self, cost: float, **token_counts: int) -> None:
        with self._lock:
            self._state["spent_usd"] += cost
            self._state["calls"] += 1
            for name, value in token_counts.items():
                self._state[name] += value
            self._unsaved_calls += 1
            if self._unsaved_calls >= 20:
                self._save_locked()

    def _save_locked(self) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        self._unsaved_calls = 0

    def save(self) -> None:
        with self._lock:
            self._save_locked()
