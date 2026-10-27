"""
tokens.py — accumulates usage across all model calls in one turn, computes cost + wall time.
Price table lives in shared/pricing.py — edit there to update both API and SDK versions.
"""
import time
from shared.pricing import compute_cost


class TurnUsage:
    def __init__(self, model: str):
        self.model = model
        self.input_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0
        self.model_calls = 0
        self._start = time.perf_counter()

    def add(self, usage) -> None:
        """Accumulate a usage object from one model response."""
        if usage is None:
            return
        self.input_tokens  += getattr(usage, "prompt_tokens", 0)
        self.output_tokens += getattr(usage, "completion_tokens", 0)
        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            self.cached_tokens += getattr(details, "cached_tokens", 0)
        self.model_calls += 1
        print(f"[tokens.py] call #{self.model_calls}: "
              f"in={usage.prompt_tokens} cached={getattr(details, 'cached_tokens', 0) if details else 0} "
              f"out={usage.completion_tokens}")

    def cost_usd(self) -> float:
        return compute_cost(self.model, self.input_tokens, self.cached_tokens, self.output_tokens)

    def wall_time(self) -> float:
        return round(time.perf_counter() - self._start, 3)

    def summary(self) -> dict:
        return {
            "model_calls":    self.model_calls,
            "input_tokens":   self.input_tokens,
            "cached_tokens":  self.cached_tokens,
            "output_tokens":  self.output_tokens,
            "cost_usd":       round(self.cost_usd(), 6),
            "wall_time_s":    self.wall_time(),
        }
