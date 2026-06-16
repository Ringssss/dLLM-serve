"""
Confidence-Aware Arbiter for DiffServe.

Uses CW-SRPT's per-request confidence signal to predict graph demand.
This is the organic link between dLLM scheduling and graph pool management:

  AR LLM Arbiter: sees batch size history → reactive (10s lag)
  dLLM Arbiter:   sees per-token confidence → predictive (1-2 iter ahead)

The confidence signal enables three unique capabilities:
  1. Predicted exits: requests with mean_confidence > 0.9 will finish in 1-2 iters
  2. Phase detection: ramp-up vs steady vs draining
  3. Batch size forecasting: predicted_next_bs = active - exits + arrivals

These predictions feed into the GraphPool to pre-materialize or evict graphs
before they're needed, eliminating all reactive latency.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ServingPhase(Enum):
    """Detected serving phase based on confidence distribution."""
    RAMP_UP = "ramp_up"      # queue growing, low avg confidence → bs will increase
    STEADY = "steady"         # stable batch size, mixed confidence
    DRAINING = "draining"     # many high-confidence requests → bs will decrease
    IDLE = "idle"             # no active requests


@dataclass
class ArbiterSignal:
    """Output signal from the Arbiter each iteration."""
    # Predictions
    predicted_next_bs: int = 0
    predicted_exits: int = 0       # requests expected to finish in 1-2 iters
    predicted_arrivals: int = 0
    phase: ServingPhase = ServingPhase.IDLE

    # Observed
    current_bs: int = 0
    active_count: int = 0
    pending_count: int = 0
    mean_confidence: float = 0.0
    graph_hit: bool = True

    # For GraphPool actions
    should_prefetch_bs: Optional[int] = None  # pre-materialize this bs
    should_evict_bs: Optional[List[int]] = None  # evict these cold bs


@dataclass
class ArbiterMetrics:
    """Cumulative accuracy metrics."""
    total_predictions: int = 0
    correct_bs_predictions: int = 0    # predicted_next_bs == actual_next_bs
    correct_exit_predictions: int = 0  # predicted_exits == actual_exits
    total_exits_predicted: int = 0
    total_exits_actual: int = 0

    # Phase distribution
    phase_counts: Dict[str, int] = field(default_factory=dict)

    # Batch size history
    bs_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    predicted_bs_history: deque = field(default_factory=lambda: deque(maxlen=1000))

    @property
    def bs_accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.correct_bs_predictions / self.total_predictions

    @property
    def bs_mae(self) -> float:
        """Mean absolute error of batch size predictions."""
        if not self.bs_history:
            return 0.0
        import numpy as np
        actual = list(self.bs_history)
        predicted = list(self.predicted_bs_history)
        n = min(len(actual), len(predicted))
        if n == 0:
            return 0.0
        return float(np.mean(np.abs(
            np.array(actual[-n:]) - np.array(predicted[-n:]))))


class ConfidenceAwareArbiter:
    """Predicts graph demand using dLLM confidence signals.

    Called each iteration by the DiffServe engine. Uses per-request
    confidence state to forecast batch size changes 1-2 iterations ahead.

    Usage:
        arbiter = ConfidenceAwareArbiter()
        # Each iteration:
        signal = arbiter.on_iteration(active_requests, batch, pending_count)
        # signal.predicted_next_bs tells GraphPool what to prepare
    """

    # Tunable thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.85  # request likely exits in 1-2 iters
    RAMP_THRESHOLD = 2.0              # pending/active ratio for ramp detection
    DRAIN_THRESHOLD = 0.5             # fraction of high-confidence requests for drain

    def __init__(self):
        self.metrics = ArbiterMetrics()
        self._prev_active_ids: set = set()
        self._prev_signal: Optional[ArbiterSignal] = None

    def on_iteration(
        self,
        active_requests: list,
        batch: list,
        pending_count: int = 0,
    ) -> ArbiterSignal:
        """Process one iteration and return predictions.

        Args:
            active_requests: All active (non-done) requests.
            batch: The subset selected for this iteration's forward.
            pending_count: Number of requests waiting in the pending queue.

        Returns:
            ArbiterSignal with predictions and recommendations.
        """
        # Validate previous prediction
        if self._prev_signal is not None:
            self._validate_prediction(
                self._prev_signal, len(batch), active_requests)

        current_bs = len(batch)
        active_count = len(active_requests)

        # ─── Confidence analysis ──────────────────────────────────
        confidences = []
        high_conf_count = 0

        for r in batch:
            nm = r.n_masked
            if nm > 0 and r.last_confidence_sum > 0:
                avg_conf = r.last_confidence_sum / nm
                confidences.append(avg_conf)
                if avg_conf > self.HIGH_CONFIDENCE_THRESHOLD:
                    high_conf_count += 1
            elif nm == 0:
                # Already done or about to be done
                high_conf_count += 1

        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

        # ─── Predict exits (core dLLM-specific capability) ────────
        # Requests with confidence > 0.85 typically exit in 1-2 iterations
        predicted_exits = high_conf_count

        # ─── Predict arrivals ─────────────────────────────────────
        # Simple heuristic: if pending > 0, expect ~1 arrival per iter
        # (actual depends on arrival rate vs iteration time)
        predicted_arrivals = min(pending_count, 2)

        # ─── Predict next batch size ──────────────────────────────
        predicted_next_bs = max(1, active_count - predicted_exits + predicted_arrivals)
        predicted_next_bs = min(predicted_next_bs, 256)  # cap

        # ─── Phase detection ──────────────────────────────────────
        phase = self._detect_phase(
            active_count, pending_count, mean_conf,
            high_conf_count, current_bs)

        # ─── Build signal ─────────────────────────────────────────
        signal = ArbiterSignal(
            predicted_next_bs=predicted_next_bs,
            predicted_exits=predicted_exits,
            predicted_arrivals=predicted_arrivals,
            phase=phase,
            current_bs=current_bs,
            active_count=active_count,
            pending_count=pending_count,
            mean_confidence=mean_conf,
        )

        # Record
        self.metrics.bs_history.append(current_bs)
        self.metrics.predicted_bs_history.append(predicted_next_bs)
        self.metrics.total_predictions += 1
        phase_name = phase.value
        self.metrics.phase_counts[phase_name] = (
            self.metrics.phase_counts.get(phase_name, 0) + 1)

        self._prev_signal = signal
        self._prev_active_ids = {r.id for r in active_requests}

        return signal

    def _detect_phase(
        self,
        active_count: int,
        pending_count: int,
        mean_conf: float,
        high_conf_count: int,
        current_bs: int,
    ) -> ServingPhase:
        """Detect the current serving phase.

        RAMP_UP:  lots of pending requests, low confidence → batch will grow
        DRAINING: many high-confidence requests → batch will shrink soon
        STEADY:   stable state
        IDLE:     nothing happening
        """
        if active_count == 0:
            return ServingPhase.IDLE

        if pending_count > 0 and pending_count > active_count * self.RAMP_THRESHOLD:
            return ServingPhase.RAMP_UP

        if current_bs > 0:
            drain_ratio = high_conf_count / current_bs
            if drain_ratio > self.DRAIN_THRESHOLD and mean_conf > 0.7:
                return ServingPhase.DRAINING

        return ServingPhase.STEADY

    def _validate_prediction(
        self,
        prev_signal: ArbiterSignal,
        actual_bs: int,
        active_requests: list,
    ):
        """Check how accurate the previous prediction was."""
        # BS accuracy
        if prev_signal.predicted_next_bs == actual_bs:
            self.metrics.correct_bs_predictions += 1

        # Exit accuracy
        current_ids = {r.id for r in active_requests}
        actual_exits = len(self._prev_active_ids - current_ids)
        self.metrics.total_exits_predicted += prev_signal.predicted_exits
        self.metrics.total_exits_actual += actual_exits
        if prev_signal.predicted_exits == actual_exits:
            self.metrics.correct_exit_predictions += 1

    def get_summary(self) -> dict:
        """Return a summary of arbiter accuracy."""
        m = self.metrics
        return {
            "total_iterations": m.total_predictions,
            "bs_accuracy": f"{m.bs_accuracy:.1%}",
            "bs_mae": f"{m.bs_mae:.2f}",
            "exits_predicted": m.total_exits_predicted,
            "exits_actual": m.total_exits_actual,
            "phase_distribution": dict(m.phase_counts),
        }
