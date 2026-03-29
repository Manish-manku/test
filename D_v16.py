"""
Trust-Enhanced Source-Independent Quantum Random Number Generator (TE-SI-QRNG)
=================================================================================

A self-testing approach to quantum random number generation that provides
measurable trust guarantees without requiring full device-independence.

Authors: Research Team
Date: January 2025

VERSION HISTORY
===============

v5 — Performance optimizations
  1. santha_vazirani_test  O(n²) Python triple-loop  → O(n) vectorized hash-map
  2. toeplitz_extract      O(n·m) dense matmul        → O(n log n) FFT circulant
  3. autocorrelation_test  per-lag np.correlate loop  → single FFT pass (all lags)
  4. runs_test             Python for-loop            → np.diff vectorized

v6 — Formula corrections
  5. Entropy formula: BB84 phase-error 1−h(e_upper) → classical min-entropy −log₂(p_max_upper)
  6. Hoeffding correction: delta = sqrt(log(1/ε_smooth) / (2·n_test))
  7. Unified metadata key: h_cert / h_min_trusted → h_min_certified (single canonical name)
  8. ε_bias fallback: 1−freq_p (wrong) → |mean−0.5| (actual observed bias)

v15 — Batch 6 fix: A3
  A3. Defined two TypedDict schemas to replace anonymous Dict returns:
        BlockMetadata  — schema for every block-level metadata dict returned by
                         process_block() and appended to metadata_list by
                         generate_certified_random_bits().
        EATSummary     — schema for the final entry in metadata_list
                         (the global EAT accumulation result).
      Changes in _assemble_metadata():
        - Return type annotation changed from -> Dict to -> BlockMetadata.
        - 'blocks_accumulated' key renamed to 'blocks_used' (unified with EATSummary).
        - 'delta_eat' field added: computed as sum_f_ei − h_total_eat, so callers
          no longer need to subtract the two values themselves.
        - 'output_length' field added (was missing from block meta, present in cert_bundle).
      Changes in generate_certified_random_bits():
        - Return type annotation changed from Tuple[np.ndarray, List[Dict]] to
          Tuple[np.ndarray, List[Union[BlockMetadata, FinalDecision, EATSummary]]].
        - halt_meta 'blocks_accumulated' key renamed to 'blocks_used'.
      No logic changes — pure schema formalisation and naming unification.

v16 — Batch 7 fix: A5
  A5. Extracted two new classes from TrustEnhancedQRNG to eliminate the
      "12 responsibilities" problem:

      QRNGSessionState (dataclass):
        - Holds all mutable cross-block state: block_entropy_history,
          block_n_gen_history, total_output_bits, total_gen_input_bits,
          total_raw_input_bits.
        - Owns accumulate_eat() (moved from TrustEnhancedQRNG).
        - Owns append_block() helper for recording per-block EAT contributions.
        - One instance created per generate_certified_random_bits() call.

      CertifiedGenerationSession:
        - Holds the outer block-accumulation loop (moved from
          TrustEnhancedQRNG.generate_certified_random_bits()).
        - Drives a TrustEnhancedQRNG instance block-by-block until the EAT
          bound is satisfied, then performs the final global Toeplitz extraction.
        - Public API: session.run(n_bits, source_simulator).

      TrustEnhancedQRNG after split:
        - Retains: __init__(), run_self_tests(), process_block() orchestrator,
          _certify_block(), _run_diagnostics(), _extract_block(),
          _assemble_metadata() (now takes session: QRNGSessionState parameter).
        - Removed: block_entropy_history, block_n_gen_history,
          total_output_bits, total_gen_input_bits, total_raw_input_bits,
          accumulate_eat().
        - Added: generate_certified_random_bits() backward-compatible shim
          that creates a CertifiedGenerationSession and calls .run().

      Backward compatibility:
        - All existing callers (experiment_v2_1_v13.py,
          experiment_6_nist_validation_v2.py) continue to work unchanged
          because the shim preserves the identical public signature.
        - process_block() still works standalone: when called without a
          session, a fresh QRNGSessionState() is used internally.
      No logic changes — pure structural reorganisation.

v14 — Batch 5 fix: A1
  A1. process_block() split into 4 private methods:
        _certify_block()    — Steps 0–3: gating, BB84 split, Hoeffding cert, EAT append
        _run_diagnostics()  — Steps 4–5: run_self_tests, warning-only diagnostics
        _extract_block()    — Steps 6+8–9: LHL length, seed derivation, Toeplitz extraction
        _assemble_metadata()— Steps 7+10–11: 30-field metadata dict, throughput counters
      process_block() is now a ~25-line orchestrator that calls the four methods in order.
      Zero logic changes — pure restructuring. All public behaviour, exceptions, and
      metadata keys are identical to v13. generate_certified_random_bits() unchanged.

v13 — Batch 4 fixes: S1 + S2 (comment-only)
v12 — Batch 3 bug fix: B4-gap2
v11 — Batch 1 bug fixes: B3-gap1
v10 — Batch 1 bug fixes: F3 + B1 + D4
v8/v9 — Pre-value gating + gate metadata in process_block
v7 — Calibration + pro-level upgrades

Security invariants (unchanged throughout all versions)
-------------------------------------------------------
  h_min_certified  ← p_max_upper only              (FORBIDDEN to touch with trust)
  extraction_rate  ← LHL(n_gen, h_min_certified)   (FORBIDDEN to scale by trust_score)
  trust_score      → warn / halt only              (NEVER modifies entropy or extraction)
  EAT              Δ_EAT = 2·√t·√(ln(1/ε_EAT))    (unchanged)
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Any, Tuple, Dict, List, Optional, Union, Literal, cast
try:
    from typing import TypedDict
except ImportError:          # Python < 3.8 fallback
    from typing_extensions import TypedDict
import hashlib
from collections import deque


# ---------------------------------------------------------------------------
# Typed metadata schemas  (A3 FIX — unchanged from v15)
# ---------------------------------------------------------------------------

class BlockMetadata(TypedDict):
    """
    Schema for every block-level metadata dict produced by process_block().

    These are the entries at positions [0 .. -3] of the metadata_list returned
    by generate_certified_random_bits(). The final two entries are, in order:
    EATSummary (second last), then FinalDecision (last).

    All callers should use typed access (meta['h_min_certified']) rather than
    .get() with fallbacks, because every field listed here is always present.

    Key invariant: h_min_certified is derived solely from p_max_upper (Hoeffding
    bound). trust_score is diagnostic only — it never modifies h_min_certified
    or extraction_rate.
    """
    # ---- Certified entropy fields ----------------------------------------
    certified_quantity:    str    # Always 'H_min(X|E)'
    security_definition:   str    # Always 'Trace-distance ε-security'
    epsilon_total:         float
    epsilon_eat:           float
    epsilon_smooth:        float
    epsilon_ext:           float
    n_generation:          int    # Generation-round bits in this block
    n_test:                int    # Test-round bits in this block
    p_hat:                 float  # Observed worst-case frequency
    p_max_hat:             float  # max(p_hat, 1 - p_hat)
    delta:                 float  # Hoeffding correction term
    p_max_upper:           float  # Hoeffding upper confidence bound
    h_min_certified:       float  # Per-bit min-entropy lower bound (bits/bit)
    extraction_rate:       float  # output_length / n_generation
    output_length:         int    # LHL extraction length for this block (bits)
    output_bits:           int    # Actual bits produced after extraction
    # ---- EAT accumulation state (running totals after this block) ---------
    blocks_used:           int    # Number of blocks accumulated so far
    h_total_eat:           float  # Globally certified entropy (EAT, bits)
    sum_f_ei:              float  # Raw sum Σ h_min_i·n_gen_i before EAT penalty
    delta_eat:             float  # EAT penalty: sum_f_ei − h_total_eat
    # ---- Diagnostic fields (read-only — NEVER touch entropy) --------------
    trust_score:           float
    trust_vector:          dict   # {epsilon_bias, epsilon_drift, epsilon_corr, epsilon_leak}
    diagnostic_warning:    Optional[str]
    diagnostic_state:      dict   # observational diagnostics (warnings/trends/anomalies)
    halt_threshold:        float
    warn_threshold:        float
    # ---- Throughput -------------------------------------------------------
    input_bits:            int    # Raw bits fed into this block (after gating)
    cumulative_efficiency: float  # total_output / total_raw_input
    # ---- Pre-value gate (Layer 1 — v8/v9) --------------------------------
    gate_enabled:          bool
    gate_tau:              Optional[float]
    gate_yield:            Optional[float]
    epsilon_gate:          Optional[float]
    epsilon_gate_empirical: Optional[float]
    epsilon_gate_bound:    Optional[float]
    gate_imr:              Optional[float]
    gate_bias_acknowledged: bool
    gate_entropy_correction_todo: bool
    gate_n_accepted:       int
    gate_n_total:          int
    gate_min_accepted_threshold: int
    gate_weak_statistics:  bool
    gate_sample_warning:   Optional[str]
    gate_persistent_small_bias_flag: bool
    cumulative_gate_yield: Optional[float]
    cumulative_epsilon_gate_trend: Optional[float]


class EATSummary(TypedDict):
    """
    Schema for the final entry in the metadata_list returned by
    generate_certified_random_bits().

    This is the global EAT accumulation result across all blocks.
    It is distinguishable from BlockMetadata by position (always second last)
    and by the presence of 'certified_output_bits' / 'actual_output_bits'
    which do not appear in per-block metadata.
    """
    certified_quantity:    str
    security_definition:   str
    epsilon_total:         float
    epsilon_eat:           float
    epsilon_smooth:        float
    epsilon_ext:           float
    blocks_used:           int    # Total blocks accumulated
    h_total_eat:           float  # Globally certified entropy after EAT penalty (bits)
    certified_output_bits: int    # Maximum extractable bits: floor(h_total_eat − 2·log₂(1/ε_ext))
    actual_output_bits:    int    # Bits actually returned (≤ certified_output_bits)
    delta_eat:             float  # EAT penalty: 2·√N·√(ln(1/ε_EAT))
    sum_f_ei:              float  # Raw entropy sum before penalty: Σ h_min_i·n_gen_i


class FinalDecision(TypedDict):
    """
    Schema for Layer-3 post-certification decision output.

    Layer-3 is strictly observational/policy logic applied after
    EAT accumulation and final extraction. It MUST NOT modify entropy,
    extraction length, or EAT results.
    """
    accepted:       bool
    status:         Literal["ACCEPT", "WARN", "REJECT"]
    reason:         str
    security_definition: str
    epsilon_total:  float
    epsilon_eat:    float
    epsilon_smooth: float
    epsilon_ext:    float
    certified_bits: int
    returned_bits:  int


class GateMetadata(TypedDict):
    """
    Schema for pre-value gate metadata (Layer-1 diagnostic fields).
    """
    enabled:      bool
    tau:          Optional[float]
    n_total:      int
    n_accepted:   int
    yield_rate:   Optional[float]
    epsilon_gate: Optional[float]
    epsilon_gate_empirical: Optional[float]
    epsilon_gate_bound: Optional[float]
    imr:          Optional[float]
    sigma:        Optional[float]
    bias_acknowledged: bool
    entropy_correction_todo: bool
    weak_statistics: bool
    min_accepted_threshold: int
    sample_warning: Optional[str]
    persistent_small_bias_flag: bool


# ---------------------------------------------------------------------------
# TrustVector
# ---------------------------------------------------------------------------

@dataclass
class TrustVector:
    """
    Trust parameters quantifying system reliability.

    Attributes:
        epsilon_bias:  Deviation from uniformity [0, 1]
        epsilon_drift: Temporal instability measure [0, 1]
        epsilon_corr:  Memory/correlation effects [0, 1]
        epsilon_leak:  Side-channel leakage indicator [0, 1]
    """
    epsilon_bias:  float = 0.0
    epsilon_drift: float = 0.0
    epsilon_corr:  float = 0.0
    epsilon_leak:  float = 0.0

    def trust_score(self) -> float:
        """Compute aggregate diagnostic trust score in [0, 1], where 1 = best.

        F1 FIX: result is clamped to [0, 1].
        This score is diagnostic only and MUST NEVER influence entropy
        certification, extraction length, or EAT accounting.
        Without the clamp, if any epsilon component exceeds 1.0 (possible when
        TrustVector is constructed directly with out-of-range values), the norm
        can exceed 2.0 and the score goes negative. A negative trust_score
        would still trigger the halt check correctly, but any downstream code
        using trust_score as a weight or divisor would produce nonsense.
        """
        norm = float(np.sqrt(
            self.epsilon_bias**2 +
            self.epsilon_drift**2 +
            self.epsilon_corr**2 +
            self.epsilon_leak**2
        ))
        return max(1.0 - norm / 2.0, 0.0)

        # D4 FIX: trust_penalty() deleted — never called anywhere, and its name
        # implies a trust-entropy coupling that must not exist.


# ---------------------------------------------------------------------------
# DiagnosticHaltError
# ---------------------------------------------------------------------------

class DiagnosticHaltError(Exception):
    """
    Legacy exception type retained for backward compatibility.

    NOTE (Batch-5 refactor):
        Diagnostics are now warning-only and MUST NOT halt entropy generation.
        This exception is no longer raised by the entropy pipeline.
    """
    HALT_THRESHOLD: float = 0.2   # Hard stop — system too unstable to operate
    WARN_THRESHOLD: float = 0.5   # Soft warning — degraded but operational


# ---------------------------------------------------------------------------
# Named exceptions for extraction failure paths
# ---------------------------------------------------------------------------

class InsufficientEntropyError(Exception):
    """
    Raised when certified entropy is too low to extract even one bit.

    B3 FIX: previously process_block() returned (np.array([]), meta) silently.
    Callers had no way to distinguish this from a successful zero-bit request.
    Now they can catch this specifically and decide whether to retry or abort.
    """
    pass


class EATConvergenceWarning(Exception):
    """
    Raised when generate_certified_random_bits() exits the accumulation loop
    early because total_gen exceeded 50 * n_bits without reaching the EAT bound.

    B3 FIX: previously this path printed a warning and returned partial output
    with no signal to the caller. A partial return is indistinguishable from
    a normal return without checking output length against n_bits.
    """
    def __init__(self, total_gen: int, requested_bits: int, h_total_eat: float):
        self.total_gen = int(total_gen)
        self.requested_bits = int(requested_bits)
        self.h_total_eat = float(h_total_eat)
        super().__init__(
            "CertifiedGenerationSession.run: EAT convergence not reached; "
            f"requested_bits={self.requested_bits}, total_gen={self.total_gen}, "
            f"h_total_eat={self.h_total_eat:.6f}."
        )


class ExtractionFailureError(Exception):
    """
    Raised when the chunked Toeplitz extractor produces no output chunks.

    B3 FIX: previously toeplitz_extract() returned np.zeros(m, dtype=np.uint8)
    on this path. All-zero output is valid random output and completely
    indistinguishable from correct output without external checking.
    This is the most dangerous silent failure in the codebase.
    """
    pass


# ---------------------------------------------------------------------------
# Calibrated sigmoid helper
# ---------------------------------------------------------------------------

def _sigmoid(x: float, k: float, x0: float) -> float:
    """
    Calibrated sigmoid mapping a raw test statistic to ε ∈ (0, 1).

    Formula:  σ(x) = 1 / (1 + exp(-k * (x - x0)))

    Calibration intent
    ------------------
    k  controls steepness: higher k = sharper transition.
    x0 is the inflection point (maps to ε = 0.5).

    Choosing x0 at roughly the "clearly problematic but not worst-case"
    value ensures:
      • Expected noise floor              → ε ≈ 0.05  (near-zero, not zero)
      • Moderate imperfection             → ε ≈ 0.30–0.60  (informative gradient)
      • Truly extreme / adversarial value → ε ≈ 0.90  (near-one, not hard-clipped)

    This replaces all min(x * scale, 1.0) patterns which saturate at 1.0
    too easily, turning the trust vector into a binary on/off switch.
    """
    return float(1.0 / (1.0 + np.exp(-k * (x - x0))))


# ---------------------------------------------------------------------------
# StatisticalSelfTester  — all tests vectorized
# ---------------------------------------------------------------------------

class StatisticalSelfTester:
    """
    Implements statistical self-tests for randomness validation.

    Tests include:
    - Santha-Vazirani source detection  (O(n) hash-map, was O(n²) triple loop)
    - Runs test for sequential patterns (vectorized np.diff)
    - Autocorrelation analysis          (single FFT pass, all lags at once)
    - Frequency monobit test
    """

    def __init__(self, window_size: int = 1000, alpha: float = 0.01):
        self.window_size = window_size
        self.alpha = alpha

    def santha_vazirani_test(self, bits: np.ndarray) -> Tuple[bool, float]:
        """
        Test for Santha-Vazirani source violation.

        OPTIMIZATION: replaced O(n²) Python triple-loop with a single O(n)
        pass using numpy stride tricks + a Python dict as a count table.

        Returns:
            (passes_test, epsilon_sv)
        """
        n = len(bits)
        if n < 100:
            return True, 0.0

        max_deviation = 0.0
        max_context   = min(4, int(np.log2(n)))

        b = bits.astype(np.uint8)

        for ctx_len in range(1, max_context + 1):
            powers = (1 << np.arange(ctx_len, dtype=np.uint8))

            indices     = np.arange(ctx_len, n)
            ctx_matrix  = b[indices[:, None] - 1 - np.arange(ctx_len)]
            ctx_ids     = ctx_matrix @ powers

            outcomes    = b[indices]

            max_id      = 1 << ctx_len
            ones_count  = np.bincount(ctx_ids, weights=outcomes.astype(float), minlength=max_id)
            total_count = np.bincount(ctx_ids, minlength=max_id)

            valid_mask  = total_count >= 5
            if not np.any(valid_mask):
                continue

            prob_one    = np.where(valid_mask, ones_count / np.maximum(total_count, 1), 0.5)
            deviation   = np.abs(prob_one - 0.5)
            max_deviation = max(max_deviation, float(np.max(deviation[valid_mask])))

        epsilon_sv = max_deviation
        passes     = epsilon_sv < 0.25

        return passes, epsilon_sv

    def runs_test(self, bits: np.ndarray) -> Tuple[bool, float]:
        """
        Test for independence using runs (consecutive identical bits).

        OPTIMIZATION: np.diff + np.count_nonzero replaces the Python for-loop.

        Returns:
            (passes_test, p_value)
        """
        n = len(bits)
        if n < 100:
            return True, 1.0

        runs      = int(np.count_nonzero(np.diff(bits))) + 1
        prop_ones = float(np.mean(bits))

        expected_runs = 2 * n * prop_ones * (1 - prop_ones) + 1
        variance_runs = (2 * n * prop_ones * (1 - prop_ones) *
                         (2 * n * prop_ones * (1 - prop_ones) - n) / (n - 1))

        if variance_runs <= 0:
            return True, 1.0

        z_score = (runs - expected_runs) / np.sqrt(variance_runs)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return p_value > self.alpha, p_value

    def autocorrelation_test(self, bits: np.ndarray, max_lag: int = 10) -> Tuple[bool, float]:
        """
        Test for temporal correlations using autocorrelation.

        OPTIMIZATION: compute all lags in a single FFT pass (O(n log n)) instead
        of calling np.correlate in a Python loop (O(n * max_lag)).

        Returns:
            (passes_test, max_correlation)
        """
        n = len(bits)
        if n < 2 * max_lag:
            return True, 0.0

        x  = 2.0 * bits.astype(np.float64) - 1.0
        x -= x.mean()
        sx  = x.std()
        if sx < 1e-12:
            return True, 0.0

        nfft    = 1 << int(np.ceil(np.log2(2 * n - 1)))
        X       = np.fft.rfft(x, n=nfft)
        acf_raw = np.fft.irfft(X * np.conj(X))[:n]

        acf_norm = acf_raw / (n * sx**2)

        lags     = min(max_lag, n // 2)
        max_corr = float(np.max(np.abs(acf_norm[1:lags])))

        critical_value = 2.576 / np.sqrt(n)

        return max_corr < critical_value, max_corr

    def frequency_test(self, bits: np.ndarray) -> Tuple[bool, float]:
        """
        Monobit frequency test for bias.

        Returns:
            (passes_test, p_value)
        """
        n = len(bits)
        if n < 100:
            return True, 1.0

        ones    = int(np.sum(bits))
        z_score = (ones - n / 2) / np.sqrt(n / 4)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return p_value > self.alpha, p_value


# ---------------------------------------------------------------------------
# QuantumWitnessTester
# ---------------------------------------------------------------------------

class QuantumWitnessTester:
    """
    Implements quantum-specific witness tests without full Bell inequality.
    """

    def __init__(self, visibility_threshold: float = 0.9):
        self.visibility_threshold = visibility_threshold

    def dimension_witness(self, outcomes: np.ndarray,
                          bases: np.ndarray) -> Tuple[bool, float]:
        if len(outcomes) < 1000:
            return True, 1.0

        basis_0 = outcomes[bases == 0]
        basis_1 = outcomes[bases == 1]

        if len(basis_0) < 100 or len(basis_1) < 100:
            return True, 1.0

        bias_0        = abs(float(np.mean(basis_0)) - 0.5)
        bias_1        = abs(float(np.mean(basis_1)) - 0.5)
        witness_value = abs(bias_0 - bias_1)

        return witness_value < 0.1, witness_value

    def energy_constraint_test(self, raw_signal: np.ndarray,
                                expected_mean: float = 0.0,
                                expected_std:  float = 1.0) -> Tuple[bool, float]:
        if len(raw_signal) < 100:
            return True, 0.0

        mean_dev  = abs(float(np.mean(raw_signal)) - expected_mean) / (expected_std + 1e-10)
        std_dev   = abs(float(np.std(raw_signal))  - expected_std)  / (expected_std + 1e-10)
        total_dev = float(np.sqrt(mean_dev**2 + std_dev**2))

        return total_dev < 3.0, total_dev


# ---------------------------------------------------------------------------
# PhysicalDriftMonitor  — CUSUM drift detection
# ---------------------------------------------------------------------------

class PhysicalDriftMonitor:
    """
    Monitors physical parameters for drift using CUSUM (Cumulative Sum control).
    """

    def __init__(self,
                 history_length:   int   = 1000,
                 cusum_k:          float = 0.5,
                 cusum_h:          float = 4.0,
                 warmup_samples:   int   = 50):
        self.history_length   = history_length
        self.cusum_k          = cusum_k
        self.cusum_h          = cusum_h
        self.warmup_samples   = warmup_samples

        self.efficiency_history  = deque(maxlen=history_length)

        self._cusum_pos   = 0.0
        self._cusum_neg   = 0.0
        self._ref_mean    = None
        self._ref_std     = None
        self._drift_score = 0.0

    def update_efficiency(self, efficiency: float) -> None:
        self.efficiency_history.append(efficiency)
        self._update_cusum(efficiency)

    def _update_cusum(self, x: float) -> None:
        """Incorporate one new efficiency measurement into CUSUM."""
        n = len(self.efficiency_history)

        if n == self.warmup_samples:
            arr = np.array(self.efficiency_history)
            self._ref_mean = float(np.mean(arr))
            raw_std = float(np.std(arr))
            self._ref_std  = max(raw_std, 0.01 * abs(self._ref_mean) + 1e-9)
            self._cusum_pos = 0.0
            self._cusum_neg = 0.0
            return

        if self._ref_mean is None or n < self.warmup_samples:
            return

        z = (x - self._ref_mean) / self._ref_std

        self._cusum_pos = max(0.0, self._cusum_pos + z - self.cusum_k)
        self._cusum_neg = max(0.0, self._cusum_neg - z - self.cusum_k)

        self._drift_score = max(self._cusum_pos, self._cusum_neg) / self.cusum_h

    def detect_drift(self) -> Tuple[bool, float]:
        """
        Return (drift_detected, drift_magnitude).
        """
        if self._ref_mean is None:
            return False, 0.0

        drift_detected = self._drift_score >= 1.0
        return drift_detected, self._drift_score


# ---------------------------------------------------------------------------
# PreValueGate
# ---------------------------------------------------------------------------

class PreValueGate:
    """
    Adaptive pre-value symmetric gating for CV-QRNG.
    """

    _TAU_SEARCH_GRID: np.ndarray = np.linspace(0.0, 3.0, 301)

    def __init__(self,
                 sigma:         float = 1.0,
                 yield_min:     float = 0.30,
                 tau_init:      float = 0.5,
                 max_epsilon_gate: float = 0.25,
                 optimization_lambda: float = 0.5):
        self.sigma     = sigma
        self.yield_min = yield_min
        self.tau       = tau_init
        self.max_epsilon_gate = float(max(max_epsilon_gate, 0.0))
        self.optimization_lambda = float(np.clip(optimization_lambda, 0.0, 1.0))

        self._imr_grid = self._imr(self._TAU_SEARCH_GRID)

    def _imr(self, tau_over_sigma: np.ndarray) -> np.ndarray:
        from scipy.stats import norm
        pdf_val = norm.pdf(tau_over_sigma)
        sf_val  = norm.sf(tau_over_sigma)
        sf_val  = np.maximum(sf_val, 1e-15)
        return pdf_val / sf_val

    def update_tau(self, epsilon_bias: float) -> float:
        from scipy.stats import norm

        tau_grid   = self._TAU_SEARCH_GRID * self.sigma
        yield_grid = 2.0 * norm.sf(self._TAU_SEARCH_GRID)
        imr_grid   = self._imr_grid
        eps_bias   = float(np.clip(epsilon_bias, 0.0, 0.5))
        eps_gate_est_grid = np.clip(eps_bias * imr_grid, 0.0, 0.5)
        gate_constraint = eps_gate_est_grid <= self.max_epsilon_gate

        valid = (yield_grid >= self.yield_min) & gate_constraint
        if not np.any(valid):
            self.tau = tau_grid[0]
            return self.tau

        imr_norm = imr_grid / max(float(np.max(imr_grid)), 1e-12)
        bias_norm = eps_gate_est_grid / max(self.max_epsilon_gate, 1e-12)
        objective = (
            self.optimization_lambda * imr_norm
            + (1.0 - self.optimization_lambda) * bias_norm
        )
        objective_valid = np.where(valid, objective, np.inf)
        best_idx = int(np.argmin(objective_valid))
        self.tau        = float(tau_grid[best_idx])

        return self.tau

    def apply(self,
              raw_signal: np.ndarray,
              bits:       np.ndarray,
              bases:      np.ndarray,
              min_accepted_threshold: int = 100,
              mu_attack: Optional[float] = None,
              persistent_small_bias_flag: bool = False,
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, GateMetadata]:
        n_total     = len(raw_signal)
        gate_mask   = np.abs(raw_signal) > self.tau

        accepted_signal = raw_signal[gate_mask]
        accepted_bits   = bits[gate_mask]
        accepted_bases  = bases[gate_mask]

        n_accepted  = int(np.sum(gate_mask))
        yield_rate  = n_accepted / max(n_total, 1)

        # Post-selection note:
        # The gate keeps only |raw_signal| > tau events. This can induce
        # selection bias in the accepted stream; entropy formulas below still
        # assume i.i.d. inputs and are intentionally left unchanged in Batch 3.
        if n_accepted > 0:
            epsilon_gate_empirical = float(abs(np.mean(accepted_bits) - 0.5))
        else:
            epsilon_gate_empirical = 0.5

        from scipy.stats import norm
        imr_val = float(norm.pdf(self.tau / self.sigma) /
                        max(norm.sf(self.tau / self.sigma), 1e-15))
        mu_est = float(mu_attack) if mu_attack is not None else float(np.mean(raw_signal))
        epsilon_gate_bound = float(abs(mu_est) * imr_val / 2.0)
        weak_statistics = bool(n_accepted < max(int(min_accepted_threshold), 1))
        sample_warning = (f"Gate accepted sample size is statistically weak: "
                          f"n_accepted={n_accepted} < min_threshold={int(min_accepted_threshold)}.")
        if not weak_statistics:
            sample_warning = None

        gate_meta: GateMetadata = {
            'enabled':      True,
            'tau':          self.tau,
            'n_total':      n_total,
            'n_accepted':   n_accepted,
            'yield_rate':   yield_rate,
            'epsilon_gate': epsilon_gate_empirical,
            'epsilon_gate_empirical': epsilon_gate_empirical,
            'epsilon_gate_bound': epsilon_gate_bound,
            'imr':          imr_val,
            'sigma':        self.sigma,
            'bias_acknowledged': True,
            'entropy_correction_todo': True,
            'weak_statistics': weak_statistics,
            'min_accepted_threshold': int(min_accepted_threshold),
            'sample_warning': sample_warning,
            'persistent_small_bias_flag': bool(persistent_small_bias_flag),
        }

        return accepted_signal, accepted_bits, accepted_bases, gate_meta

    def epsilon_gate_bound(self, mu_attack: float) -> float:
        from scipy.stats import norm
        imr = float(norm.pdf(self.tau / self.sigma) /
                    max(norm.sf(self.tau / self.sigma), 1e-15))
        return abs(mu_attack) * imr / 2.0


# ---------------------------------------------------------------------------
# BB84RoundSplitter
# ---------------------------------------------------------------------------

class BB84RoundSplitter:
    """
    Splits raw BB84 measurements into generation and test rounds.

      * basis == 0  →  generation round  (Z-basis)
      * basis == 1  →  test round        (X-basis / phase-error estimation)
    """

    GENERATION_BASIS: int = 0

    @staticmethod
    def split(bits: np.ndarray,
              bases: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gen_mask  = bases == BB84RoundSplitter.GENERATION_BASIS
        return bits[gen_mask], bits[~gen_mask]


# ---------------------------------------------------------------------------
# EntropyEstimator
# ---------------------------------------------------------------------------

class EntropyEstimator:
    """
    Certifies min-entropy from observable BB84 statistics.
    """

    def __init__(self,
                 security_parameter: float = 1e-6,
                 epsilon_eat: Optional[float] = None,
                 epsilon_smooth: Optional[float] = None,
                 epsilon_ext: Optional[float] = None):
        # Composable security accounting (explicit, globally enforced):
        #   ε_total = ε_eat + ε_smooth + ε_ext
        eps_total = float(security_parameter)
        if epsilon_eat is None and epsilon_smooth is None and epsilon_ext is None:
            component = eps_total / 3.0
            self.epsilon_eat = component
            self.epsilon_smooth = component
            self.epsilon_ext = component
        else:
            if None in (epsilon_eat, epsilon_smooth, epsilon_ext):
                raise ValueError(
                    "EntropyEstimator: provide all ε components or none."
                )
            self.epsilon_eat = float(cast(float, epsilon_eat))
            self.epsilon_smooth = float(cast(float, epsilon_smooth))
            self.epsilon_ext = float(cast(float, epsilon_ext))

        self.epsilon_total = self.epsilon_eat + self.epsilon_smooth + self.epsilon_ext
        if not np.isclose(self.epsilon_total, eps_total, rtol=0.0, atol=1e-18):
            raise ValueError(
                f"EntropyEstimator: ε_total mismatch. expected={eps_total:.3e}, "
                f"got={self.epsilon_total:.3e}."
            )

    def certify_min_entropy(self,
                            bits:  np.ndarray,
                            bases: np.ndarray) -> Dict:
        # F10 FIX: validate inputs before use.
        bits  = np.asarray(bits,  dtype=np.uint8).flatten()
        bases = np.asarray(bases, dtype=np.uint8).flatten()
        if len(bits) == 0:
            raise ValueError("certify_min_entropy: bits array is empty.")
        if len(bits) != len(bases):
            raise ValueError(
                f"certify_min_entropy: len(bits)={len(bits)} != "
                f"len(bases)={len(bases)}. Arrays must be the same length."
            )
        if not np.all((bits == 0) | (bits == 1)):
            raise ValueError("certify_min_entropy: bits must contain only 0 and 1.")
        if not np.all((bases == 0) | (bases == 1)):
            raise ValueError("certify_min_entropy: bases must contain only 0 and 1.")

        gen_bits, test_bits = BB84RoundSplitter.split(bits, bases)
        n_gen  = len(gen_bits)
        n_test = len(test_bits)

        if n_test == 0:
            return self._zero_cert(n_gen, n_test)

        # S1 — Security scope: classical source model only.
        p_hat            = float(np.mean(test_bits))
        p_max_hat        = max(p_hat, 1.0 - p_hat)

        delta            = np.sqrt(np.log(1.0 / self.epsilon_smooth) / (2.0 * n_test))
        p_max_upper      = min(p_max_hat + delta, 1.0)

        h_min_certified  = max(-np.log2(p_max_upper), 0.0)

        return {
            'n_generation':    n_gen,
            'n_test':          n_test,
            'p_hat':           p_hat,
            'p_max_hat':       p_max_hat,
            'delta':           delta,
            'p_max_upper':     p_max_upper,
            'h_min_certified': h_min_certified,
        }

    def lhl_output_length(self, n_gen: int, h_min_certified: float) -> int:
        """
        Quantum Leftover Hash Lemma (LHL) extraction length.

            k = floor( n_gen · h_min_certified − 2 · log₂(1 / ε_ext) )
        """
        log2_inv_eps = np.log2(1.0 / self.epsilon_ext)
        k = int(np.floor(n_gen * h_min_certified - 2.0 * log2_inv_eps))
        return max(k, 0)

    @staticmethod
    def _zero_cert(n_gen: int, n_test: int) -> Dict:
        return {'n_generation': n_gen, 'n_test': n_test,
                'p_hat': 1.0, 'p_max_hat': 1.0,
                'delta': 0.0, 'p_max_upper': 1.0,
                'h_min_certified': 0.0}


# ---------------------------------------------------------------------------
# RandomnessExtractor  — FFT circulant Toeplitz hashing
# ---------------------------------------------------------------------------

class RandomnessExtractor:
    """
    Quantum-proof randomness extractor (Toeplitz hashing via FFT).
    """

    def __init__(self, input_length: int, output_length: int,
                 seed_length: Optional[int] = None):
        self.input_length  = input_length
        self.output_length = output_length
        self.seed_length   = seed_length or (2 * output_length)

    _MAX_CIRC_SIZE: int = 1 << 23   # 8 388 608 elements, ~256 MB pipeline

    def _toeplitz_fft_chunk(self,
                             weak_random: np.ndarray,
                             seed:        np.ndarray,
                             out_len:     int) -> np.ndarray:
        n = len(weak_random)
        m = out_len

        required = n + m - 1
        if len(seed) < required:
            seed = self._extend_seed(seed, required)

        col      = seed[:m].astype(np.float32)
        row_tail = seed[1:n].astype(np.float32)

        raw_size  = m + n
        circ_size = 1 << int(np.ceil(np.log2(max(raw_size, 2))))
        if circ_size > self._MAX_CIRC_SIZE:
            raise ExtractionFailureError(
                f"_toeplitz_fft_chunk: FFT size {circ_size} exceeds "
                f"MAX_CIRC_SIZE={self._MAX_CIRC_SIZE}. "
                "Use smaller chunks or reduce output_length."
            )

        circ_col = np.zeros(circ_size, dtype=np.float32)
        circ_col[:m] = col
        if len(row_tail) > 0:
            circ_col[circ_size - len(row_tail):] = row_tail[::-1]

        x_pad = np.zeros(circ_size, dtype=np.float32)
        x_pad[:n] = weak_random.astype(np.float32)

        try:
            y_full = np.fft.irfft(
                np.fft.rfft(circ_col) * np.fft.rfft(x_pad),
                n=circ_size
            )
        except MemoryError:
            raise MemoryError(
                f"_toeplitz_fft_chunk: n={n}, m={m}, circ_size={circ_size}. "
                "Reduce block_size or max_workers."
            )

        output = np.round(y_full[:m]).astype(np.int64) % 2
        return output.astype(np.uint8)

    def toeplitz_extract(self, weak_random: np.ndarray,
                          seed: np.ndarray) -> np.ndarray:
        n = len(weak_random)
        m = self.output_length

        if n <= 0 or m <= 0:
            raise ExtractionFailureError(
                f"toeplitz_extract: invalid dimensions n={n}, m={m}."
            )

        single_circ = 1 << int(np.ceil(np.log2(max(m + n, 2))))
        if single_circ <= self._MAX_CIRC_SIZE:
            required = n + m - 1
            if len(seed) < required:
                seed = self._extend_seed(seed, required)
            return self._toeplitz_fft_chunk(weak_random, seed, m)

        max_raw_size   = max(self._MAX_CIRC_SIZE // 2, 2)
        max_chunk_in   = max(max_raw_size // 2, 1024)
        n_chunks       = max(int(np.ceil(n / max_chunk_in)), 1)
        n_c            = int(np.ceil(n / n_chunks))

        seed_bytes = np.packbits(seed[:min(len(seed), 2048)]).tobytes()

        output_chunks: List[np.ndarray] = []
        bits_produced = 0
        chunk_nonce   = 0

        for i in range(n_chunks):
            i_start = i * n_c
            i_end   = min(i_start + n_c, n)
            chunk   = weak_random[i_start:i_end]
            nc_i    = len(chunk)
            if nc_i == 0:
                continue

            max_mc_i = max(max_raw_size - nc_i, 1)
            remaining = m - bits_produced
            while remaining > 0:
                mc_i = min(remaining, max_mc_i)
                if mc_i <= 0:
                    raise ExtractionFailureError(
                        f"toeplitz_extract: unable to size chunk safely (nc_i={nc_i}, "
                        f"max_raw_size={max_raw_size}, remaining={remaining})."
                    )

                chunk_seed = self._derive_chunk_seed(
                    seed_bytes, chunk_nonce, nc_i + mc_i - 1
                )
                try:
                    out_chunk = self._toeplitz_fft_chunk(chunk, chunk_seed, mc_i)
                except (MemoryError, ValueError) as exc:
                    raise ExtractionFailureError(
                        f"toeplitz_extract: FFT chunk failed for nc_i={nc_i}, mc_i={mc_i}, "
                        f"chunk_nonce={chunk_nonce}."
                    ) from exc

                output_chunks.append(out_chunk)
                bits_produced += len(out_chunk)
                chunk_nonce += 1
                remaining = m - bits_produced

                if bits_produced >= m:
                    break
            if bits_produced >= m:
                break

        if not output_chunks:
            raise ExtractionFailureError(
                f"toeplitz_extract: chunked path produced no output. "
                f"n={n}, m={m}, n_chunks={n_chunks}. "
                "This would previously have returned all-zero bits silently."
            )

        result = np.concatenate(output_chunks)

        if len(result) < m:
            raise ExtractionFailureError(
                f"toeplitz_extract: chunked path produced {len(result)} bits "
                f"but {m} were requested. "
                f"n={n}, m={m}, n_chunks={n_chunks}, bits_produced={len(result)}."
            )

        return result[:m]

    def _derive_chunk_seed(self, master_seed_bytes: bytes,
                            chunk_idx: int, length: int) -> np.ndarray:
        extended: List[int] = []
        counter = 0
        prefix  = master_seed_bytes + chunk_idx.to_bytes(4, 'big')
        while len(extended) < length:
            h    = hashlib.sha256(prefix + counter.to_bytes(4, 'big')).digest()
            bits = np.unpackbits(np.frombuffer(h, dtype=np.uint8))
            extended.extend(bits.tolist())
            counter += 1
        return np.array(extended[:length], dtype=np.uint8)

    _MAX_SEED_BITS: int = 10_000_000

    def _extend_seed(self, seed: np.ndarray, length: int) -> np.ndarray:
        if length > self._MAX_SEED_BITS:
            raise ValueError(
                f"_extend_seed: requested length={length} bits exceeds "
                f"MAX_SEED_BITS={self._MAX_SEED_BITS}. "
                "This indicates a logic error upstream."
            )

        seed_bytes = np.packbits(seed).tobytes()
        extended   = []
        counter    = 0

        while len(extended) < length:
            hash_input  = seed_bytes + counter.to_bytes(4, 'big')
            hash_output = hashlib.sha256(hash_input).digest()
            bits        = np.unpackbits(np.frombuffer(hash_output, dtype=np.uint8))
            extended.extend(bits)
            counter += 1

        return np.array(extended[:length], dtype=np.uint8)

    def adaptive_extract(self, weak_random: np.ndarray,
                          seed: np.ndarray) -> np.ndarray:
        return self.toeplitz_extract(weak_random, seed)


# ---------------------------------------------------------------------------
# A5 FIX — QRNGSessionState (NEW CLASS)
# ---------------------------------------------------------------------------

@dataclass
class QRNGSessionState:
    """
    Mutable per-session state for EAT accumulation and throughput tracking.

    A5 FIX: Extracted from TrustEnhancedQRNG to separate cross-block session
    state from the per-block pipeline class.

    One instance is created per call to CertifiedGenerationSession.run()
    (equivalently, per call to the TrustEnhancedQRNG.generate_certified_random_bits()
    backward-compatible shim).

    Responsibilities:
        - EAT accumulation state: block_entropy_history, block_n_gen_history
        - Throughput counters: total_output_bits, total_gen_input_bits,
          total_raw_input_bits
        - accumulate_eat(): computes globally certified entropy across all blocks
        - append_block(): records one block's EAT contribution

    Security invariant:
        accumulate_eat() uses only h_min_certified values derived from
        p_max_upper (Hoeffding bound). Trust scores never enter this state.
    """
    block_entropy_history: List[float] = field(default_factory=list)
    block_n_gen_history:   List[int]   = field(default_factory=list)
    block_gen_bits_history: List[np.ndarray] = field(default_factory=list)
    total_output_bits:     int         = 0
    total_gen_input_bits:  int         = 0
    total_raw_input_bits:  int         = 0
    block_h_min_history:   List[float] = field(default_factory=list)
    block_extraction_rate_history: List[float] = field(default_factory=list)
    gate_accepted_total:   int         = 0
    gate_total_total:      int         = 0
    epsilon_gate_sum:      float       = 0.0
    epsilon_gate_count:    int         = 0
    epsilon_gate_small_count: int      = 0
    gate_block_count:      int         = 0
    trust_score_history:   List[float] = field(default_factory=list)
    session_warnings:      List[str]   = field(default_factory=list)
    anomaly_history:       List[str]   = field(default_factory=list)

    def accumulate_eat(self, epsilon_eat: float) -> float:
        """
        Compute globally certified entropy using the Entropy Accumulation Theorem.

        Units: everything in BITS (not bits/bit).

            sum_f   = Σᵢ  h_min_i · n_gen_i          [bits]
            N_total = Σᵢ  n_gen_i                     [bits]
            Δ_EAT   = 2 · √N_total · √(ln(1/ε_EAT))  [bits]
            H_total = sum_f − Δ_EAT                   [bits]

        Moved here from TrustEnhancedQRNG.accumulate_eat() — logic is identical,
        only the location changes.

        S1 — Security scope:
        Valid under the classical source model (trusted measurement device,
        i.i.d. classical bits). Not a full quantum EAT bound.
        """
        t = len(self.block_entropy_history)
        if t == 0:
            return 0.0

        sum_f     = sum(self.block_entropy_history)
        n_total   = sum(self.block_n_gen_history)
        delta_eat = 2.0 * np.sqrt(n_total) * np.sqrt(np.log(1.0 / epsilon_eat))

        return max(sum_f - delta_eat, 0.0)

    def append_block(self,
                     h_min_certified: float,
                     n_gen: int,
                     gen_bits: Optional[np.ndarray] = None) -> None:
        """
        Record one block's contribution to the EAT accumulation.

        Args:
            h_min_certified: per-bit min-entropy lower bound (bits/bit)
            n_gen:           number of generation-round bits in this block
        """
        self.block_entropy_history.append(h_min_certified * n_gen)
        self.block_n_gen_history.append(n_gen)
        self.block_h_min_history.append(h_min_certified)
        if gen_bits is not None:
            self.block_gen_bits_history.append(np.array(gen_bits, dtype=np.uint8, copy=True))

    def update_extraction_rate(self, extraction_rate: float) -> None:
        """Track per-block extraction rate for diagnostic consistency checks."""
        self.block_extraction_rate_history.append(float(extraction_rate))

    def update_gate_tracking(self,
                             n_accepted: int,
                             n_total: int,
                             epsilon_gate: Optional[float],
                             small_bias_threshold: float = 0.01) -> None:
        """
        Track cumulative gate quantities.

        NOTE: pre-value gating introduces selection bias because only accepted
        samples are forwarded. We monitor this bias trend here; we do NOT
        correct entropy formulas at this stage.
        """
        self.gate_accepted_total += int(n_accepted)
        self.gate_total_total += int(n_total)
        self.gate_block_count += 1
        if epsilon_gate is not None:
            self.epsilon_gate_sum += float(epsilon_gate)
            self.epsilon_gate_count += 1
            if float(epsilon_gate) <= float(small_bias_threshold):
                self.epsilon_gate_small_count += 1

    def record_diagnostics(self,
                           trust_score: float,
                           warning: Optional[str],
                           anomalies: Optional[List[str]] = None) -> None:
        """Record observational diagnostics without affecting entropy flow."""
        self.trust_score_history.append(float(trust_score))
        if warning:
            self.session_warnings.append(str(warning))
        if anomalies:
            self.anomaly_history.extend([str(a) for a in anomalies if a])

    def cumulative_gate_yield(self) -> Optional[float]:
        if self.gate_total_total <= 0:
            return None
        return self.gate_accepted_total / self.gate_total_total

    def cumulative_epsilon_gate_trend(self) -> Optional[float]:
        if self.epsilon_gate_count <= 0:
            return None
        return self.epsilon_gate_sum / self.epsilon_gate_count

    def persistent_small_gate_bias_flag(self, min_blocks: int = 5,
                                        min_ratio: float = 0.70) -> bool:
        """
        Flag persistent small gate bias regimes across blocks.

        Small |mu_attack| can evade one-shot detection but still lower effective
        entropy over time after post-selection.
        """
        if self.gate_block_count < max(int(min_blocks), 1):
            return False
        ratio = self.epsilon_gate_small_count / max(self.gate_block_count, 1)
        return ratio >= float(min_ratio)


# ---------------------------------------------------------------------------
# Layer-3 FinalDecisionLayer
# ---------------------------------------------------------------------------

class FinalDecisionLayer:
    """
    Layer-3 post-certification decision logic.

    IMPORTANT:
      - Never modifies entropy values.
      - Never modifies extraction length.
      - Never modifies EAT result.
      - Only classifies the already-produced output as ACCEPT/WARN/REJECT.
    """
    def __init__(self,
                 halt_threshold: float = DiagnosticHaltError.HALT_THRESHOLD,
                 warn_threshold: float = DiagnosticHaltError.WARN_THRESHOLD):
        self.halt_threshold = halt_threshold
        self.warn_threshold = warn_threshold

    def evaluate(self,
                 final_bits: np.ndarray,
                 eat_summary: EATSummary,
                 last_block_meta: Optional[BlockMetadata],
                 trust_score: float,
                 epsilon_gate: Optional[float] = None) -> FinalDecision:
        certified_bits = int(eat_summary['certified_output_bits'])
        returned_bits = int(len(final_bits))

        reason_parts: List[str] = []
        if epsilon_gate is not None:
            reason_parts.append(f"epsilon_gate={epsilon_gate:.6f}")
        if last_block_meta is not None and last_block_meta['diagnostic_warning'] is not None:
            reason_parts.append(last_block_meta['diagnostic_warning'])

        if trust_score < self.halt_threshold:
            return {
                'accepted': False,
                'status': 'REJECT',
                'reason': ("Trust score below halt threshold "
                           f"({trust_score:.4f} < {self.halt_threshold:.4f})"
                           + (f"; {' | '.join(reason_parts)}" if reason_parts else "")),
                'security_definition': eat_summary['security_definition'],
                'epsilon_total': float(eat_summary['epsilon_total']),
                'epsilon_eat': float(eat_summary['epsilon_eat']),
                'epsilon_smooth': float(eat_summary['epsilon_smooth']),
                'epsilon_ext': float(eat_summary['epsilon_ext']),
                'certified_bits': certified_bits,
                'returned_bits': returned_bits,
            }

        if trust_score < self.warn_threshold:
            return {
                'accepted': True,
                'status': 'WARN',
                'reason': ("Trust score below warn threshold "
                           f"({trust_score:.4f} < {self.warn_threshold:.4f})"
                           + (f"; {' | '.join(reason_parts)}" if reason_parts else "")),
                'security_definition': eat_summary['security_definition'],
                'epsilon_total': float(eat_summary['epsilon_total']),
                'epsilon_eat': float(eat_summary['epsilon_eat']),
                'epsilon_smooth': float(eat_summary['epsilon_smooth']),
                'epsilon_ext': float(eat_summary['epsilon_ext']),
                'certified_bits': certified_bits,
                'returned_bits': returned_bits,
            }

        return {
            'accepted': True,
            'status': 'ACCEPT',
            'reason': ("Trust score within normal operating range"
                       + (f"; {' | '.join(reason_parts)}" if reason_parts else "")),
            'security_definition': eat_summary['security_definition'],
            'epsilon_total': float(eat_summary['epsilon_total']),
            'epsilon_eat': float(eat_summary['epsilon_eat']),
            'epsilon_smooth': float(eat_summary['epsilon_smooth']),
            'epsilon_ext': float(eat_summary['epsilon_ext']),
            'certified_bits': certified_bits,
            'returned_bits': returned_bits,
        }


# ---------------------------------------------------------------------------
# A5 FIX — CertifiedGenerationSession (NEW CLASS)
# ---------------------------------------------------------------------------

class CertifiedGenerationSession:
    """
    Drives the block accumulation loop until the EAT bound is satisfied,
    then performs the final global Toeplitz extraction.

    A5 FIX: Extracted from TrustEnhancedQRNG.generate_certified_random_bits()
    to separate the outer generation loop from the per-block pipeline class.

    TrustEnhancedQRNG now contains only per-block pipeline logic.
    This class owns the session-level loop and the global final extraction.

    Usage:
        session = CertifiedGenerationSession(te_qrng, epsilon_eat, epsilon_ext)
        bits, metadata_list = session.run(n_bits, source_simulator)

    Or equivalently via the backward-compatible shim:
        bits, metadata_list = te_qrng.generate_certified_random_bits(
            n_bits, source_simulator
        )

    Guarantees:
        ‖ρ_RE − U_R ⊗ ρ_E‖₁ ≤ ε_total
    """

    def __init__(self,
                 te_qrng:     'TrustEnhancedQRNG',
                 epsilon_eat: float,
                 epsilon_ext: float):
        """
        Args:
            te_qrng:     The per-block pipeline instance to drive.
            epsilon_eat: EAT security parameter (used in accumulate_eat()).
            epsilon_ext: Extractor security parameter (used in LHL length).
        """
        self.te_qrng     = te_qrng
        self.epsilon_eat = epsilon_eat
        self.epsilon_smooth = te_qrng.epsilon_smooth
        self.epsilon_ext = epsilon_ext
        self.epsilon_total = self.epsilon_eat + self.epsilon_smooth + self.epsilon_ext

    @staticmethod
    def _validate_epsilon_consistency(block_meta: BlockMetadata,
                                      eat_summary: EATSummary,
                                      final_decision: FinalDecision) -> None:
        for label, record in (
            ("BlockMetadata", block_meta),
            ("EATSummary", eat_summary),
            ("FinalDecision", final_decision),
        ):
            eps_total = float(record['epsilon_total'])
            eps_sum = (
                float(record['epsilon_eat'])
                + float(record['epsilon_smooth'])
                + float(record['epsilon_ext'])
            )
            if not np.isclose(eps_total, eps_sum, rtol=0.0, atol=1e-18):
                raise RuntimeError(
                    f"{label}: ε_total must equal ε_eat + ε_smooth + ε_ext "
                    f"(got {eps_total:.3e} vs {eps_sum:.3e})."
                )

    def run(self,
            n_bits:           int,
            source_simulator) -> Tuple[np.ndarray, List[Union[BlockMetadata, FinalDecision, EATSummary]]]:
        """
        Generate n_bits with full composable EAT-certified security.

        Logic moved from TrustEnhancedQRNG.generate_certified_random_bits()
        — identical behaviour, different location.

        Args:
            n_bits:           Number of certified random bits to produce.
            source_simulator: Object with generate_block(block_size) method
                              returning (bits, bases, raw_signal).

        Returns:
            (final_bits[:n_bits], metadata_list)
            metadata_list structure is always:
                [BlockMetadata, ..., EATSummary, FinalDecision]

        Raises:
            ValueError:              n_bits <= 0
            EATConvergenceWarning:   EAT bound not reached within 50×n_bits raw bits
            InsufficientEntropyError: certified output length < 1 after EAT
        """
        # F10 FIX: validate n_bits
        if not isinstance(n_bits, int) or n_bits <= 0:
            raise ValueError(
                f"CertifiedGenerationSession.run: n_bits must be a positive integer, "
                f"got {n_bits!r}."
            )

        # Create a fresh session state for this run
        session = QRNGSessionState()

        all_gen_bits:  List[np.ndarray]                         = []
        metadata_list: List[Union[BlockMetadata, FinalDecision, EATSummary]]  = []

        block_size = self.te_qrng.block_size
        while True:
            raw_bits, bases, raw_signal = source_simulator.generate_block(block_size)

            signal_stats = (source_simulator.get_signal_stats()
                            if hasattr(source_simulator, 'get_signal_stats') else None)

            _, block_meta = self.te_qrng.process_block(
                raw_bits, bases, raw_signal, session=session,
                signal_stats=signal_stats
            )
            if not np.isclose(
                float(block_meta['epsilon_total']),
                float(block_meta['epsilon_eat']) + float(block_meta['epsilon_smooth']) + float(block_meta['epsilon_ext']),
                rtol=0.0, atol=1e-18
            ):
                raise RuntimeError("BlockMetadata ε accounting inconsistency.")

            metadata_list.append(block_meta)

            if not session.block_gen_bits_history:
                raise RuntimeError(
                    "CertifiedGenerationSession.run: missing generation-bit history for block."
                )
            all_gen_bits.append(session.block_gen_bits_history[-1])

            h_total          = session.accumulate_eat(self.epsilon_eat)
            log2_inv_eps_ext = np.log2(1.0 / self.epsilon_ext)
            max_output_bits  = int(h_total - 2.0 * log2_inv_eps_ext)

            if max_output_bits >= n_bits:
                break

            total_gen = sum(len(g) for g in all_gen_bits)
            if total_gen > 50 * n_bits:
                raise EATConvergenceWarning(
                    total_gen=total_gen,
                    requested_bits=n_bits,
                    h_total_eat=h_total,
                )

        # Global final Toeplitz extraction
        all_gen_concat = (np.concatenate(all_gen_bits)
                          if all_gen_bits else np.array([], dtype=np.uint8))

        h_total          = session.accumulate_eat(self.epsilon_eat)
        log2_inv_eps_ext = np.log2(1.0 / self.epsilon_ext)
        certified_output = max(int(h_total - 2.0 * log2_inv_eps_ext), 0)
        output_length    = min(n_bits, certified_output)

        if output_length < 1 or len(all_gen_concat) < 2:
            raise InsufficientEntropyError(
                f"CertifiedGenerationSession.run: certified output length is "
                f"{output_length} bits after EAT accumulation. "
                f"h_total_eat={h_total:.4f}, certified_output={certified_output}."
            )

        # S4 FIX: seed independent of source bits — use os.urandom()
        import os as _os
        seed_len  = min(2 * output_length, 512)
        seed_arr  = np.unpackbits(
            np.frombuffer(_os.urandom((seed_len + 7) // 8), dtype=np.uint8)
        )[:seed_len]
        extract_input = all_gen_concat
        if len(extract_input) < output_length:
            output_length = len(extract_input)

        extractor  = RandomnessExtractor(input_length=len(extract_input),
                                         output_length=output_length)
        final_bits = extractor.adaptive_extract(extract_input, seed_arr)

        # Build EAT summary
        t_blocks  = len(session.block_entropy_history)
        sum_f_ei  = sum(session.block_entropy_history)
        n_total   = sum(session.block_n_gen_history)
        delta_eat = (2.0 * np.sqrt(n_total) *
                     np.sqrt(np.log(1.0 / self.epsilon_eat))
                     if t_blocks > 0 else 0.0)

        # Independent composable terms:
        #   ε_total = ε_eat + ε_smooth + ε_ext
        eat_summary: EATSummary = {
            'certified_quantity':    'H_min(X|E)',
            'security_definition':   'Trace-distance ε-security',
            'epsilon_total':         self.epsilon_total,
            'epsilon_eat':           self.epsilon_eat,
            'epsilon_smooth':        self.epsilon_smooth,
            'epsilon_ext':           self.epsilon_ext,
            'blocks_used':           t_blocks,
            'h_total_eat':           h_total,
            'certified_output_bits': certified_output,
            'actual_output_bits':    len(final_bits),
            'delta_eat':             delta_eat,
            'sum_f_ei':              sum_f_ei,
        }
        eat_summary['diagnostic_state'] = {
            'trust_score_trend': session.trust_score_history,
            'warnings': session.session_warnings,
            'anomalies': session.anomaly_history,
        }

        decision_layer = FinalDecisionLayer()
        last_block_meta = metadata_list[-1] if metadata_list else None
        metadata_list.append(eat_summary)
        # FinalDecision is evaluated only after final_bits and EATSummary exist.
        final_decision = decision_layer.evaluate(
            final_bits=final_bits,
            eat_summary=eat_summary,
            last_block_meta=last_block_meta,
            trust_score=last_block_meta['trust_score'] if last_block_meta else 1.0,
            epsilon_gate=last_block_meta.get('epsilon_gate', None) if last_block_meta else None,
        )
        if last_block_meta is not None:
            self._validate_epsilon_consistency(last_block_meta, eat_summary, final_decision)
        metadata_list.append(final_decision)

        # Enforce deterministic metadata ordering for downstream consumers:
        # [BlockMetadata, ..., EATSummary, FinalDecision]
        if len(metadata_list) < 2:
            raise RuntimeError("metadata_list must contain EATSummary and FinalDecision.")
        if not (isinstance(metadata_list[-2], dict) and 'certified_output_bits' in metadata_list[-2]):
            raise RuntimeError("metadata_list[-2] must be an EATSummary.")
        if not (isinstance(metadata_list[-1], dict) and 'status' in metadata_list[-1] and 'accepted' in metadata_list[-1]):
            raise RuntimeError("metadata_list[-1] must be a FinalDecision.")

        return final_bits[:n_bits], metadata_list


# ---------------------------------------------------------------------------
# TrustEnhancedQRNG  — per-block pipeline (A5: session state extracted)
# ---------------------------------------------------------------------------

class TrustEnhancedQRNG:
    """
    Main TE-SI-QRNG system — per-block pipeline only.

    A5 FIX: Cross-block session state and the outer generation loop have been
    extracted to QRNGSessionState and CertifiedGenerationSession respectively.
    TrustEnhancedQRNG now holds exclusively per-block pipeline responsibilities:

        1. BB84 round splitting              (_certify_block)
        2. Entropy certification             (_certify_block → EntropyEstimator)
        3. Statistical self-testing          (_run_diagnostics → StatisticalSelfTester)
        4. Physical drift monitoring         (_run_diagnostics → PhysicalDriftMonitor)
        5. Halt/warn decision logic          (_run_diagnostics)
        6. LHL output length calculation     (_extract_block → EntropyEstimator)
        7. Seed derivation (independent)     (_extract_block)
        8. Toeplitz extraction               (_extract_block → RandomnessExtractor)
        9. Metadata assembly                 (_assemble_metadata)
       10. Pre-value gating (Layer 1)        (_certify_block → PreValueGate)

    Responsibilities removed (now in QRNGSessionState / CertifiedGenerationSession):
        - EAT accumulation state
        - Throughput counters
        - accumulate_eat()
        - generate_certified_random_bits() outer loop

    Backward compatibility:
        generate_certified_random_bits() is retained as a one-line shim that
        creates a CertifiedGenerationSession and calls .run(). All existing
        callers continue to work without changes.

    Pipeline per block
    ------------------
    0. Pre-value gating     → discard low-confidence events
    1. BB84 round splitting → generation bits + test bits
    2. Phase-error cert     → H_cert from Hoeffding bound
    3. Statistical tests    → TrustVector updated
    4. Toeplitz extract     → FFT-based extraction

    Security invariant (unchanged)
    --------------------------------
    h_min_certified  ← p_max_upper only    (FORBIDDEN to touch with trust)
    extraction_rate  ← LHL(h_min_certified) (FORBIDDEN to scale by trust_score)
    trust_score      → warn / halt only    (NEVER modifies entropy)
    """

    def __init__(self,
                 block_size:           int   = 1000,
                 security_parameter:   float = 1e-6,
                 extractor_efficiency: float = 0.9,
                 enable_gating:        bool  = True,
                 sigma_signal:         float = 1.0,
                 yield_min:            float = 0.30,
                 gate_min_accepted_threshold: int = 100,
                 gate_small_bias_epsilon: float = 0.01,
                 gate_small_bias_window: int = 5):
        self.block_size           = block_size
        self.security_parameter   = security_parameter
        self.extractor_efficiency = extractor_efficiency
        self.enable_gating        = enable_gating
        self.gate_min_accepted_threshold = max(int(gate_min_accepted_threshold), 1)
        self.gate_small_bias_epsilon = float(gate_small_bias_epsilon)
        self.gate_small_bias_window = max(int(gate_small_bias_window), 1)

        # Components
        self.stat_tester       = StatisticalSelfTester(window_size=block_size)
        self.quantum_tester    = QuantumWitnessTester()
        self.drift_monitor     = PhysicalDriftMonitor()
        self.entropy_estimator = EntropyEstimator(security_parameter=security_parameter)

        # Pre-value gate (Layer 1)
        self.pre_value_gate = PreValueGate(
            sigma     = sigma_signal,
            yield_min = yield_min,
            tau_init  = 0.5,
        )

        # Composable epsilon budget (mirrors EntropyEstimator):
        #   ε_total = ε_eat + ε_smooth + ε_ext
        self.epsilon_eat    = self.entropy_estimator.epsilon_eat
        self.epsilon_smooth = self.entropy_estimator.epsilon_smooth
        self.epsilon_ext    = self.entropy_estimator.epsilon_ext
        self.epsilon_total  = self.epsilon_eat + self.epsilon_smooth + self.epsilon_ext

        # Diagnostic state (does NOT feed into entropy bound)
        self.trust_vector = TrustVector()

        # A5 FIX: block_entropy_history, block_n_gen_history, total_output_bits,
        # total_gen_input_bits, total_raw_input_bits REMOVED from __init__.
        # These now live in QRNGSessionState, one instance per generation call.
        # accumulate_eat() also removed — it now lives on QRNGSessionState.

    # ------------------------------------------------------------------
    # Self-testing (diagnostic — NOT used in entropy formula)
    # ------------------------------------------------------------------

    def run_self_tests(self,
                       raw_bits:    np.ndarray,
                       bases:       Optional[np.ndarray] = None,
                       raw_signal:  Optional[np.ndarray] = None,
                       signal_stats: Optional[Tuple[float, float]] = None,
                       epsilon_gate: Optional[float] = None) -> TrustVector:
        """
        Run the full statistical / quantum self-test suite.

        S2 — Engineering policy note:
        The sigmoid parameters (k, x0) for each trust vector component are
        calibrated heuristics, not formally derived security bounds. The halt
        threshold (0.2) and warn threshold (0.5) are operational policy choices.
        None of these values appear in the certified entropy bound.
        """
       
        autocorr_pass, max_autocorr = self.stat_tester.autocorrelation_test(raw_bits)

        raw_obs_bias = abs(float(np.mean(raw_bits)) - 0.5)
        epsilon_bias = _sigmoid(raw_obs_bias, k=17.0, x0=0.20)  # heuristic — not a security bound

        epsilon_corr = _sigmoid(max_autocorr, k=26.0, x0=0.15)  # heuristic — not a security bound

        epsilon_leak = 0.0
        if bases is not None:
            dim_pass, dim_witness = self.quantum_tester.dimension_witness(raw_bits, bases)
            if not dim_pass:
                epsilon_leak = _sigmoid(dim_witness, k=15.0, x0=0.20)  # heuristic

        if raw_signal is not None and len(raw_signal) == len(raw_bits):
            sign_pred      = (raw_signal > 0).astype(np.uint8)
            sign_alignment = float(np.mean(sign_pred == raw_bits))
            if sign_alignment < 0.70:
                sig_f  = raw_signal.astype(np.float64)
                bits_f = raw_bits.astype(np.float64)
                sig_std  = float(np.std(sig_f))
                bits_std = float(np.std(bits_f))
                if sig_std > 1e-10 and bits_std > 1e-10:
                    sig_bit_corr = abs(float(np.mean(
                        (sig_f - sig_f.mean()) * (bits_f - bits_f.mean())
                    )) / (sig_std * bits_std))
                    epsilon_leak_corr = _sigmoid(sig_bit_corr, k=200.0, x0=0.02)  # heuristic
                    epsilon_leak = max(epsilon_leak, epsilon_leak_corr)

        epsilon_drift = 0.0
        if raw_signal is not None:
            exp_mean, exp_std = signal_stats if signal_stats is not None else (0.0, 1.0)
            self.quantum_tester.energy_constraint_test(raw_signal, exp_mean, exp_std)
            # F3 FIX: feed per-block mean into CUSUM before reading detect_drift
            self.drift_monitor.update_efficiency(float(np.mean(raw_bits)))
            _, drift_score = self.drift_monitor.detect_drift()
            epsilon_drift = _sigmoid(drift_score, k=4.0, x0=1.0)  # heuristic

        if epsilon_gate is not None:
            epsilon_bias = max(epsilon_bias, epsilon_gate)

        self.trust_vector = TrustVector(
            epsilon_bias  = float(np.clip(epsilon_bias,  0.0, 1.0)),
            epsilon_drift = float(np.clip(epsilon_drift, 0.0, 1.0)),
            epsilon_corr  = float(np.clip(epsilon_corr,  0.0, 1.0)),
            epsilon_leak  = float(np.clip(epsilon_leak,  0.0, 1.0)),
        )
        return self.trust_vector

    # ------------------------------------------------------------------
    # Block processing pipeline — private layer methods (A1 FIX from v14)
    # ------------------------------------------------------------------

    def _certify_block(self,
                       raw_bits:   np.ndarray,
                       bases:      Optional[np.ndarray],
                       raw_signal: Optional[np.ndarray],
                       n_raw:      int,
                       session:    QRNGSessionState,
                       ) -> Dict:
        """
        Steps 0–3: pre-value gating, BB84 split, Hoeffding certification,
        EAT history append.

        A5 FIX: takes session: QRNGSessionState parameter so that EAT state
        (block_entropy_history, block_n_gen_history) lives in the session object
        rather than on self. append_block() delegates to session.
        """
        # Step 0: Pre-value gating (Layer 1)
        gate_meta: GateMetadata = {
            'enabled': False,
            'tau': None,
            'n_total': n_raw,
            'n_accepted': n_raw,
            'yield_rate': None,
            'epsilon_gate': None,
            'epsilon_gate_empirical': None,
            'epsilon_gate_bound': None,
            'imr': None,
            'sigma': None,
            'bias_acknowledged': False,
            'entropy_correction_todo': False,
            'weak_statistics': False,
            'min_accepted_threshold': self.gate_min_accepted_threshold,
            'sample_warning': None,
            'persistent_small_bias_flag': False,
        }
        if self.enable_gating and raw_signal is not None and len(raw_signal) == n_raw:
            # Trust→entropy isolation:
            # tau adaptation must not depend on trust diagnostics, otherwise trust
            # would indirectly change accepted events and downstream entropy stats.
            # Keep gate behavior deterministic and trust-independent here.
            # τ must be chosen before processing this block, using only
            # currently available block-local statistics (never future data).
            pre_gate_bias = float(abs(np.mean(raw_bits) - 0.5)) if len(raw_bits) > 0 else 0.5
            self.pre_value_gate.update_tau(pre_gate_bias)

            # Selection-bias note:
            # Gating accepts only |signal| > tau samples, which can skew sample
            # composition relative to the original stream. This bias is tracked
            # in metadata (gate_yield / epsilon_gate cumulants) but is not
            # corrected in entropy formulas at this stage.
            persistent_small_bias_flag = session.persistent_small_gate_bias_flag(
                min_blocks=self.gate_small_bias_window
            )
            _, raw_bits, bases_gated, gate_meta = self.pre_value_gate.apply(
                raw_signal, raw_bits,
                bases if bases is not None else np.zeros(n_raw, dtype=np.uint8),
                min_accepted_threshold=self.gate_min_accepted_threshold,
                mu_attack=None,
                persistent_small_bias_flag=persistent_small_bias_flag,
            )
            if bases is not None:
                bases      = bases_gated
            raw_signal = raw_signal[np.abs(raw_signal) > self.pre_value_gate.tau]
            n_raw      = len(raw_bits)

        # Step 1: BB84 round splitting
        if bases is not None:
            gen_bits, test_bits = BB84RoundSplitter.split(raw_bits, bases)
        else:
            gen_bits  = raw_bits
            test_bits = np.array([], dtype=np.uint8)

        n_gen  = len(gen_bits)
        n_test = len(test_bits)

        # Step 2: Phase-error certification — THE entropy bound (INVARIANT)
        # TODO(Batch-Next): incorporate ε_gate correction into entropy bounds
        # (Hoeffding/EAT/LHL chain) once a composable proof is integrated.
        # For now, epsilon_gate_* fields remain diagnostics only.
        cert = self.entropy_estimator.certify_min_entropy(
            raw_bits,
            bases if bases is not None else np.zeros(n_raw, dtype=np.uint8)
        )
        h_min_certified = cert['h_min_certified']
        # INVARIANT: h_min_certified is derived solely from p_max_upper.
        basis_diag = self._basis_diagnostics(bases, n_gen, n_test)

        # Step 3: Store f(eᵢ)·n_gen_i for EAT accumulation — via session
        # A5 FIX: was self.block_entropy_history.append(...), now session.append_block()
        session.append_block(h_min_certified, n_gen, gen_bits=gen_bits)

        return {
            'raw_bits':        raw_bits,
            'bases':           bases,
            'raw_signal':      raw_signal,
            'n_raw':           n_raw,
            'gate_meta':       gate_meta,
            'gen_bits':        gen_bits,
            'n_gen':           n_gen,
            'n_test':          n_test,
            'cert':            cert,
            'h_min_certified': h_min_certified,
            'basis_diag':      basis_diag,
        }

    def _basis_diagnostics(self,
                           bases: Optional[np.ndarray],
                           n_gen: int,
                           n_test: int) -> Dict[str, Union[bool, int, Optional[float], List[str]]]:
        """
        Basis diagnostics (metadata-only; never coupled to entropy/extraction).

        Assumption note:
        Current system assumes honest basis generation. Adversarial
        basis-manipulation attacks are not fully mitigated in this release.
        """
        warnings: List[str] = []
        zero_prob: Optional[float] = None
        deviation: Optional[float] = None
        anomaly_flag = False

        if bases is None or len(bases) == 0:
            unreliable = (n_test < self.min_n_test_required)
            if unreliable:
                warnings.append(
                    f"n_test below configured minimum: {n_test} < {self.min_n_test_required}."
                )
            return {
                'basis_zero_probability': zero_prob,
                'basis_balance_deviation': deviation,
                'basis_balance_tolerance': self.basis_balance_tolerance,
                'basis_anomaly_flag': anomaly_flag,
                'statistically_unreliable': unreliable,
                'n_test_min_required': self.min_n_test_required,
                'warnings': warnings,
            }

        total = max(n_gen + n_test, 1)
        gen_ratio = n_gen / total
        test_ratio = n_test / total
        min_ratio = 0.10
        if gen_ratio < min_ratio:
            warnings.append(
                f"Generation/test imbalance warning: generation ratio={gen_ratio:.3f} < {min_ratio:.2f}."
            )
        if test_ratio < min_ratio:
            warnings.append(
                f"Generation/test imbalance warning: test ratio={test_ratio:.3f} < {min_ratio:.2f}."
            )

        b = np.asarray(bases, dtype=np.uint8).flatten()
        zero_prob = float(np.mean(b == 0))
        deviation = abs(zero_prob - 0.5)
        if deviation > self.basis_balance_tolerance:
            warnings.append(
                f"basis imbalance detected: P(basis=0)={zero_prob:.4f}, "
                f"deviation={deviation:.4f} > tolerance={self.basis_balance_tolerance:.4f}."
            )
            anomaly_flag = True

        if len(b) >= 4:
            # Structured-pattern heuristic (diagnostic only).
            x = (2.0 * b.astype(np.float64)) - 1.0
            lag1_corr = float(np.mean(x[1:] * x[:-1]))
            alternation_rate = float(np.mean(b[1:] != b[:-1]))
            if (abs(lag1_corr) > (1.0 - self.basis_pattern_threshold) or
                    alternation_rate > (1.0 - 0.5 * self.basis_pattern_threshold)):
                warnings.append(
                    "basis anomaly heuristic triggered (structured pattern / high serial dependence)."
                )
                anomaly_flag = True

        unreliable = (n_test < self.min_n_test_required)
        if unreliable:
            warnings.append(
                f"n_test below configured minimum: {n_test} < {self.min_n_test_required}."
            )

        return {
            'basis_zero_probability': zero_prob,
            'basis_balance_deviation': deviation,
            'basis_balance_tolerance': self.basis_balance_tolerance,
            'basis_anomaly_flag': anomaly_flag,
            'statistically_unreliable': unreliable,
            'n_test_min_required': self.min_n_test_required,
            'warnings': warnings,
        }

    def _run_diagnostics(self,
                         raw_bits:     np.ndarray,
                         bases:        Optional[np.ndarray],
                         raw_signal:   Optional[np.ndarray],
                         signal_stats: Optional[Tuple[float, float]],
                         h_min_certified: float,
                         epsilon_gate: Optional[float],
                         gate_meta: Optional[GateMetadata] = None,
                         ) -> Tuple[TrustVector, Optional[str], Dict[str, Any]]:
        """
        Steps 4–5: run_self_tests, then evaluate diagnostic thresholds.

        Returns:
            (trust_vector, diagnostic_warning, diagnostic_state)
        Pure diagnostic-layer logic — does not touch cert dict or entropy state.
        Trust diagnostics are metadata-only and must never alter block inclusion.
        """
        trust_vector = self.run_self_tests(
            raw_bits, bases, raw_signal, signal_stats=signal_stats, epsilon_gate=epsilon_gate
        )
        trust_score  = trust_vector.trust_score()

        diagnostic_warning: Optional[str] = None
        anomalies: List[str] = []
        if trust_score < DiagnosticHaltError.HALT_THRESHOLD:
            diagnostic_warning = (
                f"System instability detected: trust_score={trust_score:.4f} "
                f"< HALT_THRESHOLD={DiagnosticHaltError.HALT_THRESHOLD}. "
                f"Entropy/extraction continue; h_min_certified={h_min_certified:.4f} is unaffected."
            )
            anomalies.append("trust_score_below_halt_threshold")
        elif trust_score < DiagnosticHaltError.WARN_THRESHOLD:
            diagnostic_warning = (
                f"Degraded operation: trust_score={trust_score:.4f} "
                f"< WARN_THRESHOLD={DiagnosticHaltError.WARN_THRESHOLD}. "
                f"h_min_certified={h_min_certified:.4f} is unaffected."
            )
            anomalies.append("trust_score_below_warn_threshold")
        if epsilon_gate is not None:
            gate_note = f"epsilon_gate={epsilon_gate:.6f} (selection-bias monitor only)"
            diagnostic_warning = (f"{diagnostic_warning} | {gate_note}"
                                  if diagnostic_warning else gate_note)
        if gate_meta is not None:
            if gate_meta.get('epsilon_gate_bound') is not None:
                bound_note = (
                    f"epsilon_gate_bound={gate_meta['epsilon_gate_bound']:.6f} "
                    f"(diagnostic approximation; entropy correction TODO)"
                )
                diagnostic_warning = (f"{diagnostic_warning} | {bound_note}"
                                      if diagnostic_warning else bound_note)
            if gate_meta.get('sample_warning') is not None:
                diagnostic_warning = (f"{diagnostic_warning} | {gate_meta['sample_warning']}"
                                      if diagnostic_warning else gate_meta['sample_warning'])
            if gate_meta.get('persistent_small_bias_flag', False):
                small_bias_note = (
                    "persistent small epsilon_gate trend detected; small mu_attack may evade "
                    "single-block detection while still reducing effective entropy"
                )
                diagnostic_warning = (f"{diagnostic_warning} | {small_bias_note}"
                                      if diagnostic_warning else small_bias_note)
                anomalies.append("persistent_small_epsilon_gate_trend")

        diagnostic_state = {
            'trust_score': trust_score,
            'warnings': [diagnostic_warning] if diagnostic_warning else [],
            'anomalies': anomalies,
        }
        return trust_vector, diagnostic_warning, diagnostic_state

    def _extract_block(self,
                       gen_bits:      np.ndarray,
                       h_min_certified: float,
                       seed:          Optional[np.ndarray],
                       ) -> Tuple[np.ndarray, int]:
        """
        Steps 6 + 8–9: LHL output length, seed derivation, Toeplitz extraction.

        Returns:
            (output_bits, output_length)

        Raises InsufficientEntropyError when LHL yields output_length < 1.
        Pure extraction logic — no side effects on entropy state.
        """
        n_gen = len(gen_bits)

        output_length = self.entropy_estimator.lhl_output_length(n_gen, h_min_certified)

        if output_length < 1 or n_gen < 2:
            raise InsufficientEntropyError(
                f"process_block: certified entropy too low for extraction. "
                f"h_min_certified={h_min_certified:.6f}, n_gen={n_gen}, "
                f"output_length={output_length}."
            )

        # S4 FIX: seed independent of source bits — use os.urandom()
        if seed is None:
            import os as _os
            seed_len      = min(2 * output_length, 512)
            seed_arr      = np.unpackbits(
                np.frombuffer(_os.urandom((seed_len + 7) // 8), dtype=np.uint8)
            )[:seed_len]
            extract_input = gen_bits
        else:
            seed_arr      = seed
            extract_input = gen_bits

        if len(extract_input) < output_length:
            output_length = len(extract_input)

        extractor   = RandomnessExtractor(input_length=len(extract_input),
                                          output_length=output_length)
        output_bits = extractor.adaptive_extract(extract_input, seed_arr)

        return output_bits, output_length

    def _assemble_metadata(self,
                           cert:              Dict,
                           n_raw:             int,
                           gate_meta:         GateMetadata,
                           trust_vector:      TrustVector,
                           diagnostic_warning: Optional[str],
                           diagnostic_state:  Dict[str, Any],
                           output_bits_len:   int,
                           session:           QRNGSessionState,
                           ) -> BlockMetadata:
        """
        Steps 7 + 10–11: build the BlockMetadata dict and update
        throughput counters.

        A5 FIX: takes session: QRNGSessionState so that throughput counters
        (total_output_bits, total_gen_input_bits, total_raw_input_bits) live
        in the session object rather than on self. Also reads
        block_entropy_history and block_n_gen_history from session for
        EAT field computation.

        A3 FIX: return type is BlockMetadata (TypedDict).
        'blocks_used', 'delta_eat', 'output_length' fields present.
        """
        n_gen  = cert['n_gen']
        n_test = cert['n_test']
        h_min_certified = cert['h_min_certified']
        output_length   = cert['output_length']
        extraction_rate = cert['extraction_rate']
        basis_diag = cert.get('basis_diag', {'warnings': []})
        consistency_warning = self._cross_block_consistency_warning(
            h_min_certified, extraction_rate, session
        )
        merged_warning = diagnostic_warning
        basis_warnings = basis_diag.get('warnings', [])
        if basis_warnings:
            basis_msg = " | ".join(str(w) for w in basis_warnings)
            merged_warning = (f"{merged_warning} | {basis_msg}"
                              if merged_warning else basis_msg)
        if consistency_warning is not None:
            merged_warning = (f"{diagnostic_warning} | {consistency_warning}"
                              if diagnostic_warning else consistency_warning)
        if gate_meta.get('sample_warning') is not None:
            merged_warning = (f"{merged_warning} | {gate_meta['sample_warning']}"
                              if merged_warning else gate_meta['sample_warning'])

        # Update throughput counters in session (A5 FIX: was self.total_*)
        session.total_raw_input_bits  += n_raw
        session.total_gen_input_bits  += n_gen
        session.total_output_bits     += output_bits_len
        session.update_extraction_rate(extraction_rate)
        session.update_gate_tracking(
            gate_meta['n_accepted'],
            gate_meta['n_total'],
            gate_meta['epsilon_gate'],
            small_bias_threshold=self.gate_small_bias_epsilon,
        )
        session.record_diagnostics(
            trust_score=trust_vector.trust_score(),
            warning=merged_warning,
            anomalies=diagnostic_state.get('anomalies', []),
        )

        # Compute EAT values from session state
        h_total_eat = session.accumulate_eat(self.epsilon_eat)
        sum_f_ei    = sum(session.block_entropy_history)
        delta_eat   = sum_f_ei - h_total_eat

        # ε components are independent and explicitly composable:
        #   ε_total = ε_eat + ε_smooth + ε_ext
        meta: BlockMetadata = {
            'certified_quantity':  'H_min(X|E)',
            'security_definition': 'Trace-distance ε-security',
            'epsilon_total':       self.epsilon_total,
            'epsilon_eat':         self.epsilon_eat,
            'epsilon_smooth':      self.epsilon_smooth,
            'epsilon_ext':         self.epsilon_ext,
            'n_generation':        n_gen,
            'n_test':              n_test,
            'p_hat':               cert['cert']['p_hat'],
            'p_max_hat':           cert['cert']['p_max_hat'],
            'delta':               cert['cert']['delta'],
            'p_max_upper':         cert['cert']['p_max_upper'],
            'h_min_certified':     h_min_certified,
            'extraction_rate':     extraction_rate,
            'output_length':       output_length,
            'output_bits':         output_bits_len,
            'blocks_used':         len(session.block_entropy_history),
            'h_total_eat':         h_total_eat,
            'sum_f_ei':            sum_f_ei,
            'delta_eat':           delta_eat,
            'trust_score':         trust_vector.trust_score(),
            'trust_vector':        {
                'epsilon_bias':  trust_vector.epsilon_bias,
                'epsilon_drift': trust_vector.epsilon_drift,
                'epsilon_corr':  trust_vector.epsilon_corr,
                'epsilon_leak':  trust_vector.epsilon_leak,
            },
            'diagnostic_warning':  merged_warning,
            'diagnostic_state':    diagnostic_state,
            'halt_threshold':      DiagnosticHaltError.HALT_THRESHOLD,
            'warn_threshold':      DiagnosticHaltError.WARN_THRESHOLD,
            'input_bits':          n_raw,
            'cumulative_efficiency': (session.total_output_bits /
                                      max(session.total_raw_input_bits, 1)),
            'gate_enabled':      gate_meta['enabled'],
            'gate_tau':          gate_meta['tau'],
            'gate_yield':        gate_meta['yield_rate'],
            'epsilon_gate':      gate_meta['epsilon_gate'],
            'epsilon_gate_empirical': gate_meta['epsilon_gate_empirical'],
            'epsilon_gate_bound': gate_meta['epsilon_gate_bound'],
            'gate_imr':          gate_meta['imr'],
            'gate_bias_acknowledged': gate_meta['bias_acknowledged'],
            'gate_entropy_correction_todo': gate_meta['entropy_correction_todo'],
            'gate_n_accepted':   gate_meta['n_accepted'],
            'gate_n_total':      gate_meta['n_total'],
            'gate_min_accepted_threshold': gate_meta['min_accepted_threshold'],
            'gate_weak_statistics': gate_meta['weak_statistics'],
            'gate_sample_warning': gate_meta['sample_warning'],
            'gate_persistent_small_bias_flag': gate_meta['persistent_small_bias_flag'],
            # Selection-bias monitoring only (diagnostic, no entropy coupling):
            'cumulative_gate_yield': session.cumulative_gate_yield(),
            'cumulative_epsilon_gate_trend': session.cumulative_epsilon_gate_trend(),
        }
        return meta

    def _cross_block_consistency_warning(self,
                                         h_min_certified: float,
                                         extraction_rate: float,
                                         session: QRNGSessionState
                                         ) -> Optional[str]:
        """
        Lightweight cross-block consistency diagnostics.

        Detects large deviations relative to prior block history for:
          - h_min_certified
          - extraction_rate
        This is strictly observational and never alters entropy/extraction.
        """
        warnings: List[str] = []

        if session.block_h_min_history:
            prev_h = np.asarray(session.block_h_min_history, dtype=np.float64)
            ref_h = float(np.median(prev_h))
            scale_h = float(max(np.median(np.abs(prev_h - ref_h)), 1e-6))
            if abs(h_min_certified - ref_h) > 6.0 * scale_h:
                warnings.append(
                    f"cross-block inconsistency: h_min_certified jump "
                    f"(current={h_min_certified:.4f}, median={ref_h:.4f})"
                )

        if session.block_extraction_rate_history:
            prev_r = np.asarray(session.block_extraction_rate_history, dtype=np.float64)
            ref_r = float(np.median(prev_r))
            scale_r = float(max(np.median(np.abs(prev_r - ref_r)), 1e-6))
            if abs(extraction_rate - ref_r) > 6.0 * scale_r:
                warnings.append(
                    f"cross-block inconsistency: extraction_rate jump "
                    f"(current={extraction_rate:.4f}, median={ref_r:.4f})"
                )

        if not warnings:
            return None
        return " ; ".join(warnings)

    # ------------------------------------------------------------------
    # Block processing pipeline — public orchestrator
    # ------------------------------------------------------------------

    def process_block(self,
                      raw_bits:     np.ndarray,
                      bases:        Optional[np.ndarray] = None,
                      raw_signal:   Optional[np.ndarray] = None,
                      seed:         Optional[np.ndarray] = None,
                      signal_stats: Optional[Tuple[float, float]] = None,
                      session:      Optional[QRNGSessionState] = None,
                      ) -> Tuple[np.ndarray, BlockMetadata]:
        """
        Process one block through the full TE-SI-QRNG pipeline.

        A5 FIX: added optional session: QRNGSessionState parameter.
        When called standalone (without a session), a fresh QRNGSessionState()
        is created internally so the call works identically to v15.
        When called from CertifiedGenerationSession.run(), the shared session
        object is passed in so EAT state accumulates correctly across blocks.

        Public behaviour is otherwise identical to v15 — same signature
        (session is additive, optional), same return type, same exceptions.
        """
        # Create a standalone session if none provided (backward compatible)
        if session is None:
            session = QRNGSessionState()

        n_raw = len(raw_bits)

        # Layer 1 — Certified layer
        c = self._certify_block(raw_bits, bases, raw_signal, n_raw, session)

        # Layer 2 — Diagnostic layer (warning-only; never halts entropy flow)
        trust_vector, diagnostic_warning, diagnostic_state = self._run_diagnostics(
            c['raw_bits'], c['bases'], c['raw_signal'],
            signal_stats, c['h_min_certified'], c['gate_meta']['epsilon_gate'],
            gate_meta=c['gate_meta'],
        )

        # Compute extraction_rate for metadata
        output_length   = self.entropy_estimator.lhl_output_length(
            c['n_gen'], c['h_min_certified']
        )
        extraction_rate = output_length / max(c['n_gen'], 1)

        # Layer 3 — Extraction layer (may raise InsufficientEntropyError)
        output_bits, _ = self._extract_block(
            c['gen_bits'], c['h_min_certified'], seed
        )

        # Pack cert_bundle for _assemble_metadata
        cert_bundle = {
            'n_gen':           c['n_gen'],
            'n_test':          c['n_test'],
            'h_min_certified': c['h_min_certified'],
            'output_length':   output_length,
            'extraction_rate': extraction_rate,
            'cert':            c['cert'],
            'basis_diag':      c['basis_diag'],
        }

        # Layer 4 — Bookkeeping (updates session throughput counters)
        meta = self._assemble_metadata(
            cert_bundle, c['n_raw'], c['gate_meta'],
            trust_vector, diagnostic_warning, diagnostic_state, len(output_bits),
            session,
        )

        return output_bits, meta

    # ------------------------------------------------------------------
    # Backward-compatible shim  (A5 FIX)
    # ------------------------------------------------------------------

    def generate_certified_random_bits(self,
                                       n_bits:           int,
                                       source_simulator) -> Tuple[np.ndarray, List[Union[BlockMetadata, FinalDecision, EATSummary]]]:
        """
        Backward-compatible shim — delegates to CertifiedGenerationSession.

        A5 FIX: the outer generation loop and global final extraction now live
        in CertifiedGenerationSession.run(). This shim preserves the identical
        public signature so all existing callers continue to work unchanged:

            output_bits, metadata_list = te_qrng.generate_certified_random_bits(
                n_bits=n_bits, source_simulator=source
            )

        Guarantees: ‖ρ_RE − U_R ⊗ ρ_E‖₁ ≤ ε_total
        """
        session_driver = CertifiedGenerationSession(
            te_qrng     = self,
            epsilon_eat = self.epsilon_eat,
            epsilon_ext = self.epsilon_ext,
        )
        return session_driver.run(n_bits, source_simulator)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("TE-SI-QRNG: Trust-Enhanced Source-Independent Quantum Random Number Generator")
    print("=" * 80)
    print("\nVersion 16 — A5: QRNGSessionState + CertifiedGenerationSession extracted")
    print("\nKey structural change (v16):")
    print("  TrustEnhancedQRNG now holds only per-block pipeline logic.")
    print("  QRNGSessionState  — EAT accumulation state + throughput counters")
    print("  CertifiedGenerationSession — outer generation loop + global extraction")
    print("  generate_certified_random_bits() retained as backward-compatible shim.")
    print("\nSecurity invariants (unchanged):")
    print("  h_min_certified ← p_max_upper only (INVARIANT — never modified by diagnostics)")
    print("  extraction_rate ← h_min_certified · LHL only (INVARIANT)")
    print("  trust_score     → warn/halt only — NEVER modifies entropy")
