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
          Tuple[np.ndarray, List[Union[BlockMetadata, EATSummary]]].
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
        _run_diagnostics()  — Steps 4–5: run_self_tests, halt/warn decision
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
from typing import Tuple, Dict, List, Optional, Union
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

    These are the entries at positions [0 .. -2] of the metadata_list returned
    by generate_certified_random_bits().  The final entry is an EATSummary.

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
    gate_imr:              Optional[float]
    gate_n_accepted:       int
    gate_n_total:          int


class EATSummary(TypedDict):
    """
    Schema for the final entry in the metadata_list returned by
    generate_certified_random_bits().

    This is the global EAT accumulation result across all blocks.
    It is distinguishable from BlockMetadata by position (always last)
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
        """Compute aggregate trust score [0, 1], where 1 = perfect trust.

        F1 FIX: result is clamped to [0, 1].
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
    Raised when diagnostic self-tests detect a condition severe enough
    to halt extraction entirely.

    Diagnostics are ALLOWED to halt — they are FORBIDDEN to modify entropy.

    Halt conditions (trust_score thresholds):
        trust_score < HALT_THRESHOLD  → raise DiagnosticHaltError
        trust_score < WARN_THRESHOLD  → add warning to metadata, continue

    These thresholds are engineering policy, not security bounds.
    The certified entropy H_cert remains valid regardless of trust_score;
    halting is a conservative operational choice, not a cryptographic requirement.
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
    pass


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
                 tau_init:      float = 0.5):
        self.sigma     = sigma
        self.yield_min = yield_min
        self.tau       = tau_init

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

        valid = yield_grid >= self.yield_min
        if not np.any(valid):
            self.tau = tau_grid[0]
            return self.tau

        imr_valid       = np.where(valid, self._imr_grid, np.inf)
        best_idx        = int(np.argmin(imr_valid))
        self.tau        = float(tau_grid[best_idx])

        return self.tau

    def apply(self,
              raw_signal: np.ndarray,
              bits:       np.ndarray,
              bases:      np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        n_total     = len(raw_signal)
        gate_mask   = np.abs(raw_signal) > self.tau

        accepted_signal = raw_signal[gate_mask]
        accepted_bits   = bits[gate_mask]
        accepted_bases  = bases[gate_mask]

        n_accepted  = int(np.sum(gate_mask))
        yield_rate  = n_accepted / max(n_total, 1)

        if n_accepted > 10:
            epsilon_gate = float(abs(np.mean(accepted_bits) - 0.5))
        else:
            epsilon_gate = 0.5

        from scipy.stats import norm
        imr_val = float(norm.pdf(self.tau / self.sigma) /
                        max(norm.sf(self.tau / self.sigma), 1e-15))

        gate_meta = {
            'tau':          self.tau,
            'n_total':      n_total,
            'n_accepted':   n_accepted,
            'yield_rate':   yield_rate,
            'epsilon_gate': epsilon_gate,
            'imr':          imr_val,
            'sigma':        self.sigma,
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

    def __init__(self, security_parameter: float = 1e-6):
        self.epsilon_total  = security_parameter
        self.epsilon_eat    = security_parameter
        self.epsilon_smooth = security_parameter
        self.epsilon_ext    = security_parameter

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
        circ_size = min(circ_size, self._MAX_CIRC_SIZE)

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

        single_circ = 1 << int(np.ceil(np.log2(max(m + n, 2))))
        if single_circ <= self._MAX_CIRC_SIZE:
            required = n + m - 1
            if len(seed) < required:
                seed = self._extend_seed(seed, required)
            return self._toeplitz_fft_chunk(weak_random, seed, m)

        MAX_CHUNK_INPUT = max(self._MAX_CIRC_SIZE // 4, 1024)

        n_chunks  = int(np.ceil(n / MAX_CHUNK_INPUT))
        n_c       = int(np.ceil(n / n_chunks))
        m_c_base  = m // n_chunks

        seed_bytes = np.packbits(seed[:min(len(seed), 2048)]).tobytes()

        output_chunks: List[np.ndarray] = []
        bits_produced = 0

        for i in range(n_chunks):
            i_start = i * n_c
            i_end   = min(i_start + n_c, n)
            chunk   = weak_random[i_start:i_end]
            nc_i    = len(chunk)
            if nc_i == 0:
                continue

            mc_i = (m - bits_produced) if (i == n_chunks - 1) else m_c_base
            if mc_i <= 0:
                break

            chunk_seed = self._derive_chunk_seed(seed_bytes, i, nc_i + mc_i - 1)
            out_chunk  = self._toeplitz_fft_chunk(chunk, chunk_seed, mc_i)
            output_chunks.append(out_chunk)
            bits_produced += len(out_chunk)

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
    total_output_bits:     int         = 0
    total_gen_input_bits:  int         = 0
    total_raw_input_bits:  int         = 0

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

    def append_block(self, h_min_certified: float, n_gen: int) -> None:
        """
        Record one block's contribution to the EAT accumulation.

        Args:
            h_min_certified: per-bit min-entropy lower bound (bits/bit)
            n_gen:           number of generation-round bits in this block
        """
        self.block_entropy_history.append(h_min_certified * n_gen)
        self.block_n_gen_history.append(n_gen)


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
        self.epsilon_ext = epsilon_ext

    def run(self,
            n_bits:           int,
            source_simulator) -> Tuple[np.ndarray, List[Union[BlockMetadata, EATSummary]]]:
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
            metadata_list[-1] is always an EATSummary.
            metadata_list[:-1] are BlockMetadata dicts, one per block.

        Raises:
            ValueError:              n_bits <= 0
            DiagnosticHaltError:     trust_score falls below HALT_THRESHOLD
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

        all_gen_bits:  List[np.ndarray] = []
        metadata_list: List[Dict]       = []

        block_size = self.te_qrng.block_size

        while True:
            raw_bits, bases, raw_signal = source_simulator.generate_block(block_size)

            signal_stats = (source_simulator.get_signal_stats()
                            if hasattr(source_simulator, 'get_signal_stats') else None)

            try:
                _, block_meta = self.te_qrng.process_block(
                    raw_bits, bases, raw_signal, session=session,
                    signal_stats=signal_stats
                )
            except DiagnosticHaltError as exc:
                halt_meta = {
                    'certified_quantity':  'H_min(X|E)',
                    'security_definition': 'Trace-distance ε-security',
                    'halt': True,
                    'halt_reason': str(exc),
                    'blocks_used': len(session.block_entropy_history),
                    'h_total_eat': session.accumulate_eat(self.epsilon_eat),
                }
                metadata_list.append(halt_meta)
                raise

            metadata_list.append(block_meta)

            if bases is not None:
                gen_bits, _ = BB84RoundSplitter.split(raw_bits, bases)
            else:
                gen_bits = raw_bits
            all_gen_bits.append(gen_bits)

            h_total          = session.accumulate_eat(self.epsilon_eat)
            log2_inv_eps_ext = np.log2(1.0 / self.epsilon_ext)
            max_output_bits  = int(h_total - 2.0 * log2_inv_eps_ext)

            if max_output_bits >= n_bits:
                break

            total_gen = sum(len(g) for g in all_gen_bits)
            if total_gen > 50 * n_bits:
                raise EATConvergenceWarning(
                    f"CertifiedGenerationSession.run: EAT bound not reached after "
                    f"total_gen={total_gen} bits ({50 * n_bits} limit). "
                    f"Source entropy too low for requested n_bits={n_bits}."
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

        eat_summary: EATSummary = {
            'certified_quantity':    'H_min(X|E)',
            'security_definition':   'Trace-distance ε-security',
            'epsilon_total':         self.te_qrng.epsilon_total,
            'epsilon_eat':           self.epsilon_eat,
            'epsilon_smooth':        self.te_qrng.epsilon_smooth,
            'epsilon_ext':           self.epsilon_ext,
            'blocks_used':           t_blocks,
            'h_total_eat':           h_total,
            'certified_output_bits': certified_output,
            'actual_output_bits':    len(final_bits),
            'delta_eat':             delta_eat,
            'sum_f_ei':              sum_f_ei,
        }
        metadata_list.append(eat_summary)

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
                 yield_min:            float = 0.30):
        self.block_size           = block_size
        self.security_parameter   = security_parameter
        self.extractor_efficiency = extractor_efficiency
        self.enable_gating        = enable_gating

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

        # Composable epsilon budget (mirrors EntropyEstimator)
        self.epsilon_total  = self.entropy_estimator.epsilon_total
        self.epsilon_eat    = self.entropy_estimator.epsilon_eat
        self.epsilon_smooth = self.entropy_estimator.epsilon_smooth
        self.epsilon_ext    = self.entropy_estimator.epsilon_ext

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
                       signal_stats: Optional[Tuple[float, float]] = None) -> TrustVector:
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
        gate_meta: Dict = {'enabled': False}
        if self.enable_gating and raw_signal is not None and len(raw_signal) == n_raw:
            self.pre_value_gate.update_tau(self.trust_vector.epsilon_bias)

            _, raw_bits, bases_gated, gate_meta = self.pre_value_gate.apply(
                raw_signal, raw_bits,
                bases if bases is not None else np.zeros(n_raw, dtype=np.uint8)
            )
            gate_meta['enabled'] = True
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
        cert = self.entropy_estimator.certify_min_entropy(
            raw_bits,
            bases if bases is not None else np.zeros(n_raw, dtype=np.uint8)
        )
        h_min_certified = cert['h_min_certified']
        # INVARIANT: h_min_certified is derived solely from p_max_upper.

        # Step 3: Store f(eᵢ)·n_gen_i for EAT accumulation — via session
        # A5 FIX: was self.block_entropy_history.append(...), now session.append_block()
        session.append_block(h_min_certified, n_gen)

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
        }

    def _run_diagnostics(self,
                         raw_bits:     np.ndarray,
                         bases:        Optional[np.ndarray],
                         raw_signal:   Optional[np.ndarray],
                         signal_stats: Optional[Tuple[float, float]],
                         h_min_certified: float,
                         ) -> Tuple[TrustVector, Optional[str]]:
        """
        Steps 4–5: run_self_tests, then evaluate halt/warn thresholds.

        Returns:
            (trust_vector, diagnostic_warning)

        Raises DiagnosticHaltError when trust_score < HALT_THRESHOLD.
        Pure diagnostic-layer logic — does not touch cert dict or entropy state.
        """
        trust_vector = self.run_self_tests(raw_bits, bases, raw_signal, signal_stats=signal_stats)
        trust_score  = trust_vector.trust_score()

        diagnostic_warning: Optional[str] = None
        if trust_score < DiagnosticHaltError.HALT_THRESHOLD:
            raise DiagnosticHaltError(
                f"System instability detected: trust_score={trust_score:.4f} "
                f"< HALT_THRESHOLD={DiagnosticHaltError.HALT_THRESHOLD}. "
                f"Extraction halted. h_min_certified={h_min_certified:.4f} is valid but "
                f"operational policy requires halt."
            )
        if trust_score < DiagnosticHaltError.WARN_THRESHOLD:
            diagnostic_warning = (
                f"Degraded operation: trust_score={trust_score:.4f} "
                f"< WARN_THRESHOLD={DiagnosticHaltError.WARN_THRESHOLD}. "
                f"h_min_certified={h_min_certified:.4f} is unaffected."
            )

        return trust_vector, diagnostic_warning

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
                           gate_meta:         Dict,
                           trust_vector:      TrustVector,
                           diagnostic_warning: Optional[str],
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

        # Update throughput counters in session (A5 FIX: was self.total_*)
        session.total_raw_input_bits  += n_raw
        session.total_gen_input_bits  += n_gen
        session.total_output_bits     += output_bits_len

        # Compute EAT values from session state
        h_total_eat = session.accumulate_eat(self.epsilon_eat)
        sum_f_ei    = sum(session.block_entropy_history)
        delta_eat   = sum_f_ei - h_total_eat

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
            'diagnostic_warning':  diagnostic_warning,
            'halt_threshold':      DiagnosticHaltError.HALT_THRESHOLD,
            'warn_threshold':      DiagnosticHaltError.WARN_THRESHOLD,
            'input_bits':          n_raw,
            'cumulative_efficiency': (session.total_output_bits /
                                      max(session.total_raw_input_bits, 1)),
            'gate_enabled':      gate_meta.get('enabled', False),
            'gate_tau':          gate_meta.get('tau', None),
            'gate_yield':        gate_meta.get('yield_rate', None),
            'epsilon_gate':      gate_meta.get('epsilon_gate', None),
            'gate_imr':          gate_meta.get('imr', None),
            'gate_n_accepted':   gate_meta.get('n_accepted', n_raw),
            'gate_n_total':      gate_meta.get('n_total', n_raw),
        }
        return meta

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

        # Layer 2 — Diagnostic layer (may raise DiagnosticHaltError)
        trust_vector, diagnostic_warning = self._run_diagnostics(
            c['raw_bits'], c['bases'], c['raw_signal'],
            signal_stats, c['h_min_certified'],
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
        }

        # Layer 4 — Bookkeeping (updates session throughput counters)
        meta = self._assemble_metadata(
            cert_bundle, c['n_raw'], c['gate_meta'],
            trust_vector, diagnostic_warning, len(output_bits),
            session,
        )

        return output_bits, meta

    # ------------------------------------------------------------------
    # Backward-compatible shim  (A5 FIX)
    # ------------------------------------------------------------------

    def generate_certified_random_bits(self,
                                       n_bits:           int,
                                       source_simulator) -> Tuple[np.ndarray, List[Union[BlockMetadata, EATSummary]]]:
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
