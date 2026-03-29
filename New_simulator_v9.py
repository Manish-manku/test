"""
Quantum Source Simulator for TE-SI-QRNG Testing
================================================

Simulates various quantum random number generation sources with
realistic imperfections, drift, and attack scenarios.
"""

import numpy as np
from typing import NamedTuple, Optional
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SourceType(Enum):
    """Types of quantum sources to simulate."""
    IDEAL                    = "ideal"
    BIASED                   = "biased"
    DRIFTING                 = "drifting"
    CORRELATED               = "correlated"
    ATTACKED                 = "attacked"
    NON_COOPERATIVE_ATTACKED = "non_cooperative_attacked"
    PHOTON_COUNTING          = "photon_counting"
    PHASE_NOISE              = "phase_noise"


# ---------------------------------------------------------------------------
# Change 6: Named result type instead of anonymous 3-tuple
# ---------------------------------------------------------------------------

class GeneratedBlock(NamedTuple):
    """Output of a single generate_block() call."""
    bits:       np.ndarray   # Binary output (0/1), shape (n_bits,)
    bases:      np.ndarray   # Measurement bases (0/1), shape (n_bits,)
    raw_signal: np.ndarray   # Underlying analog signal, shape (n_bits,)


# ---------------------------------------------------------------------------
# Change 3: Per-source parameter dataclasses instead of one monolithic class
# ---------------------------------------------------------------------------

@dataclass
class IdealParams:
    """Parameters for the ideal quantum source (no configuration needed)."""
    source_type: SourceType = field(default=SourceType.IDEAL, init=False)


@dataclass
class BiasedParams:
    """Parameters for a statically biased quantum source."""
    source_type: SourceType = field(default=SourceType.BIASED, init=False)
    bias: float = 0.05  # Probability shift toward 1, in range (0, 0.5)

    def __post_init__(self):
        if not (0.0 < self.bias < 0.5):
            raise ValueError(
                f"BiasedParams: bias={self.bias} must be in (0, 0.5). "
                "Values outside this range produce prob_one outside (0.5, 1.0)."
            )


@dataclass
class DriftingParams:
    """Parameters for a time-drifting quantum source."""
    source_type: SourceType = field(default=SourceType.DRIFTING, init=False)
    drift_rate: float = 0.1  # Controls amplitude of sinusoidal + linear drift

    def __post_init__(self):
        if self.drift_rate < 0.0:
            raise ValueError(
                f"DriftingParams: drift_rate={self.drift_rate} must be >= 0."
            )


@dataclass
class CorrelatedParams:
    """Parameters for a source with memory/correlation effects."""
    source_type: SourceType = field(default=SourceType.CORRELATED, init=False)
    correlation_length: int = 10  # Number of past bits influencing the current bit

    def __post_init__(self):
        if self.correlation_length < 1:
            raise ValueError(
                f"CorrelatedParams: correlation_length={self.correlation_length} must be >= 1."
            )


@dataclass
class AttackedParams:
    """Parameters for a source under adversarial attack."""
    source_type: SourceType = field(default=SourceType.ATTACKED, init=False)
    attack_strength: float = 0.1  # Fraction of bits forcibly set to 1

    def __post_init__(self):
        if not (0.0 <= self.attack_strength <= 1.0):
            raise ValueError(
                f"AttackedParams: attack_strength={self.attack_strength} must be in [0, 1]. "
                "Values above 1.0 force all bits to 1 silently."
            )


@dataclass
class NonCooperativeAttackedParams:
    """
    Parameters for a realistic non-cooperative adversarial attack.

    The adversary shifts the Gaussian mean of the quadrature signal by
    mu_attack, biasing P(bit=1) above 0.5 while keeping the signal
    distribution visually close to normal.

    This is the physically correct attack model for CV-QRNG:
        measured ~ N(mu_attack, sigma²)
        bit      = sign(measured)   [thresholding as normal]

    The signal is NOT post-hoc constructed to match forced bits.
    Eve controls the source mean, not individual bit values.

    Bug S5 fix: replaces the cooperative attack model where
    bit → signal (physically backwards and circular).
    """
    source_type: SourceType = field(default=SourceType.NON_COOPERATIVE_ATTACKED, init=False)
    mu_attack: float = 0.05   # Mean shift in quadrature units (σ units)
    sigma:     float = 1.0    # Standard deviation (shot noise = 1.0)

    def __post_init__(self):
        if abs(self.mu_attack) > 2.0 * self.sigma:
            raise ValueError(
                f"mu_attack={self.mu_attack} is too large relative to sigma={self.sigma}. "
                "A realistic non-cooperative adversary keeps the distribution near-normal. "
                "Use |mu_attack| ≤ 2·sigma."
            )
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")


@dataclass
class PhotonCountingParams:
    """Parameters for a photon-counting QRNG source."""
    source_type: SourceType = field(default=SourceType.PHOTON_COUNTING, init=False)
    detector_efficiency: float = 0.95  # Probability a photon triggers the detector
    dark_count_rate:     float = 0.001 # Probability of a spurious dark-count click


@dataclass
class PhaseNoiseParams:
    """Parameters for a phase-noise (homodyne/heterodyne) QRNG source."""
    source_type: SourceType = field(default=SourceType.PHASE_NOISE, init=False)
    noise_level: float = 0.1  # Technical noise added on top of vacuum fluctuations


# Union type alias for any valid parameter object
SourceParameters = (
    IdealParams | BiasedParams | DriftingParams | CorrelatedParams |
    AttackedParams | NonCooperativeAttackedParams |
    PhotonCountingParams | PhaseNoiseParams
)


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

class QuantumSourceSimulator:
    """
    Simulates a quantum random number source with configurable imperfections.

    Supported source types
    ----------------------
    IDEAL           — Perfect quantum randomness, 50/50 output.
    BIASED          — Static bias shifts P(1) above 0.5.
    DRIFTING        — Time-varying bias (temperature cycling + aging).
    CORRELATED      — Memory effects: current bit may repeat recent history.
    ATTACKED        — Adversarial bit forcing and pattern injection.
    PHOTON_COUNTING — Realistic single-photon detection with efficiency & dark counts.
    PHASE_NOISE     — Vacuum quadrature measurement (Gaussian source).
    """

    def __init__(self, params: SourceParameters, seed: Optional[int] = None):
        """
        Args:
            params: Source-specific parameter object (e.g. BiasedParams).
            seed:   Optional RNG seed for reproducibility.
        """
        self.params    = params
        self.rng       = np.random.RandomState(seed)
        self.time_step = 0

        # State used only by specific source types
        self._memory_buffer: list[int] = []  # Used by CORRELATED

        # Change 2: dispatch table replaces the long if/elif chain
        self._dispatch = {
            SourceType.IDEAL:                    self._generate_ideal,
            SourceType.BIASED:                   self._generate_biased,
            SourceType.DRIFTING:                 self._generate_drifting,
            SourceType.CORRELATED:               self._generate_correlated,
            SourceType.ATTACKED:                 self._generate_attacked,
            SourceType.NON_COOPERATIVE_ATTACKED: self._generate_non_cooperative_attacked,
            SourceType.PHOTON_COUNTING:          self._generate_photon_counting,
            SourceType.PHASE_NOISE:              self._generate_phase_noise,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_block(self, n_bits: int) -> GeneratedBlock:
        """
        Generate a block of quantum random bits with metadata.

        Args:
            n_bits: Number of bits to generate.

        Returns:
            GeneratedBlock(bits, bases, raw_signal)
        """
        generator = self._dispatch[self.params.source_type]
        return generator(n_bits)

    def reset(self) -> None:
        """Reset all stateful variables to their initial values."""
        self.time_step      = 0
        self._memory_buffer = []

    # ------------------------------------------------------------------
    # Physical-parameter accessors (used by TE-SI-QRNG drift monitor)
    # ------------------------------------------------------------------

    def get_efficiency(self) -> float:
        """
        Return the current detector efficiency for this source.

        For PHOTON_COUNTING sources this reflects the actual configured
        efficiency.  All other source types are treated as ideal detectors
        (efficiency = 1.0).
        """
        if isinstance(self.params, PhotonCountingParams):
            return self.params.detector_efficiency
        return 1.0

    def get_dark_count_rate(self) -> float:
        """
        Return the current dark-count rate for this source.

        For PHOTON_COUNTING sources this reflects the actual configured
        dark-count rate.  All other source types produce no dark counts
        (rate = 0.0).
        """
        if isinstance(self.params, PhotonCountingParams):
            return self.params.dark_count_rate
        return 0.0

    def get_signal_stats(self) -> tuple[float, float]:
        """
        Return the (expected_mean, expected_std) of raw_signal for this source.

        Used by run_self_tests to pass the correct physical baseline to
        energy_constraint_test().  Without this, the test compares every source
        against (0.0, 1.0) — correct only for IDEAL and PHASE_NOISE sources;
        wrong for BIASED, DRIFTING, ATTACKED, and PHOTON_COUNTING sources,
        which produce signals with different statistical baselines.

        Returns:
            (expected_mean, expected_std): The physically correct baseline
            for this source's raw_signal output.
        """
        p = self.params
        st = p.source_type

        if st == SourceType.IDEAL:
            # randn: N(0,1)
            return 0.0, 1.0

        if st == SourceType.BIASED:
            # randn + bias*2: N(bias*2, 1)
            return p.bias * 2.0, 1.0

        if st == SourceType.DRIFTING:
            # randn + drift*2, drift bounded by drift_rate * (1 + 1/100k)
            # Conservative: mean ≈ 0, std ≈ 1 (drift is small near t=0)
            return 0.0, 1.0

        if st == SourceType.CORRELATED:
            # randn: N(0,1) — signal is independent of the memory bits
            return 0.0, 1.0

        if st == SourceType.ATTACKED:
            # randn + attack_strength: N(attack_strength, 1)
            # Periodic forcing is also added: mean shifts by attack_strength
            return float(p.attack_strength), 1.0

        if st == SourceType.NON_COOPERATIVE_ATTACKED:
            # measured ~ N(mu_attack, sigma²) — this IS the physical baseline
            # The diagnostic should flag deviation from (0, 1) ideal
            return float(p.mu_attack), float(p.sigma)

        if st == SourceType.PHOTON_COUNTING:
            # Signal is in {0.0, 0.5, 1.0}.
            # P(detected)=η, P(dark)≈(1-η)·dc, P(missed)≈(1-η)·(1-dc)
            # mean ≈ η·1.0 + (1-η)·dc·0.5 + (1-η)·(1-dc)·0.0 = η + 0.5·(1-η)·dc
            eta = p.detector_efficiency
            dc  = p.dark_count_rate
            mean = eta * 1.0 + (1 - eta) * dc * 0.5
            # var  ≈ E[x²] - mean²; x∈{0,0.5,1} weighted by above probabilities
            e_x2 = eta * 1.0 + (1 - eta) * dc * 0.25
            std  = float(np.sqrt(max(e_x2 - mean**2, 1e-10)))
            return float(mean), std

        if st == SourceType.PHASE_NOISE:
            # x = randn + randn*noise_level → std = sqrt(1 + noise_level²)
            # Previously returned 1+noise_level which is wrong for noise_level > 0
            return 0.0, float(np.sqrt(1.0 + p.noise_level ** 2))

        # Fallback
        return 0.0, 1.0

    def _generate_ideal(self, n_bits: int) -> GeneratedBlock:
        """Perfect quantum randomness — 50/50, no correlations."""
        bits       = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)
        bases      = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)
        raw_signal = self.rng.randn(n_bits)

        self.time_step += n_bits
        return GeneratedBlock(bits, bases, raw_signal)

    def _generate_biased(self, n_bits: int) -> GeneratedBlock:
        """Static bias: P(1) = 0.5 + bias."""
        p          = self.params
        prob_one   = 0.5 + p.bias

        bits       = (self.rng.rand(n_bits) < prob_one).astype(np.uint8)
        bases      = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)
        raw_signal = self.rng.randn(n_bits) + p.bias * 2

        self.time_step += n_bits
        return GeneratedBlock(bits, bases, raw_signal)

    def _generate_drifting(self, n_bits: int) -> GeneratedBlock:
        """
        Time-varying bias: sinusoidal oscillation + linear aging ramp.

        Change 1: Fully vectorized — no Python-level loop.
        """
        p = self.params

        t      = np.arange(self.time_step, self.time_step + n_bits, dtype=float)
        drift  = p.drift_rate * (np.sin(2 * np.pi * t / 5000) + t / 100_000)
        probs  = np.clip(0.5 + drift, 0.01, 0.99)

        bits       = (self.rng.rand(n_bits) < probs).astype(np.uint8)
        bases      = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)
        raw_signal = self.rng.randn(n_bits) + drift * 2

        self.time_step += n_bits
        return GeneratedBlock(bits, bases, raw_signal)

    def _generate_correlated(self, n_bits: int) -> GeneratedBlock:
        """
        Memory effects: each bit may repeat its predecessor.

        Change 1: Vectorized using a pre-drawn decision array.
        The correlation decision (repeat vs. fresh) is drawn all at once;
        only the sequential dependency on the previous bit requires a small loop.
        """
        p = self.params
        correlation_strength = min(p.correlation_length / 100, 0.9)

        # Pre-draw all randomness up front
        use_memory   = self.rng.rand(n_bits) < correlation_strength
        fresh_bits   = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)
        raw_signal   = self.rng.randn(n_bits)
        bases        = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)

        bits = np.empty(n_bits, dtype=np.uint8)
        last = self._memory_buffer[-1] if self._memory_buffer else fresh_bits[0]

        for i in range(n_bits):
            bit    = last if use_memory[i] else fresh_bits[i]
            bits[i] = bit
            last   = bit

        # Update memory buffer (keep only last correlation_length entries)
        self._memory_buffer.extend(bits.tolist())
        self._memory_buffer = self._memory_buffer[-p.correlation_length:]

        self.time_step += n_bits
        return GeneratedBlock(bits, bases, raw_signal)

    def _generate_attacked(self, n_bits: int) -> GeneratedBlock:
        """
        Cooperative adversarial attack: random bit-forcing + periodic pattern injection.

        COOPERATIVE MODEL — the adversary encodes the attack directly into
        raw_signal (signal spikes at forced positions). This makes the attack
        visible to energy_constraint_test() and is useful for testing that the
        diagnostic layer responds to obvious interference.

        For a realistic non-cooperative attack (adversary hides in the signal
        distribution), use NonCooperativeAttackedParams and
        _generate_non_cooperative_attacked() instead.
        """
        p = self.params

        bits = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)

        # Random forcing: set chosen bits to 1
        attack_mask       = self.rng.rand(n_bits) < p.attack_strength
        bits[attack_mask] = 1

        # Periodic pattern injection every 100 bits
        pattern_width = max(1, int(p.attack_strength * 10))
        starts        = np.arange(0, n_bits, 100)
        ends          = np.minimum(starts + pattern_width, n_bits)

        # Build a forcing indicator array (1 where forced, 0 elsewhere)
        forcing = np.zeros(n_bits, dtype=float)
        for s, e in zip(starts, ends):
            bits[s:e]    = 1
            forcing[s:e] = 1.0   # mark forced positions in signal

        bases = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)

        # raw_signal: base noise + constant attack offset + periodic forcing spike.
        # The forcing spike raises local signal values and inflates std beyond the
        # expected baseline (attack_strength, 1.0), making the attack visible to
        # energy_constraint_test via an elevated std_deviation score.
        raw_signal = self.rng.randn(n_bits) + p.attack_strength + forcing

        self.time_step += n_bits
        return GeneratedBlock(bits, bases, raw_signal)

    def _generate_non_cooperative_attacked(self, n_bits: int) -> GeneratedBlock:
        """
        Realistic non-cooperative adversarial attack for CV-QRNG.

        Bug S5 fix — physically correct order:
            signal → measurement → bit      (was: bit → signal, backwards)

        The adversary shifts the Gaussian mean of the quadrature measurement
        by mu_attack, biasing P(bit=1) above 0.5. Crucially:
            - The signal distribution remains visually close to N(0, sigma²)
            - Eve does NOT reveal the attack in the signal shape
            - The diagnostic must detect subtle mean shift, not obvious forcing

        Physical model:
            measured_i ~ N(mu_attack, sigma²)    [Eve controls source mean]
            bit_i       = 1 if measured_i > 0    [honest threshold, unchanged]
            raw_signal  = measured_i              [pre-threshold amplitude]

        This is the correct test of the pre-value gating framework:
        can Φ_local detect asymmetry when signal looks nearly normal?

        Expected P(bit=1) = Φ_cdf(mu_attack / sigma) where Φ_cdf is
        the standard normal CDF. For mu_attack=0.05, sigma=1.0:
            P(bit=1) ≈ 0.520  (subtle but real bias)
        """
        p = self.params

        # Step 1: Generate quadrature amplitudes from shifted Gaussian
        # This is the ONLY place Eve's influence enters — the mean, not the bits
        measured = self.rng.normal(loc=p.mu_attack, scale=p.sigma, size=n_bits)

        # Step 2: Honest thresholding — bit follows from signal, not vice versa
        bits = (measured > 0).astype(np.uint8)

        # Step 3: Bases chosen independently of signal
        bases = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)

        # Step 4: raw_signal IS the pre-threshold measurement
        # No post-hoc construction — this is genuinely pre-value
        raw_signal = measured

        self.time_step += n_bits
        return GeneratedBlock(bits, bases, raw_signal)

    def _generate_photon_counting(self, n_bits: int) -> GeneratedBlock:
        """
        Realistic single-photon detection.

        Change 1: Vectorized using masks for detected / dark-count / missed cases.
        Three outcomes per slot:
          detected (efficiency)  → true quantum bit, raw = 1.0
          dark count (rare)      → random bit,       raw = 0.5
          missed                 → random bit,       raw = 0.0
        """
        p = self.params

        true_bits  = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)
        detected   = self.rng.rand(n_bits) < p.detector_efficiency
        dark_count = (~detected) & (self.rng.rand(n_bits) < p.dark_count_rate)
        missed     = ~detected & ~dark_count

        bits       = np.empty(n_bits, dtype=np.uint8)
        raw_signal = np.zeros(n_bits)

        bits[detected]   = true_bits[detected];  raw_signal[detected]   = 1.0
        bits[dark_count] = self.rng.randint(0, 2, size=dark_count.sum()); raw_signal[dark_count] = 0.5
        bits[missed]     = self.rng.randint(0, 2, size=missed.sum());     raw_signal[missed]     = 0.0

        bases = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)

        self.time_step += n_bits
        return GeneratedBlock(bits, bases, raw_signal)

    def _generate_phase_noise(self, n_bits: int) -> GeneratedBlock:
        """
        Vacuum quadrature measurement (homodyne / heterodyne QRNG).

        Change 1: Vectorized using NumPy where() instead of a per-bit loop.
        """
        p = self.params

        x = self.rng.randn(n_bits) + self.rng.randn(n_bits) * p.noise_level
        q = self.rng.randn(n_bits) + self.rng.randn(n_bits) * p.noise_level

        bases        = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)
        measured     = np.where(bases == 0, x, q)          # Choose quadrature
        bits         = (measured > 0).astype(np.uint8)     # Sign → bit

        self.time_step += n_bits
        return GeneratedBlock(bits, bases, measured)


# ---------------------------------------------------------------------------
# Attack scenario wrapper
# ---------------------------------------------------------------------------

class AttackScenarioSimulator:
    """
    Applies adversarial transformations on top of any QuantumSourceSimulator.

    Attack types
    ------------
    source_tampering        — Replace a fraction of bits with an alternating pattern.
    side_channel_injection  — Inject a sinusoidal EM signal that biases detections.
    """

    def __init__(self, base_source: QuantumSourceSimulator):
        self.base_source = base_source

    # ------------------------------------------------------------------
    # Physical-parameter accessors — delegate to the wrapped source
    # ------------------------------------------------------------------

    def get_efficiency(self) -> float:
        """Delegate to the wrapped base source."""
        return self.base_source.get_efficiency()

    def get_dark_count_rate(self) -> float:
        """Delegate to the wrapped base source."""
        return self.base_source.get_dark_count_rate()

    def get_signal_stats(self) -> tuple[float, float]:
        """
        Delegate to the wrapped base source.

        Note: attack transformations may shift the actual signal stats
        (e.g. side_channel_injection adds a sinusoidal term that raises std).
        The delegated baseline is still the correct *physical expectation* —
        deviation from it is precisely what the energy_constraint_test should flag.
        """
        return self.base_source.get_signal_stats()

    def source_tampering_attack(
        self,
        n_bits:      int,
        tamper_rate: float = 0.1,
    ) -> GeneratedBlock:
        """
        Replace `tamper_rate` fraction of bits with an alternating 0,1,0,1 pattern.

        Args:
            n_bits:      Number of bits to generate.
            tamper_rate: Fraction of bits the adversary replaces (0–1).
        """
        block = self.base_source.generate_block(n_bits)
        bits  = block.bits.copy()

        tamper_mask           = self.base_source.rng.rand(n_bits) < tamper_rate
        alternating           = (np.arange(n_bits) % 2).astype(np.uint8)
        bits[tamper_mask]     = alternating[tamper_mask]

        return GeneratedBlock(bits, block.bases, block.raw_signal)

    def side_channel_injection_attack(
        self,
        n_bits:             int,
        injection_strength: float = 0.2,
    ) -> GeneratedBlock:
        """
        Inject a 50-bit-period sinusoidal EM signal that biases the detector.

        The signal is visible in raw_signal, allowing detection by a vigilant
        self-testing system even when the bit-level bias is subtle.

        Args:
            n_bits:             Number of bits to generate.
            injection_strength: Peak amplitude of the injected signal (0–1).
        """
        block      = self.base_source.generate_block(n_bits)
        bits       = block.bits.copy()
        raw_signal = block.raw_signal.copy()

        injection  = injection_strength * np.sin(2 * np.pi * np.arange(n_bits) / 50)
        bias_mask  = (injection > 0) & (self.base_source.rng.rand(n_bits) < np.abs(injection))
        bits[bias_mask]  = 1
        raw_signal      += injection

        return GeneratedBlock(bits, block.bases, raw_signal)


# ---------------------------------------------------------------------------
# Standard test scenarios
# ---------------------------------------------------------------------------

def create_test_scenarios() -> dict[str, SourceParameters]:
    """
    Return 14 pre-built SourceParameter objects for standard TE-SI-QRNG evaluation.

    Covers: ideal, biased, drifting, correlated, attacked (cooperative + non-cooperative),
            photon-counting, and phase-noise sources.

    v8 additions:
        non_cooperative_weak  — subtle mean shift (mu=0.05), hard to detect
        non_cooperative_strong — larger mean shift (mu=0.20), moderately visible
    """
    return {
        "ideal":                  IdealParams(),
        "small_bias":             BiasedParams(bias=0.05),
        "large_bias":             BiasedParams(bias=0.2),
        "temporal_drift":         DriftingParams(drift_rate=0.1),
        "short_correlation":      CorrelatedParams(correlation_length=10),
        "long_correlation":       CorrelatedParams(correlation_length=100),
        "weak_attack":            AttackedParams(attack_strength=0.1),
        "strong_attack":          AttackedParams(attack_strength=0.3),
        "non_cooperative_weak":   NonCooperativeAttackedParams(mu_attack=0.05, sigma=1.0),
        "non_cooperative_strong": NonCooperativeAttackedParams(mu_attack=0.20, sigma=1.0),
        "realistic_photon":       PhotonCountingParams(detector_efficiency=0.90, dark_count_rate=0.001),
        "degraded_photon":        PhotonCountingParams(detector_efficiency=0.70, dark_count_rate=0.010),
        "phase_noise_clean":      PhaseNoiseParams(noise_level=0.1),
        "phase_noise_noisy":      PhaseNoiseParams(noise_level=0.5),
    }


# ---------------------------------------------------------------------------
# Change 4: Demo logic moved into a proper function
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """Quick sanity-check: generate from each source and print summary stats."""
    scenarios = create_test_scenarios()

    print("Quantum Source Simulator for TE-SI-QRNG")
    print("=" * 60)
    print(f"\nAvailable test scenarios: {len(scenarios)}")
    for name, params in scenarios.items():
        print(f"  {name:<22} → {params.source_type.value}")

    # --- Ideal source ---
    print("\n[ Ideal source | n=1 000 ]")
    src   = QuantumSourceSimulator(scenarios["ideal"], seed=42)
    block = src.generate_block(1_000)
    print(f"  mean  : {np.mean(block.bits):.4f}  (expected ≈ 0.5000)")
    print(f"  std   : {np.std(block.bits.astype(float)):.4f}  (expected ≈ 0.5000)")

    # --- Biased source ---
    print("\n[ Large-bias source (bias=0.2) | n=10 000 ]")
    src   = QuantumSourceSimulator(scenarios["large_bias"], seed=42)
    block = src.generate_block(10_000)
    print(f"  mean  : {np.mean(block.bits):.4f}  (expected ≈ 0.7000)")
    print(f"  |bias|: {abs(np.mean(block.bits) - 0.5):.4f}")

    # --- Photon counting ---
    print("\n[ Realistic photon-counting source | n=5 000 ]")
    src   = QuantumSourceSimulator(scenarios["realistic_photon"], seed=42)
    block = src.generate_block(5_000)
    detected    = np.sum(block.raw_signal == 1.0)
    dark_counts = np.sum(block.raw_signal == 0.5)
    print(f"  detected   : {detected}  ({100*detected/5000:.1f}%)")
    print(f"  dark counts: {dark_counts}")

    # --- Named-tuple access demo ---
    print("\n[ GeneratedBlock named-tuple access ]")
    src   = QuantumSourceSimulator(scenarios["phase_noise_clean"], seed=0)
    block = src.generate_block(500)
    print(f"  block.bits[:8]       = {block.bits[:8].tolist()}")
    print(f"  block.bases[:8]      = {block.bases[:8].tolist()}")
    print(f"  block.raw_signal[:4] = {np.round(block.raw_signal[:4], 3).tolist()}")


if __name__ == "__main__":
    run_demo()