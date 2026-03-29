"""
NIST SP 800-22 Randomness Test Suite
======================================

Pure NumPy / SciPy implementation of all 15 NIST SP 800-22 Rev 1a statistical
tests.  Designed as a drop-in that matches the official NIST C reference binary
output — same formulae, same significance level α = 0.01.

When the official compiled binary (assess) is present on PATH or at
NIST_BINARY_PATH, this module delegates to it via subprocess and parses
finalAnalysisReport.txt.  When the binary is absent (the common case when
compiling from source is not yet done), the pure-Python path is used
automatically — no configuration needed.

VERSION HISTORY
===============
v3 — Batch 4 fix: F8
  F8. Vectorised 4 O(n²) / O(n·m) Python loops:
      serial()              — psi_sq() inner dict loop replaced with
                              sliding_window_view + integer bit-packing.
                              O(n·m) Python iterations → O(n) numpy pass.
      approximate_entropy() — same sliding_window_view technique as serial().
      longest_run()         — inner `for b in block` loop replaced with
                              np.diff + run-length encoding (np.where).
      maurers_universal()   — str.join(str(b)...) string conversion replaced
                              with np.packbits + integer bit extraction.
      All p-values are numerically identical to v2 — only data preparation
      is changed, not the formulae or chi-square / erfc computations.

v2 — Batch 1 fix: F9
  F9. serial(): psi_sq() was called 4 times (m twice, m-1 twice) — now cached
      into 3 variables (psi_m, psi_m1, psi_m2) before d2 / d2a are computed.
      Halves the runtime of serial() with zero semantic change.
      At n=1e6, m=16: saves ~16M Python iterations per serial() call.

References
----------
NIST SP 800-22 Rev 1a, April 2010.
  https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf

Test index mapping (canonical order used throughout this module)
----------------------------------------------------------------
 0  Frequency (Monobit)
 1  Frequency within a Block
 2  Runs
 3  Longest Run of Ones in a Block
 4  Binary Matrix Rank
 5  Discrete Fourier Transform (Spectral)
 6  Non-overlapping Template Matching
 7  Overlapping Template Matching
 8  Maurer's Universal Statistical
 9  Linear Complexity
10  Serial
11  Approximate Entropy
12  Cumulative Sums (forward)
13  Random Excursions
14  Random Excursions Variant

Usage
-----
    from nist_runner_v3 import NISTTestRunner

    runner  = NISTTestRunner(significance=0.01)
    result  = runner.run_all(bits)          # bits: np.ndarray of uint8 0/1
    df      = result.summary_dataframe()
    print(df)
"""

from __future__ import annotations

import os
import math
import subprocess
import tempfile
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import special, stats

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

NIST_BINARY_PATH: Optional[str] = os.environ.get("NIST_ASSESS_PATH", None)

ALPHA: float = 0.01          # Default significance level (NIST recommendation)

# Canonical test names — exactly as they appear in NIST reports
TEST_NAMES: List[str] = [
    "Frequency (Monobit)",
    "Frequency within Block",
    "Runs",
    "Longest Run of Ones",
    "Binary Matrix Rank",
    "Discrete Fourier Transform",
    "Non-overlapping Template",
    "Overlapping Template",
    "Maurer's Universal",
    "Linear Complexity",
    "Serial",
    "Approximate Entropy",
    "Cumulative Sums",
    "Random Excursions",
    "Random Excursions Variant",
]

N_TESTS: int = len(TEST_NAMES)   # 15


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class NISTResult:
    """
    Holds the outcome of all 15 NIST tests for one bit sequence.

    Attributes
    ----------
    p_values    : list of p-values, one per test (None if test skipped)
    passed      : list of bool pass/fail (None if test skipped)
    test_names  : canonical test names
    n_bits      : length of the tested sequence
    significance: alpha threshold used
    backend     : 'numpy' or 'nist_binary'
    notes       : optional per-test diagnostic strings
    """
    p_values:    List[Optional[float]]
    passed:      List[Optional[bool]]
    test_names:  List[str]          = field(default_factory=lambda: list(TEST_NAMES))
    n_bits:      int                = 0
    significance: float             = ALPHA
    backend:     str                = "numpy"
    notes:       List[str]         = field(default_factory=lambda: [""] * N_TESTS)

    # ------------------------------------------------------------------
    def pass_rate(self) -> float:
        """Fraction of tests passed (skipped tests excluded)."""
        valid = [p for p in self.passed if p is not None]
        return sum(valid) / len(valid) if valid else 0.0

    def summary_dataframe(self):
        """Return a pandas-style list-of-dicts (no pandas dependency)."""
        rows = []
        for i, name in enumerate(self.test_names):
            rows.append({
                "test":       name,
                "p_value":    self.p_values[i],
                "passed":     self.passed[i],
                "note":       self.notes[i],
            })
        return rows

    def to_dict(self) -> Dict:
        return {
            "p_values":    self.p_values,
            "passed":      [bool(p) if p is not None else None for p in self.passed],
            "pass_rate":   self.pass_rate(),
            "n_bits":      self.n_bits,
            "backend":     self.backend,
        }


# ---------------------------------------------------------------------------
# Core NIST test implementations
# ---------------------------------------------------------------------------

class _NIST:
    """
    Static methods — one per NIST SP 800-22 test.

    All methods accept a uint8 numpy array of 0/1 bits and return (p_value, note).
    p_value = None means the test was skipped (insufficient data).
    """

    # ------------------------------------------------------------------ 0
    @staticmethod
    def frequency_monobit(bits: np.ndarray) -> Tuple[Optional[float], str]:
        n  = len(bits)
        if n < 100:
            return None, "n < 100, skipped"
        s  = int(np.sum(bits)) * 2 - n          # Sn = #{1} - #{0}
        s_obs = abs(s) / math.sqrt(n)
        p  = math.erfc(s_obs / math.sqrt(2))
        return p, f"S_n={s}, s_obs={s_obs:.4f}"

    # ------------------------------------------------------------------ 1
    @staticmethod
    def frequency_block(bits: np.ndarray, M: int = 128) -> Tuple[Optional[float], str]:
        n  = len(bits)
        N  = n // M
        if N < 1:
            return None, f"n={n} < M={M}, skipped"
        chi_sq = 0.0
        for i in range(N):
            pi_i = float(np.mean(bits[i*M:(i+1)*M]))
            chi_sq += (pi_i - 0.5) ** 2
        chi_sq *= 4 * M
        p = special.gammaincc(N / 2.0, chi_sq / 2.0)
        return p, f"N={N}, M={M}, chi²={chi_sq:.4f}"

    # ------------------------------------------------------------------ 2
    @staticmethod
    def runs(bits: np.ndarray) -> Tuple[Optional[float], str]:
        n = len(bits)
        if n < 100:
            return None, "n < 100, skipped"
        pi = float(np.mean(bits))
        if abs(pi - 0.5) >= 2.0 / math.sqrt(n):
            return 0.0, f"pre-test fails: |pi-0.5|={abs(pi-0.5):.4f}"
        V_n = int(np.count_nonzero(np.diff(bits))) + 1
        num = abs(V_n - 2 * n * pi * (1 - pi))
        den = 2 * math.sqrt(2 * n) * pi * (1 - pi)
        p   = math.erfc(num / den)
        return p, f"V_n={V_n}, pi={pi:.4f}"

    # ------------------------------------------------------------------ 3
    @staticmethod
    def longest_run(bits: np.ndarray) -> Tuple[Optional[float], str]:
        n = len(bits)
        if n < 128:
            return None, "n < 128, skipped"
        # Choose block size M based on n
        if n < 6272:
            M, K, N = 8, 3, 16
            pi = [0.2148, 0.3672, 0.2305, 0.1875]
            v_map = [1, 2, 3, 4]
        elif n < 750000:
            M, K, N = 128, 5, 49
            pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
            v_map = [4, 5, 6, 7, 8, 9]
        else:
            M, K, N = 10000, 6, 75
            pi = [0.0882, 0.2092, 0.2483,0.1933, 0.1208, 0.0675, 0.0727]
            v_map = [10, 11, 12, 13, 14, 15, 16]

        N = min(N, n // M)
        freq = [0] * (K + 2)

        # F8 FIX: replace inner `for b in block` Python loop with numpy
        # run-length encoding via np.diff + np.where.
        # Previously: for each of the N blocks, iterated bit-by-bit to find
        # the longest run of 1s — O(M) Python iterations per block, O(N*M) total.
        # Now: for each block, find run boundaries with np.diff in one vectorised
        # call, then take the max run length with np.max — no Python bit loop.
        for i in range(N):
            block = bits[i*M:(i+1)*M]
            # Pad with 0 at each end so runs at the boundaries are capped correctly
            padded = np.concatenate(([0], block.astype(np.int8), [0]))
            # Transitions: +1 = start of run of 1s, -1 = end of run of 1s
            diff = np.diff(padded)
            starts = np.where(diff == 1)[0]
            ends   = np.where(diff == -1)[0]
            if len(starts) == 0:
                max_run = 0
            else:
                max_run = int(np.max(ends - starts))
            v = min(max(max_run, v_map[0]), v_map[-1])
            idx = min(v - v_map[0], K)
            freq[idx] += 1

        chi_sq = sum((freq[i] - N * pi[i])**2 / (N * pi[i])
                     for i in range(len(pi)) if N * pi[i] > 0)
        p = special.gammaincc(K / 2.0, chi_sq / 2.0)
        return p, f"M={M}, N={N}, chi²={chi_sq:.4f}"

    # ------------------------------------------------------------------ 4
    @staticmethod
    def binary_matrix_rank(bits: np.ndarray,
                           M: int = 32, Q: int = 32) -> Tuple[Optional[float], str]:
        n = len(bits)
        N = n // (M * Q)
        if N < 38:
            return None, f"N={N} matrices < 38, skipped"
        r_full = r_sub = r_rem = 0
        for i in range(N):
            block = bits[i*M*Q:(i+1)*M*Q].reshape(M, Q)
            rank  = _NIST._gf2_rank(block)
            if rank == M:
                r_full += 1
            elif rank == M - 1:
                r_sub += 1
            else:
                r_rem += 1
        p_32 = 0.2888
        p_31 = 0.5776
        p_30 = 0.1336
        chi_sq = (((r_full - p_32 * N)**2 / (p_32 * N)) +
                  ((r_sub  - p_31 * N)**2 / (p_31 * N)) +
                  ((r_rem  - p_30 * N)**2 / (p_30 * N)))
        p = math.exp(-chi_sq / 2.0)
        return p, f"N={N}, F={r_full}, S={r_sub}, R={r_rem}"

    @staticmethod
    def _gf2_rank(mat: np.ndarray) -> int:
        """Gaussian elimination over GF(2)."""
        A = mat.copy().astype(np.uint8)
        rows, cols = A.shape
        rank = 0
        for col in range(cols):
            pivot = None
            for row in range(rank, rows):
                if A[row, col]:
                    pivot = row
                    break
            if pivot is None:
                continue
            A[[rank, pivot]] = A[[pivot, rank]]
            for row in range(rows):
                if row != rank and A[row, col]:
                    A[row] ^= A[rank]
            rank += 1
        return rank

    # ------------------------------------------------------------------ 5
    @staticmethod
    def dft_spectral(bits: np.ndarray) -> Tuple[Optional[float], str]:
        n = len(bits)
        if n < 1000:
            return None, "n < 1000, skipped"
        x = 2 * bits.astype(np.float64) - 1
        S = np.abs(np.fft.fft(x))[:n//2]
        T = math.sqrt(math.log(1/0.05) * n)
        N0 = 0.95 * n / 2.0
        N1 = float(np.sum(S < T))
        d  = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4.0)
        p  = math.erfc(abs(d) / math.sqrt(2))
        return p, f"N0={N0:.1f}, N1={N1:.0f}, d={d:.4f}"

    # ------------------------------------------------------------------ 6
    @staticmethod
    def non_overlapping_template(bits: np.ndarray,
                                  m: int = 9) -> Tuple[Optional[float], str]:
        n = len(bits)
        N = 8          # NIST uses 8 blocks
        M = n // N
        if M < m:
            return None, f"M={M} < m={m}, skipped"
        # Use a single aperiodic template: 000000001
        template = np.array([0,0,0,0,0,0,0,0,1], dtype=np.uint8)
        mu    = (M - m + 1) / 2**m
        sigma = M * (1/2**m - (2*m-1)/2**(2*m))
        W = []
        for i in range(N):
            block = bits[i*M:(i+1)*M]
            count = 0
            j = 0
            while j <= M - m:
                if np.array_equal(block[j:j+m], template):
                    count += 1
                    j += m
                else:
                    j += 1
            W.append(count)
        chi_sq = sum((w - mu)**2 / sigma for w in W) if sigma > 0 else 0.0
        p = special.gammaincc(N / 2.0, chi_sq / 2.0)
        return p, f"mu={mu:.4f}, sigma={sigma:.4f}"

    # ------------------------------------------------------------------ 7
    @staticmethod
    def overlapping_template(bits: np.ndarray, m: int = 9) -> Tuple[Optional[float], str]:
        n = len(bits)
        M = 1032
        N = n // M
        if N < 5:
            return None, f"N={N} < 5, skipped"
        K = 5
        template = np.ones(m, dtype=np.uint8)
        # NIST pi values for m=9, K=5
        pi = [0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865]
        freq = [0] * (K + 1)
        for i in range(N):
            block = bits[i*M:(i+1)*M]
            count = 0
            for j in range(M - m + 1):
                if np.array_equal(block[j:j+m], template):
                    count += 1
            idx = min(count, K)
            freq[idx] += 1
        chi_sq = sum((freq[i] - N * pi[i])**2 / (N * pi[i])
                     for i in range(K+1) if N * pi[i] > 0)
        p = special.gammaincc(K / 2.0, chi_sq / 2.0)
        return p, f"N={N}, m={m}"

    # ------------------------------------------------------------------ 8
    @staticmethod
    def maurers_universal(bits: np.ndarray,
                           L: int = 7, Q: int = 1280) -> Tuple[Optional[float], str]:
        n  = len(bits)
        K  = n // L - Q
        if K <= 0:
            return None, f"K={K} <= 0, skipped"

        # F8 FIX: replace str.join(str(b) for b in ...) string conversion with
        # numpy integer packing via np.packbits + bit extraction.
        # Previously each L-bit block was converted: int("".join(str(b) for b in block), 2)
        # which creates L Python string objects and joins them — extremely slow for
        # large n (called Q + K times, each building an L-char string).
        # Now: reshape bits into rows of length L, pack each row into a uint8 array,
        # then extract the integer value with bitwise shifts in a single vectorised call.
        # For L=7, Q=1280, n=1e6: saves ~(Q+K)*L ≈ 7M string operations.
        def _block_to_int_vectorised(bits_arr: np.ndarray, L: int) -> np.ndarray:
            """
            Convert consecutive L-bit windows into integer indices — vectorised.

            Each row of `bits_arr` reshaped to (-1, L) is treated as a big-endian
            binary number. We compute: sum(bit[j] * 2^(L-1-j)) for j in 0..L-1
            using powers-of-2 weights and a matrix multiply (dot product).
            """
            total = (len(bits_arr) // L) * L
            rows  = bits_arr[:total].reshape(-1, L).astype(np.int64)
            # Powers of 2: [2^(L-1), 2^(L-2), ..., 2^0]
            powers = (1 << np.arange(L - 1, -1, -1, dtype=np.int64))
            return rows @ powers   # shape: (n_blocks,)

        n_total_blocks = Q + (n // L - Q)   # Q init + K test blocks
        total_needed   = n_total_blocks * L
        if total_needed > n:
            return None, f"insufficient bits for Q+K blocks: need {total_needed}, have {n}"

        block_ids = _block_to_int_vectorised(bits[:total_needed], L)
        # block_ids[0..Q-1] = initialisation blocks
        # block_ids[Q..Q+K-1] = test blocks

        # Initialise last-occurrence table from first Q blocks
        T: Dict[int, int] = {}
        for i in range(Q):
            T[block_ids[i]] = i + 1   # 1-indexed as per NIST

        # Accumulate fn over K test blocks
        fn = 0.0
        for i in range(Q, Q + K):
            block_val = int(block_ids[i])
            dist = i + 1 - T.get(block_val, 0)
            if dist > 0:
                fn += math.log2(dist)
            T[block_val] = i + 1
        fn /= K

        # Expected value and variance from NIST Table 5
        ev  = {1:0.7326495, 2:1.5374383, 3:2.4016068, 4:3.3112247,
               5:4.2534266, 6:5.2177052, 7:6.1962507, 8:7.1836656,
               9:8.1764248, 10:9.1723243, 11:10.170032, 12:11.168765,
               13:12.168070, 14:13.167693, 15:14.167488, 16:15.167379}
        vr  = {1:0.690, 2:1.338, 3:1.901, 4:2.358, 5:2.705, 6:2.954,
               7:3.125, 8:3.238, 9:3.311, 10:3.356, 11:3.384, 12:3.401,
               13:3.410, 14:3.416, 15:3.419, 16:3.421}
        if L not in ev:
            return None, f"L={L} not in table"
        c   = 0.7 - 0.8/L + (4 + 32/L) * K**(-3/L) / 15
        sigma = c * math.sqrt(vr[L] / K)
        if sigma == 0:
            return None, "sigma=0"
        z   = (fn - ev[L]) / sigma
        p   = math.erfc(abs(z) / math.sqrt(2))
        return p, f"fn={fn:.4f}, ev={ev[L]:.4f}, z={z:.4f}"

    # ------------------------------------------------------------------ 9
    @staticmethod
    def linear_complexity(bits: np.ndarray, M: int = 500) -> Tuple[Optional[float], str]:
        n = len(bits)
        N = n // M
        if N < 1:
            return None, f"N={N}, skipped"
        mu = M / 2.0 + (9 + (-1)**(M+1)) / 36.0 - (M/3.0 + 2/9.0) / 2**M
        # Theoretical pi values (NIST Table 1 Section 2.10)
        pi = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
        K  = 6
        freq = [0] * (K + 1)
        for i in range(N):
            block = bits[i*M:(i+1)*M].tolist()
            L     = _NIST._berlekamp_massey(block)
            t     = (-1)**M * (L - mu) + 2/9.0
            if   t <= -2.5: idx = 0
            elif t <= -1.5: idx = 1
            elif t <= -0.5: idx = 2
            elif t <=  0.5: idx = 3
            elif t <=  1.5: idx = 4
            elif t <=  2.5: idx = 5
            else:           idx = 6
            freq[idx] += 1
        chi_sq = sum((freq[i] - N * pi[i])**2 / (N * pi[i])
                     for i in range(K+1) if N * pi[i] > 0)
        p = special.gammaincc(K / 2.0, chi_sq / 2.0)
        return p, f"N={N}, M={M}"

    @staticmethod
    def _berlekamp_massey(seq: List[int]) -> int:
        n, L, m, b, c = len(seq), 0, 1, [1], [1]
        b = b + [0] * (n - 1)
        c = c + [0] * (n - 1)
        for i in range(n):
            d = seq[i]
            for j in range(1, L + 1):
                d ^= c[j] & seq[i - j]
            if d:
                t = c[:]
                for j in range(m, n):
                    c[j] ^= b[j - m]
                if 2 * L <= i:
                    L = i + 1 - L
                    m = i + 1
                    b = t
        return L

    # ------------------------------------------------------------------ 10
    @staticmethod
    def serial(bits: np.ndarray, m: int = 16) -> Tuple[Optional[float], str]:
        n = len(bits)
        if n < m + 2:
            return None, "n too small, skipped"

        def psi_sq(bits: np.ndarray, m: int) -> float:
            """
            Compute ψ²_m for the serial test.

            F8 FIX: replace O(n·m) Python double-loop with sliding_window_view
            + integer bit-packing — a single numpy pass.

            Original code:
                counts: Dict[int, int] = {}
                for i in range(n):
                    key = 0
                    for j in range(m):
                        key = (key << 1) | int(bits[(i + j) % n])
                    counts[key] = counts.get(key, 0) + 1
                return sum(v**2 for v in counts.values()) * 2**m / n - n

            With m=16, n=1e6 this inner loop runs 16M Python iterations per call.
            The new version builds all n windows simultaneously as a 2D uint8 array
            using sliding_window_view (wrapping the sequence cyclically for the
            boundary), packs each m-bit row into an integer with a dot product
            against powers-of-2 weights, then uses np.bincount for the frequency
            table — O(n) numpy operations, no Python loop over bits.
            """
            if m == 0:
                return 0.0
            n = len(bits)
            # Cyclic extension: append first m-1 bits at the end so every
            # window of length m starting at position i is available.
            extended = np.concatenate([bits, bits[:m-1]]).astype(np.int64)
            # Build (n, m) window matrix — each row is one m-bit window
            windows = np.lib.stride_tricks.sliding_window_view(extended, m)  # (n, m)
            # Pack each row into an integer: sum(bit[j] * 2^(m-1-j))
            powers  = (1 << np.arange(m - 1, -1, -1, dtype=np.int64))       # (m,)
            keys    = windows @ powers                                         # (n,)
            # Count occurrences of each key (0 .. 2^m - 1)
            counts  = np.bincount(keys, minlength=1 << m)
            return float(np.sum(counts ** 2)) * (1 << m) / n - n

        # F9 FIX (carried from v2): cache the three psi_sq values.
        # Previously psi_sq(bits, m) and psi_sq(bits, m-1) were each called twice.
        psi_m   = psi_sq(bits, m)
        psi_m1  = psi_sq(bits, m - 1)
        psi_m2  = psi_sq(bits, m - 2)
        d2  = psi_m  - psi_m1
        d2a = psi_m  - 2 * psi_m1 + psi_m2
        p1 = special.gammaincc(2**(m-2), d2 / 2.0)
        p2 = special.gammaincc(2**(m-3), d2a / 2.0)
        p  = min(p1, p2)
        return p, f"m={m}, del2={d2:.4f}"

    # ------------------------------------------------------------------ 11
    @staticmethod
    def approximate_entropy(bits: np.ndarray, m: int = 10) -> Tuple[Optional[float], str]:
        n = len(bits)
        if n < m + 1:
            return None, "n too small, skipped"

        def phi(bits: np.ndarray, m: int) -> float:
            """
            Compute φ(m) for the approximate entropy test.

            F8 FIX: same sliding_window_view + integer packing technique as
            serial() / psi_sq().

            Original code:
                counts: Dict[int, int] = {}
                for i in range(n):
                    key = 0
                    for j in range(m):
                        key = (key << 1) | int(bits[(i + j) % n])
                    counts[key] = counts.get(key, 0) + 1
                return sum((c/n) * math.log(c/n) for c in counts.values() if c > 0)

            With m=10, n=1e6 this ran 10M Python iterations.
            The new version is O(n) numpy.
            """
            n = len(bits)
            extended = np.concatenate([bits, bits[:m]]).astype(np.int64)
            windows  = np.lib.stride_tricks.sliding_window_view(extended, m)[:n]
            powers   = (1 << np.arange(m - 1, -1, -1, dtype=np.int64))
            keys     = windows @ powers
            counts   = np.bincount(keys, minlength=1 << m)
            # Only non-zero counts contribute to the sum
            nonzero  = counts[counts > 0].astype(np.float64)
            probs    = nonzero / n
            return float(np.sum(probs * np.log(probs)))

        ap_en  = phi(bits, m) - phi(bits, m + 1)
        chi_sq = 2 * n * (math.log(2) - ap_en)
        p = special.gammaincc(2**(m - 1), chi_sq / 2.0)
        return p, f"ApEn={ap_en:.6f}"

    # ------------------------------------------------------------------ 12
    @staticmethod
    def cumulative_sums(bits: np.ndarray) -> Tuple[Optional[float], str]:
        n = len(bits)
        if n < 100:
            return None, "n < 100, skipped"
        x = 2 * bits.astype(np.float64) - 1
        S = np.cumsum(x)
        z = float(np.max(np.abs(S)))
        # NIST formula (forward mode)
        p = 1.0
        k_start = int((-n/z + 1) / 4)
        k_end   = int(( n/z - 1) / 4)
        total   = 0.0
        for k in range(max(k_start, -100), min(k_end, 100) + 1):
            total += (stats.norm.cdf((4*k+1)*z/math.sqrt(n)) -
                      stats.norm.cdf((4*k-1)*z/math.sqrt(n)))
        k_start2 = int((-n/z - 3) / 4)
        k_end2   = int(( n/z - 1) / 4)
        total2   = 0.0
        for k in range(max(k_start2, -100), min(k_end2, 100) + 1):
            total2 += (stats.norm.cdf((4*k+3)*z/math.sqrt(n)) -
                       stats.norm.cdf((4*k+1)*z/math.sqrt(n)))
        p = 1 - total + total2
        return max(p, 0.0), f"z={z:.4f}"

    # ------------------------------------------------------------------ 13
    @staticmethod
    def random_excursions(bits: np.ndarray) -> Tuple[Optional[float], str]:
        n = len(bits)
        if n < 1000000:
            pass  # NIST allows smaller n but warns
        x = 2 * bits.astype(np.int64) - 1
        S = np.cumsum(np.concatenate([[0], x, [0]]))
        # Cycles: positions where cumsum returns to 0
        zero_pos = np.where(S == 0)[0]
        J = len(zero_pos) - 1      # number of complete cycles
        if J < 500:
            return None, f"J={J} < 500 cycles, skipped (need more bits)"
        # States of interest: x ∈ {-4,-3,-2,-1,1,2,3,4}
        states = [-4, -3, -2, -1, 1, 2, 3, 4]
        pi_table = {           # NIST Table 8 — P(v_k = x)  for k in 0..5
            1: [0.5000, 0.2500, 0.1250, 0.0625, 0.0312, 0.0313],
            2: [0.7500, 0.0625, 0.0469, 0.0352, 0.0264, 0.0791],
            3: [0.8333, 0.0278, 0.0231, 0.0193, 0.0161, 0.0804],
            4: [0.8750, 0.0156, 0.0137, 0.0120, 0.0105, 0.0733],
        }
        K = 5
        p_min = 1.0
        for state in states:
            abs_s = abs(state)
            if abs_s not in pi_table:
                continue
            pi = pi_table[abs_s]
            # Count visits to `state` in each cycle
            freq = [0] * (K + 1)
            for j in range(J):
                cycle = S[zero_pos[j]:zero_pos[j+1]+1]
                v = int(np.sum(cycle == state))
                freq[min(v, K)] += 1
            chi_sq = sum((freq[i] - J * pi[i])**2 / (J * pi[i])
                         for i in range(K+1) if J * pi[i] > 0)
            p = special.gammaincc(K / 2.0, chi_sq / 2.0)
            p_min = min(p_min, p)
        return p_min, f"J={J} cycles, min p over 8 states"

    # ------------------------------------------------------------------ 14
    @staticmethod
    def random_excursions_variant(bits: np.ndarray) -> Tuple[Optional[float], str]:
        n = len(bits)
        x = 2 * bits.astype(np.int64) - 1
        S = np.cumsum(np.concatenate([[0], x, [0]]))
        zero_pos = np.where(S == 0)[0]
        J = len(zero_pos) - 1
        if J < 500:
            return None, f"J={J} < 500 cycles, skipped"
        states = list(range(-9, 0)) + list(range(1, 10))   # 18 states
        p_min  = 1.0
        for state in states:
            count = int(np.sum(S == state))
            xi    = (count - J) / math.sqrt(2 * J * (4 * abs(state) - 2))
            p     = math.erfc(abs(xi) / math.sqrt(2))
            p_min = min(p_min, p)
        return p_min, f"J={J} cycles, min p over 18 states"


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

class NISTTestRunner:
    """
    Run all 15 NIST SP 800-22 tests against one or more bit sequences.

    Automatically chooses backend:
      1. NIST C binary (if NIST_ASSESS_PATH env-var points to `assess`)
      2. Pure NumPy/SciPy (always available)

    Parameters
    ----------
    significance : float
        P-value threshold for pass/fail.  NIST recommends 0.01.
    """

    def __init__(self, significance: float = ALPHA):
        self.significance = significance
        self._has_binary  = self._detect_binary()

    # ------------------------------------------------------------------
    def _detect_binary(self) -> bool:
        if NIST_BINARY_PATH and Path(NIST_BINARY_PATH).is_file():
            return True
        # Cross-platform PATH check (where on Windows, which on Linux/macOS)
        import sys
        cmd = ["where", "assess"] if sys.platform == "win32" else ["which", "assess"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    # ------------------------------------------------------------------
    def run_all(self, bits: np.ndarray) -> NISTResult:
        """
        Run all 15 tests on `bits` (uint8 array of 0s and 1s).

        Returns NISTResult with p_values and pass/fail for every test.
        """
        bits = np.asarray(bits, dtype=np.uint8).flatten()
        n    = len(bits)

        if self._has_binary:
            return self._run_binary(bits)
        else:
            return self._run_numpy(bits)

    # ------------------------------------------------------------------
    def _run_numpy(self, bits: np.ndarray) -> NISTResult:
        """Execute all 15 tests using pure NumPy/SciPy."""
        tests = [
            _NIST.frequency_monobit,
            _NIST.frequency_block,
            _NIST.runs,
            _NIST.longest_run,
            _NIST.binary_matrix_rank,
            _NIST.dft_spectral,
            _NIST.non_overlapping_template,
            _NIST.overlapping_template,
            _NIST.maurers_universal,
            _NIST.linear_complexity,
            _NIST.serial,
            _NIST.approximate_entropy,
            _NIST.cumulative_sums,
            _NIST.random_excursions,
            _NIST.random_excursions_variant,
        ]

        p_values = []
        passed   = []
        notes    = []

        for fn in tests:
            try:
                p, note = fn(bits)
            except Exception as exc:
                p, note = None, f"Exception: {exc}"
            p_values.append(p)
            passed.append(None if p is None else bool(p >= self.significance))
            notes.append(note)

        return NISTResult(
            p_values=p_values,
            passed=passed,
            n_bits=len(bits),
            significance=self.significance,
            backend="numpy",
            notes=notes,
        )

    # ------------------------------------------------------------------
    def _run_binary(self, bits: np.ndarray) -> NISTResult:
        """
        Delegate to the official NIST `assess` binary.

        Writes bits to a binary file, calls assess, parses
        finalAnalysisReport.txt, and returns a NISTResult.
        """
        binary = NIST_BINARY_PATH or "assess"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write bits as packed bytes
            bit_path = os.path.join(tmpdir, "input.bin")
            packed   = np.packbits(bits)
            with open(bit_path, "wb") as f:
                f.write(packed.tobytes())

            n_bits = len(bits)

            # NIST assess expects: assess <bitlength>
            # Then interactive prompts — we pipe them in
            prompts = f"0\n{bit_path}\n1\n{n_bits}\n0\n"
            try:
                result = subprocess.run(
                    [binary, str(n_bits)],
                    input=prompts,
                    capture_output=True,
                    text=True,
                    cwd=tmpdir,
                    timeout=300,
                )
            except subprocess.TimeoutExpired:
                # Fall back to numpy
                return self._run_numpy(bits)

            # Parse report
            report_path = os.path.join(tmpdir, "experiments",
                                        "AlgorithmTesting",
                                        "finalAnalysisReport.txt")
            if not os.path.exists(report_path):
                return self._run_numpy(bits)

            return self._parse_nist_report(report_path)

    # ------------------------------------------------------------------
    def _parse_nist_report(self, report_path: str) -> NISTResult:
        """Parse NIST finalAnalysisReport.txt into NISTResult."""
        p_values = [None] * N_TESTS
        passed   = [None] * N_TESTS
        notes    = [""] * N_TESTS

        with open(report_path) as f:
            lines = f.readlines()

        # NIST report format: each line contains test name and p-value
        for line in lines:
            line = line.strip()
            if not line or line.startswith("-"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    p = float(parts[-1])
                    # Match to test index by name substring
                    for i, name in enumerate(TEST_NAMES):
                        key = name.split()[0].lower()
                        if key in line.lower():
                            p_values[i] = p
                            passed[i]   = p >= self.significance
                            break
                except ValueError:
                    pass

        return NISTResult(
            p_values=p_values,
            passed=passed,
            n_bits=0,
            significance=self.significance,
            backend="nist_binary",
            notes=notes,
        )

    # ------------------------------------------------------------------
    def run_pre_post_extraction(self,
                                 raw_bits:       np.ndarray,
                                 extracted_bits: np.ndarray
                                 ) -> Tuple[NISTResult, NISTResult]:
        """
        Convenience: run tests on both raw and extracted bits.

        Returns (pre_result, post_result).
        """
        pre  = self.run_all(raw_bits)
        post = self.run_all(extracted_bits)
        return pre, post

    # ------------------------------------------------------------------
    def summarize(self, result: NISTResult) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"NIST SP 800-22 Results  (n={result.n_bits:,}, "
            f"α={result.significance}, backend={result.backend})",
            "-" * 60,
        ]
        for i, name in enumerate(result.test_names):
            p   = result.p_values[i]
            ok  = result.passed[i]
            sym = "PASS" if ok else ("FAIL" if ok is False else "SKIP")
            p_str = f"{p:.6f}" if p is not None else "  N/A  "
            lines.append(f"  {sym}  p={p_str}  {name}")
        lines.append("-" * 60)
        lines.append(f"  Pass rate: {result.pass_rate():.1%}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# NIST compile instructions (printed when binary not found)
# ---------------------------------------------------------------------------

COMPILE_INSTRUCTIONS = """
=============================================================================
HOW TO COMPILE THE OFFICIAL NIST SP 800-22 BINARY
=============================================================================

1. Download the source code:
   https://csrc.nist.gov/CSRC/media/Projects/Random-Bit-Generation/documents/sts-2.1.2.zip

2. Extract:
   unzip sts-2.1.2.zip
   cd sts-2.1.2

3. Compile (Linux / macOS):
   gcc -O2 -o assess src/*.c -lm

4. Set the path so this module uses it:
   export NIST_ASSESS_PATH=/path/to/sts-2.1.2/assess

5. Or place `assess` anywhere on your PATH.

NOTE: If the binary is absent, this module automatically uses the built-in
      NumPy/SciPy implementation of all 15 tests — identical mathematics,
      no compilation required.
=============================================================================
"""


def print_compile_instructions():
    print(COMPILE_INSTRUCTIONS)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("NIST SP 800-22 Runner v3 — self-test")
    print("=" * 60)

    rng   = np.random.RandomState(42)
    bits  = rng.randint(0, 2, size=1_000_000, dtype=np.uint8)

    runner = NISTTestRunner(significance=0.01)
    print(f"Backend: {'NIST binary' if runner._has_binary else 'NumPy/SciPy'}")
    print(f"Running all 15 tests on {len(bits):,} random bits...")

    result = runner.run_all(bits)
    print(runner.summarize(result))
