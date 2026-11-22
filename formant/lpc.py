"""Linear-predictive formant analysis and tracking.

The eight public functions document their full numerical contract.  Only the
Python standard library (``math``, ``cmath``) and pytest are required.
"""

from __future__ import annotations

import cmath
import math
from typing import Any, Mapping, Sequence

DEFAULT_MAX_JUMP_HZ = 400.0
MIN_FORMANT_HZ = 50.0
MAX_FORMANT_RATIO = 0.45
STABILITY_TOL = 1e-6
SYNTH_PREWARM = 512


def autocorr(frame: Sequence[float], order: int) -> list[float]:
    """Return unnormalised autocorrelation lags ``0 .. order``.

    ``r[k] = sum(frame[n] * frame[n + k] for n in range(len(frame) - k))``
    with zero-padding beyond the frame end.  Lag zero is first; no mean
    removal and no normalisation are applied.  The result always has
    ``order + 1`` entries, including the final lag.

    Tolerance for comparisons: 1e-6.
    """
    n = len(frame)
    return [
        sum(frame[t] * frame[t + k] for t in range(n - k))
        for k in range(order + 1)
    ]


def levinson(r: Sequence[float], order: int) -> dict:
    """Solve the order-``order`` Yule-Walker system with Levinson-Durbin.

    ``r`` must contain at least ``order + 1`` non-negative-lag
    autocorrelations, starting at lag zero.  The returned mapping has:

    ``coefficients``
        ``[1.0, a1, ..., ap]`` for the prediction polynomial
        ``A(z) = 1 + a1*z**-1 + ... + ap*z**-p``.  The one-step
        predictor is ``x[n] ~= -(a1*x[n-1] + ... + ap*x[n-p])``.
    ``reflection``
        ``[k1, ..., kp]`` reflection coefficients; for a stable all-pole
        model every value lies strictly between -1 and 1.
    ``error``
        Final prediction error ``r[0] * prod(1 - ki**2)``.

    A sign error in either the reflection coefficients or the returned
    prediction polynomial is observable as an inverted synthetic signal.
    Tolerance for comparisons: 1e-6.
    """
    r = [float(value) for value in r]
    a = [1.0]
    reflection: list[float] = []
    error = r[0]
    for i in range(1, order + 1):
        acc = r[i] + sum(a[j] * r[i - j] for j in range(1, i))
        k = -acc / error if error != 0.0 else 0.0
        reflection.append(k)
        prev = a
        a = [1.0] + [prev[j] + k * prev[i - j] for j in range(1, i)] + [k]
        error *= 1.0 - k * k
    return {"coefficients": a, "reflection": reflection, "error": error}


def _polyval(coeffs: Sequence[complex], z: complex) -> complex:
    value = 0j
    for c in coeffs:
        value = value * z + c
    return value


def _poly_roots(coeffs: Sequence[complex]) -> list[complex]:
    """Durand-Kerner root finder for a polynomial, highest degree first."""
    coeffs = [complex(c) for c in coeffs]
    while len(coeffs) > 1 and coeffs[0] == 0:
        coeffs.pop(0)
    degree = len(coeffs) - 1
    if degree <= 0:
        return []
    lead = coeffs[0]
    coeffs = [c / lead for c in coeffs]
    if degree == 1:
        return [-coeffs[1]]
    seed = 0.4 + 0.9j
    z = 1 + 0j
    roots: list[complex] = []
    for _ in range(degree):
        z *= seed
        roots.append(z)
    for _ in range(500):
        max_delta = 0.0
        for i in range(degree):
            denom = 1 + 0j
            for j in range(degree):
                if i != j:
                    denom *= roots[i] - roots[j]
            if denom == 0:
                denom = 1e-12 + 0j
            delta = _polyval(coeffs, roots[i]) / denom
            roots[i] -= delta
            if abs(delta) > max_delta:
                max_delta = abs(delta)
        if max_delta < 1e-13:
            break
    return roots


def lpc_roots(coeffs: Sequence[complex]) -> list[complex]:
    """Return all poles of ``A(z)`` whose magnitude is strictly below one.

    ``coeffs`` are ordered from ``z**p`` down to ``z**0``.  Equivalently the
    input ``[1, a1, ..., ap]`` describes
    ``1 + a1*z**-1 + ... + ap*z**-p = 0``; after multiplying by ``z**p``
    that is a degree-``p`` polynomial in ``z``.  Every stable root is
    returned exactly once (conjugate pairs are both present); roots with
    ``abs(z) >= 1`` must be discarded because a formant resonator is only
    stable inside the unit circle.  Order is unspecified.  Root residual
    accuracy: 1e-6.
    """
    return [z for z in _poly_roots(coeffs) if abs(z) < 1.0]


def poles_to_formants(roots: Sequence[complex], sample_rate: float) -> list[dict]:
    """Convert stable LPC poles to frequency/bandwidth pairs.

    For a pole ``z = exp((-pi*B/fs) + j*2*pi*f/fs)`` with ``fs`` in Hz:

    * ``frequency = abs(phase(z)) * fs / (2*pi)`` in Hz,
    * ``bandwidth = -log(abs(z)) * fs / pi`` in Hz.

    One entry is emitted for every root with non-negative imaginary part;
    the negative-imaginary conjugate is the mirror image and is skipped.
    Real roots are converted too (they map to 0 Hz or Nyquist), so callers
    that only want formants must reject them first.  Results are sorted by
    increasing frequency.  Tolerance: 1e-6.
    """
    formants = []
    for z in roots:
        z = complex(z)
        if z.imag < 0.0:
            continue
        magnitude = abs(z)
        formants.append(
            {
                "frequency": abs(cmath.phase(z)) * sample_rate / (2.0 * math.pi),
                "bandwidth": -math.log(magnitude) * sample_rate / math.pi,
            }
        )
    formants.sort(key=lambda entry: entry["frequency"])
    return formants


def _frame_formants(
    frame: Sequence[float], sample_rate: float, order: int
) -> list[dict]:
    """Candidate formants of a single frame, non-formant poles rejected."""
    r = autocorr(frame, order)
    coeffs = levinson(r, order)["coefficients"]
    roots = lpc_roots(coeffs)
    upper = MAX_FORMANT_RATIO * sample_rate
    candidates = []
    for root in roots:
        if root.imag <= 0.0:
            # Real poles are not resonances; the lower-half-plane mirrors
            # are the same resonances as their upper-half conjugates.
            continue
        if not any(abs(root.conjugate() - other) <= 1e-6 for other in roots):
            continue
        frequency = abs(cmath.phase(root)) * sample_rate / (2.0 * math.pi)
        bandwidth = -math.log(abs(root)) * sample_rate / math.pi
        if frequency < MIN_FORMANT_HZ or frequency > upper:
            continue
        if bandwidth <= 0.0:
            continue
        candidates.append({"frequency": frequency, "bandwidth": bandwidth})
    candidates.sort(key=lambda entry: entry["frequency"])
    return candidates


def _track(
    frames: Sequence[Sequence[float]],
    sample_rate: float,
    order: int,
    max_jump_hz: float,
) -> list:
    results = []
    tracks: list[dict] = []  # persistent label -> last frequency state
    next_label = 1
    for frame in frames:
        candidates = _frame_formants(frame, sample_rate, order)
        used: set[int] = set()
        entries = []
        for candidate in candidates:
            best = None
            best_diff = max_jump_hz
            for index, track in enumerate(tracks):
                if index in used:
                    continue
                diff = abs(candidate["frequency"] - track["frequency"])
                if diff <= best_diff:
                    best = index
                    best_diff = diff
            if best is not None:
                label = tracks[best]["label"]
                tracks[best]["frequency"] = candidate["frequency"]
                used.add(best)
            else:
                label = f"F{next_label}"
                next_label += 1
                tracks.append({"label": label, "frequency": candidate["frequency"]})
                used.add(len(tracks) - 1)
            entries.append(
                {
                    "label": label,
                    "frequency": candidate["frequency"],
                    "bandwidth": candidate["bandwidth"],
                }
            )
        entries.sort(key=lambda entry: entry["frequency"])
        results.append(entries)
    return results


def track_formants(frames: Sequence[Sequence[float]], sample_rate: float, order: int) -> list:
    """Track formants across consecutive frames.

    Every frame is analysed independently, then neighbouring frames are
    linked by nearest-frequency matching with a hard continuity gate of
    ``DEFAULT_MAX_JUMP_HZ`` (400 Hz).  The result is a list with exactly one
    entry per input frame; each entry is a list of mappings
    ``{"label": str, "frequency": Hz, "bandwidth": Hz}`` sorted by
    frequency.  Labels are ``"F1"``, ``"F2"``, ... assigned on the first
    frame by ascending frequency; a matched candidate inherits its label.
    A candidate farther than the gate from every previous track starts a
    new label instead of being forced onto the old track.  Thus a 1 kHz
    jump must not keep an ``"F1"`` label.

    Candidates below 50 Hz, above ``0.45 * sample_rate``, with a
    non-positive bandwidth, or arising from real/non-conjugate poles are
    discarded as non-formants.  Missing tracks may be absent from a frame;
    labels are never renumbered per frame.  Tolerance: 1e-6.
    """
    return _track(frames, sample_rate, order, DEFAULT_MAX_JUMP_HZ)


def synthesize(formants: Sequence[Mapping[str, float]], sample_rate: float, n: int) -> list[float]:
    """Synthesize a deterministic sum of damped sinusoids from formants.

    Each formant mapping has ``frequency`` in Hz and ``bandwidth`` in Hz,
    optionally ``amplitude`` (default 1.0).  The damped oscillators are
    evaluated after a deterministic pre-roll of 512 samples, so sample ``t``
    uses ``tau = t + 512``::

        raw[t] = sum(A_i * exp(-pi*B_i*tau/fs) * sin(2*pi*f_i*tau/fs))

    The pre-roll removes the un-prewarmed IIR startup transient: the first
    20 samples must equal the steady damped-oscillator formula, not a
    zero-state filter response.  The whole signal is then scaled so its
    peak absolute value is 0.9.  ``n`` samples are returned as floats.
    Tolerance: 1e-6.
    """
    raw = []
    for t in range(n):
        tau = t + SYNTH_PREWARM
        total = 0.0
        for formant in formants:
            amplitude = float(formant.get("amplitude", 1.0))
            frequency = float(formant["frequency"])
            bandwidth = float(formant["bandwidth"])
            total += (
                amplitude
                * math.exp(-math.pi * bandwidth * tau / sample_rate)
                * math.sin(2.0 * math.pi * frequency * tau / sample_rate)
            )
        raw.append(total)
    peak = max((abs(value) for value in raw), default=0.0)
    if peak > 0.0:
        raw = [0.9 * value / peak for value in raw]
    return raw


def validate_poles(roots: Sequence[complex]) -> list[str]:
    """Return human-readable stability/pairing errors for LPC poles.

    An empty list means the poles are valid formant poles.  The checks are:
    at least one conjugate pair is present; every ``abs(z) < 1`` (within
    1e-6); every non-real pole has a conjugate partner within 1e-6; and a
    purely real pole is rejected because it is not a resonance.  Unstable
    poles and unpaired real poles must both be reported, never treated as
    formants.  Tolerance: 1e-6.
    """
    roots = [complex(z) for z in roots]
    errors: list[str] = []
    has_conjugate_pair = False
    for z in roots:
        if abs(z) >= 1.0 - STABILITY_TOL:
            errors.append(f"unstable pole {z}: |z| = {abs(z):.6f} is not inside the unit circle")
        if abs(z.imag) <= STABILITY_TOL:
            errors.append(f"real pole {z.real:.6f} is not a resonance")
        elif any(abs(z.conjugate() - other) <= 1e-6 for other in roots):
            has_conjugate_pair = True
        else:
            errors.append(f"pole {z} has no conjugate partner")
    if not has_conjugate_pair:
        errors.append("no conjugate pole pair present")
    return errors


def track_batch(tracks: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict:
    """Track several independent recordings without sharing state.

    ``tracks`` is a sequence of mappings, each with ``frames``,
    ``sample_rate`` and ``order``.  ``config`` may provide
    ``max_jump_hz`` (default 400 Hz).  Each track starts with fresh LPC and
    label state: the second track must never continue from the first
    track's last frame.  Returns
    ``{"results": [track_formants(...), ...], "count": len(tracks)}``.
    Tolerance: 1e-6.
    """
    max_jump_hz = float(config.get("max_jump_hz", DEFAULT_MAX_JUMP_HZ))
    results = [
        _track(
            track["frames"],
            float(track["sample_rate"]),
            int(track["order"]),
            max_jump_hz,
        )
        for track in tracks
    ]
    return {"results": results, "count": len(tracks)}
