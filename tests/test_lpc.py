"""Tests for :mod:`formant.lpc`.

Every test calls at least one public stub.  The suite is intentionally red
until the LPC pole analysis, continuity tracking and synthetic round-trip are
implemented.
"""

from __future__ import annotations

import math

from formant import lpc

FS = 8000.0
PREWARM = 512


def _damped_frame(specs, sample_rate=FS, n=2048):
    """Build a deterministic sum of damped sinusoids.

    ``specs`` contains ``(frequency_hz, bandwidth_hz[, amplitude])`` tuples.
    The oscillators are evaluated after a 512-sample pre-roll so the frame
    starts in steady state rather than with a zero-state filter transient.
    This helper is independent of ``lpc.synthesize`` so tracking tests do not
    depend on the very function they are cross-checking.
    """
    out = [0.0] * n
    for spec in specs:
        frequency = float(spec[0])
        bandwidth = float(spec[1])
        amplitude = float(spec[2]) if len(spec) > 2 else 1.0
        damping = math.exp(-math.pi * bandwidth / sample_rate)
        for t in range(n):
            tau = t + PREWARM
            out[t] += amplitude * (damping**tau) * math.sin(2.0 * math.pi * frequency * tau / sample_rate)
    peak = max((abs(value) for value in out), default=0.0)
    if peak > 0.0:
        out = [0.9 * value / peak for value in out]
    return out


def _entry_near(entries, frequency, tolerance):
    for entry in entries:
        if abs(entry["frequency"] - frequency) <= tolerance:
            return entry
    return None


# ---------------------------------------------------------------------------
# The ten defect-specific tests
# ---------------------------------------------------------------------------


def test_autocorr_includes_lag_order():
    """Defect 1: stopping at order-1 loses the final Yule-Walker lag."""
    frame = [1.0, 2.0, 3.0, 4.0]
    result = lpc.autocorr(frame, 3)
    expected = [30.0, 20.0, 11.0, 4.0]  # lags 0, 1, 2, 3
    assert len(result) == 4
    for got, want in zip(result, expected):
        assert abs(got - want) < 1e-6


def test_levinson_reflection_sign_matches_ar():
    """Defect 2: flipped reflection coefficients invert the prediction.

    For ``r = [1.0, 0.9]`` the correct order-1 model is
    ``A(z) = 1 - 0.9*z**-1``, hence ``k1 = -0.9`` and ``a1 = -0.9``.
    """
    result = lpc.levinson([1.0, 0.9], 1)
    assert len(result["coefficients"]) == 2
    assert abs(result["coefficients"][0] - 1.0) < 1e-6
    assert abs(result["coefficients"][1] + 0.9) < 1e-6
    assert abs(result["reflection"][0] + 0.9) < 1e-6
    assert abs(result["error"] - 0.19) < 1e-6


def test_lpc_roots_excludes_unstable_poles():
    """Defect 3: a pole outside the unit circle must not survive.

    ``(1 - 0.9*z**-1) * (1 - 1.1*z**-1) = 1 - 2.0*z**-1 + 0.99*z**-2``
    has roots 0.9 and 1.1; only 0.9 is a stable formant pole.
    """
    roots = lpc.lpc_roots([1.0, -2.0, 0.99])
    assert len(roots) == 1
    assert abs(roots[0] - 0.9) < 1e-6
    assert abs(roots[0]) < 1.0


def test_poles_to_formants_scales_frequency():
    """Defect 4: phase in radians must be converted to hertz."""
    pole = 0.9 + 0.1j
    expected = math.atan2(0.1, 0.9) * FS / (2.0 * math.pi)
    formants = lpc.poles_to_formants([pole], FS)
    assert len(formants) == 1
    assert abs(formants[0]["frequency"] - expected) < 1e-6
    assert formants[0]["frequency"] > 1.0  # radians alone would be ~0.11


def test_poles_to_formants_bandwidth_is_positive():
    """Defect 5: log|z| is negative inside the unit circle."""
    pole = 0.9 + 0.1j
    expected = -math.log(abs(pole)) * FS / math.pi
    formants = lpc.poles_to_formants([pole], FS)
    assert len(formants) == 1
    assert formants[0]["bandwidth"] > 0.0
    assert abs(formants[0]["bandwidth"] - expected) < 1e-6


def test_track_formants_keeps_labels_continuous():
    """Defect 6: per-frame sorting steals F2 when a spurious peak appears.

    Frame 1 carries F1=500 and F2=1800.  Frame 2 introduces a new 1100 Hz
    peak.  Continuous labels require 1100 to become a new track and 1800 to
    remain F2; sorting each frame afresh would rename 1800 to F3.
    """
    frame_one = _damped_frame([(500.0, 5.0), (1800.0, 5.0)])
    frame_two = _damped_frame([(500.0, 5.0), (1100.0, 5.0), (1800.0, 5.0)])
    tracked = lpc.track_formants([frame_one, frame_two], FS, 8)
    assert len(tracked) == 2
    target = _entry_near(tracked[1], 1800.0, 120.0)
    assert target is not None
    assert target["label"] == "F2"
    newcomer = _entry_near(tracked[1], 1100.0, 100.0)
    assert newcomer is not None
    assert newcomer["label"] != "F2"


def test_track_formants_rejects_large_jump():
    """Defect 7: a 1 kHz jump is beyond the continuity gate."""
    first = _damped_frame([(500.0, 80.0)])
    second = _damped_frame([(1500.0, 80.0)])
    tracked = lpc.track_formants([first, second], FS, 2)
    assert tracked[0][0]["label"] == "F1"
    assert tracked[1][0]["label"] != "F1"
    assert tracked[1][0]["frequency"] > 1000.0


def test_synthesize_has_no_startup_transient():
    """Defect 8: an un-prewarmed IIR changes the first 20 samples."""
    frequency, bandwidth = 500.0, 80.0
    n = 400
    raw = [
        math.exp(-math.pi * bandwidth * (t + PREWARM) / FS)
        * math.sin(2.0 * math.pi * frequency * (t + PREWARM) / FS)
        for t in range(n)
    ]
    peak = max(abs(value) for value in raw)
    expected = [0.9 * value / peak for value in raw[:20]]
    actual = lpc.synthesize([{"frequency": frequency, "bandwidth": bandwidth}], FS, n)
    assert len(actual) == n
    for index, want in enumerate(expected):
        assert abs(actual[index] - want) < 1e-6


def test_validate_poles_rejects_real_poles():
    """Defect 9: real poles are not resonances and must be reported."""
    errors = lpc.validate_poles([0.9 + 0.0j, 0.8 + 0.0j])
    assert errors
    assert any("real" in message.lower() or "conjugate" in message.lower() for message in errors)
    valid_pair = [0.9 + 0.1j, 0.9 - 0.1j]
    assert lpc.validate_poles(valid_pair) == []


def test_track_batch_does_not_share_state():
    """Defect 10: the second track must start a fresh label sequence.

    The first track ends with F1~520 and F2~1480.  Reusing its matching state
    would label the second track's 1500 Hz peak as F2; a fresh tracker must
    call it F1.
    """
    track_a = {
        "frames": [
            _damped_frame([(500.0, 5.0), (1500.0, 6.0)]),
            _damped_frame([(520.0, 5.0), (1480.0, 6.0)]),
        ],
        "sample_rate": FS,
        "order": 4,
    }
    track_b = {
        "frames": [_damped_frame([(1500.0, 5.0)])],
        "sample_rate": FS,
        "order": 2,
    }
    batch = lpc.track_batch([track_a, track_b], {"max_jump_hz": 400.0})
    assert batch["count"] == 2
    assert len(batch["results"][1]) == 1
    assert len(batch["results"][1][0]) == 1
    assert batch["results"][1][0][0]["label"] == "F1"
    assert abs(batch["results"][1][0][0]["frequency"] - 1500.0) < 100.0


# ---------------------------------------------------------------------------
# Five cross-cutting verification tests
# ---------------------------------------------------------------------------


def test_known_formants_recovered_from_synthetic_signal():
    """A clean 500/1500/2500 Hz signal yields the first three formants."""
    signal = _damped_frame([(500.0, 5.0), (1500.0, 6.0), (2500.0, 7.0)])
    tracked = lpc.track_formants([signal], FS, 8)
    assert len(tracked) == 1
    assert len(tracked[0]) >= 3
    for entry, target in zip(tracked[0][:3], (500.0, 1500.0, 2500.0)):
        assert abs(entry["frequency"] - target) < 50.0, f"bad first formant: {tracked[0]}"


def test_all_poles_are_inside_unit_circle():
    """Every returned pole must satisfy |z| < 1 (with 1e-6 tolerance)."""
    signal = _damped_frame([(500.0, 80.0), (1500.0, 100.0)])
    roots = lpc.lpc_roots(lpc.levinson(lpc.autocorr(signal, 4), 4)["coefficients"])
    assert roots
    for pole in roots:
        assert abs(pole) < 1.0 + 1e-6


def test_synthesize_analyze_roundtrip():
    """Synthesized formants must be recoverable within 100 Hz."""
    formants = [
        {"frequency": 400.0, "bandwidth": 5.0},
        {"frequency": 1200.0, "bandwidth": 6.0},
        {"frequency": 2400.0, "bandwidth": 7.0},
    ]
    signal = lpc.synthesize(formants, FS, 2048)
    tracked = lpc.track_formants([signal], FS, 8)
    assert len(tracked) == 1
    for formant in formants:
        entry = _entry_near(tracked[0], formant["frequency"], 100.0)
        assert entry is not None, f"missing round-trip formant {formant}"


def test_consecutive_f1_jump_is_bounded():
    """The tracked F1 must move less than 200 Hz from frame to frame."""
    frames = [
        _damped_frame([(500.0, 80.0)]),
        _damped_frame([(520.0, 80.0)]),
        _damped_frame([(540.0, 80.0)]),
    ]
    tracked = lpc.track_formants(frames, FS, 2)
    f1_values = []
    for frame in tracked:
        entry = next((item for item in frame if item["label"] == "F1"), None)
        assert entry is not None
        f1_values.append(entry["frequency"])
    jumps = [abs(right - left) for left, right in zip(f1_values, f1_values[1:])]
    assert len(jumps) == 2
    assert max(jumps) < 200.0


def test_batch_matches_individual_tracks():
    """Batch results for each track equal its standalone tracking result."""
    track_a = {
        "frames": [_damped_frame([(500.0, 80.0)]), _damped_frame([(520.0, 80.0)])],
        "sample_rate": FS,
        "order": 2,
    }
    track_b = {
        "frames": [_damped_frame([(900.0, 90.0)]), _damped_frame([(920.0, 90.0)])],
        "sample_rate": FS,
        "order": 2,
    }
    batch = lpc.track_batch([track_a, track_b], {"max_jump_hz": 400.0})
    individual_a = lpc.track_formants(track_a["frames"], FS, 2)
    individual_b = lpc.track_formants(track_b["frames"], FS, 2)
    assert batch["results"][0] == individual_a
    assert batch["results"][1] == individual_b
