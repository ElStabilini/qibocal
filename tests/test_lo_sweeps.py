"""Test the batched LO sweeps of qubit spectroscopy-like acquisitions.

In the mock platform both drive LOs are at 4.9 GHz.
"""

from itertools import pairwise

import numpy as np
import pytest
from qibolab._core.components import IqChannel

from qibocal.protocols.flux_dependence.qubit_flux_dependence import (
    QubitFluxParameters,
)
from qibocal.protocols.flux_dependence.qubit_flux_dependence import (
    _acquisition as qubit_flux_acquisition,
)
from qibocal.protocols.lo_sweeps import MAX_IF_BANDWIDTH, plan_batches, stitch

LO_FREQUENCY = 4.9e9


def _share_drive_lo(platform, monkeypatch):
    """Drive qubit 1 with the LO of qubit 0."""
    channels = platform.instruments["dummy"].channels
    monkeypatch.setitem(
        channels, platform.qubits[1].drive, IqChannel(mixer=None, lo="0/drive_lo")
    )


def _spy_execute(platform, monkeypatch) -> list:
    """Record the updates of every execution."""
    executed_updates = []
    execute = platform.execute

    def spy(*args, updates=None, **kwargs):
        executed_updates.append(updates)
        return execute(*args, updates=updates, **kwargs)

    monkeypatch.setattr(platform, "execute", spy)
    return executed_updates


def test_lo_kept_within_bandwidth():
    freqs = np.linspace(4.7e9, 4.8e9, 11)
    [batch] = plan_batches(freqs, LO_FREQUENCY)
    assert batch.anchor
    assert batch.lo_frequency == LO_FREQUENCY
    np.testing.assert_array_equal(batch.frequencies, freqs)


def test_lo_retuned_outside_bandwidth():
    """Narrow window, which does not require batching, but too far from the LO."""
    freqs = np.linspace(4.0e9, 4.3e9, 31)
    [batch] = plan_batches(freqs, LO_FREQUENCY)
    assert batch.anchor
    assert batch.lo_frequency == pytest.approx(4.15e9)
    np.testing.assert_array_equal(batch.frequencies, freqs)


def test_wide_window_batches():
    freqs = np.arange(3.5e9, 5.5e9, 5e6)
    batches = plan_batches(freqs, LO_FREQUENCY)

    assert len(batches) > 1
    [anchor] = [b for b in batches if b.anchor]
    assert anchor.lo_frequency == LO_FREQUENCY
    np.testing.assert_array_equal(
        np.unique(np.concatenate([b.frequencies for b in batches])), freqs
    )
    for batch in batches:
        assert np.all(
            np.abs(batch.frequencies - batch.lo_frequency) <= MAX_IF_BANDWIDTH
        )
    for inner, outer in pairwise(batches):
        assert len(np.intersect1d(inner.frequencies, outer.frequencies)) > 0


def test_stitch():
    """A constant IQ offset of a batch is removed, and the overlap is merged."""
    rng = np.random.default_rng(0)
    frequencies = np.arange(30.0)
    clean = rng.normal(size=(4, 30)) + 1j * rng.normal(size=(4, 30))
    offset = 0.3 + 0.2j
    lower, upper = np.arange(20), np.arange(15, 30)
    signal = np.concatenate([clean[:, lower], clean[:, upper] + offset], axis=1)

    merged, iq, steps, _ = stitch(
        np.concatenate([frequencies[lower], frequencies[upper]]),
        np.stack([signal.real, signal.imag], axis=-1),
        np.concatenate([np.zeros(len(lower), int), np.ones(len(upper), int)]),
        anchor=0,
    )

    np.testing.assert_array_equal(merged, frequencies)
    np.testing.assert_allclose(iq[..., 0] + 1j * iq[..., 1], clean)
    assert steps == pytest.approx([0, abs(offset)])


def test_qubit_flux_window_far_from_lo(platform, monkeypatch):
    """A single batch far from the LO retunes it, and is not flagged as a wide scan."""
    executed_updates = _spy_execute(platform, monkeypatch)
    params = QubitFluxParameters.load(
        {
            "frequency": ["linspace", 4.0e9, 4.3e9, 21],
            "bias": ["linspace", -0.2, -0.08, 5],
            "nshots": 10,
        }
    )
    data = qubit_flux_acquisition(params, platform, [0])

    assert not data.wide_scan
    assert len(executed_updates) == 1
    merged = {k: v for update in executed_updates[0] for k, v in update.items()}
    freqs = np.unique(data[0].freq)
    assert freqs.min() == pytest.approx(4.0e9)
    assert merged["0/drive_lo"]["frequency"] == pytest.approx(
        (freqs.min() + freqs.max()) / 2
    )
    assert data.lo_frequencies[0] == [merged["0/drive_lo"]["frequency"]]


def test_qubit_flux_wide_scan(platform):
    """Wide windows are acquired in batches, and stitched on a regular grid."""
    params = QubitFluxParameters.load(
        {
            "freq_width": 2_000_000_000,
            "freq_step": 10_000_000,
            "bias_width": 0.1,
            "bias_step": 0.04,
            "nshots": 10,
        }
    )
    targets = [0, 1]
    data = qubit_flux_acquisition(params, platform, targets)

    assert data.wide_scan
    for q in targets:
        center = platform.config(platform.qubits[q].drive).frequency
        expected = np.arange(*params.frequency_range(center))
        freqs = np.unique(data[q].freq)
        biases = np.unique(data[q].bias)
        np.testing.assert_allclose(freqs, expected)
        assert len(data[q]) == len(freqs) * len(biases)
        assert len(data.lo_frequencies[q]) > 1
        assert LO_FREQUENCY in data.lo_frequencies[q]


def test_qubit_flux_different_bias_lengths(platform):
    """Qubits whose bias sweeps differ in length are not swept in parallel."""
    params = QubitFluxParameters.load(
        {
            "freq_width": 100_000_000,
            "freq_step": 10_000_000,
            "bias_width": 0.1,
            "bias_step": 0.02,
            "nshots": 10,
        }
    )
    targets = [0, 1]
    data = qubit_flux_acquisition(params, platform, targets)

    biases = {}
    for q in targets:
        offset = platform.config(platform.qubits[q].flux).offset
        biases[q] = np.arange(*params.bias_range(offset))
        np.testing.assert_allclose(np.unique(data[q].bias), biases[q])
        assert len(data[q]) == len(np.unique(data[q].freq)) * len(biases[q])
    # the floating point grids around the two sweetspots have different lengths
    assert len(biases[0]) != len(biases[1])


def test_qubit_flux_shared_lo_moved(platform, monkeypatch):
    """Moving an LO shared by multiple qubits is not supported."""
    _share_drive_lo(platform, monkeypatch)
    params = QubitFluxParameters.load(
        {
            "freq_width": 100_000_000,
            "freq_step": 10_000_000,
            "bias_width": 0.1,
            "bias_step": 0.04,
            "nshots": 10,
        }
    )
    with pytest.raises(NotImplementedError):
        qubit_flux_acquisition(params, platform, [0, 1])
