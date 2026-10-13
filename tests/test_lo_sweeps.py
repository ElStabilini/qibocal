"""Test the retuning of the drive LOs in qubit spectroscopy-like acquisitions.

In the mock platform both drive LOs are at 4.9 GHz.
"""

import numpy as np
import pytest
from qibolab._core.components import IqChannel

from qibocal.protocols.flux_dependence.qubit_flux_dependence import (
    QubitFluxParameters,
)
from qibocal.protocols.flux_dependence.qubit_flux_dependence import (
    _acquisition as qubit_flux_acquisition,
)
from qibocal.protocols.qubit_spectroscopies.qubit_spectroscopy import (
    _drive_lo_updates,
)


def _share_drive_lo(platform, monkeypatch):
    """Drive qubit 1 with the LO of qubit 0."""
    channels = platform.instruments["dummy"].channels
    monkeypatch.setitem(
        channels, platform.qubits[1].drive, IqChannel(mixer=None, lo="0/drive_lo")
    )


def test_lo_kept_within_bandwidth(platform):
    freqs = np.linspace(4.7e9, 4.8e9, 11)
    assert _drive_lo_updates(platform, {0: freqs}) == {}


def test_lo_retuned_outside_bandwidth(platform):
    """Narrow window, which does not require batching, but too far from the LO."""
    freqs = np.linspace(4.2e9, 4.64e9, 201)
    updates = _drive_lo_updates(platform, {0: freqs})
    assert updates == {
        "0/drive_lo": {"frequency": pytest.approx(4.42e9)},
        platform.qubits[0].drive: {"frequency": pytest.approx(4.42e9)},
    }


def test_lo_window_too_wide(platform):
    freqs = np.linspace(4.0e9, 4.7e9, 11)
    with pytest.raises(ValueError):
        _drive_lo_updates(platform, {0: freqs})


def test_shared_lo_retuned(platform, monkeypatch):
    _share_drive_lo(platform, monkeypatch)
    updates = _drive_lo_updates(
        platform,
        {0: np.linspace(4.0e9, 4.1e9, 11), 1: np.linspace(4.2e9, 4.3e9, 11)},
    )
    assert updates == {
        "0/drive_lo": {"frequency": pytest.approx(4.15e9)},
        platform.qubits[0].drive: {"frequency": pytest.approx(4.15e9)},
        platform.qubits[1].drive: {"frequency": pytest.approx(4.15e9)},
    }


def test_shared_lo_too_far(platform, monkeypatch):
    """Qubit 0 alone would keep the LO, but qubit 1 cannot be covered with it."""
    _share_drive_lo(platform, monkeypatch)
    with pytest.raises(ValueError):
        _drive_lo_updates(
            platform,
            {0: np.linspace(4.7e9, 4.8e9, 11), 1: np.linspace(4.0e9, 4.1e9, 11)},
        )


def test_qubit_flux_window_far_from_lo(platform, monkeypatch):
    """A single batch far from the LO retunes it, and is not flagged as a wide scan."""
    executed_updates = []
    execute = platform.execute

    def spy(*args, updates=None, **kwargs):
        executed_updates.append(updates)
        return execute(*args, updates=updates, **kwargs)

    monkeypatch.setattr(platform, "execute", spy)
    params = QubitFluxParameters.load(
        {
            "frequency": ["linspace", 4.2e9, 4.64e9, 21],
            "bias": ["linspace", -0.2, -0.08, 5],
            "nshots": 10,
        }
    )
    data = qubit_flux_acquisition(params, platform, [0])

    assert not data.wide_scan
    assert len(executed_updates) == 1
    merged = {k: v for update in executed_updates[0] for k, v in update.items()}
    freqs = np.unique(data[0].freq)
    assert freqs.min() == pytest.approx(4.2e9)
    assert merged["0/drive_lo"]["frequency"] == pytest.approx(
        (freqs.min() + freqs.max()) / 2
    )
