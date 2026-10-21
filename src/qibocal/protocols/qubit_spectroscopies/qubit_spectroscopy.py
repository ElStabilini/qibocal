from dataclasses import dataclass, field
from typing import cast

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy.constants
from plotly.subplots import make_subplots
from qibolab import IqConfig, PulseSequence
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Protocol, QubitId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.lo_sweeps import (
    acquire,
    drive_sequence,
    plan_batches,
    plan_drive_sweeps,
    stitch,
)
from qibocal.protocols.utils import (
    Range,
    RangeLike,
    lorentzian,
    table_dict,
    table_html,
    to_range,
)
from qibocal.result import magnitude, phase

__all__ = [
    "QubitSpectroscopyData",
    "QubitSpectroscopyParameters",
    "QubitSpectroscopyResults",
    "_fit",
    "qubit_spectroscopy",
]


@dataclass
class QubitSpectroscopyParameters(Parameters):
    """QubitSpectroscopy runcard inputs."""

    frequency: RangeLike | None = None
    """Drive frequency [Hz] range for sweep."""
    freq_width: int | None = None
    """Width [Hz] for frequency sweep relative  to the qubit frequency."""
    freq_step: int | None = None
    """Frequency [Hz] step for sweep."""
    drive_duration: int = 4000
    """Drive pulse duration [ns]. Same for all qubits."""
    drive_amplitude: float = 1.0
    """Drive pulse amplitude (optional). Same for all qubits."""

    def frequency_range(self, center: float = 0.0) -> Range:
        def legacy_range() -> Range:
            assert self.freq_width is not None and self.freq_step is not None
            return (
                center - self.freq_width / 2,
                center + self.freq_width / 2,
                self.freq_step,
            )

        return (
            to_range(self.frequency, center=center)
            if self.frequency is not None
            else legacy_range()
        )

    def f0(self, center: float = 0.0) -> float:
        """Return the center frequency of the sweep."""
        start, end, _ = self.frequency_range(center)
        return (start + end) / 2


@dataclass
class QubitSpectroscopyResults(Results):
    """QubitSpectroscopy outputs."""

    frequency: dict[QubitId, float]
    """Drive frequecy [GHz] for each qubit."""
    amplitude: dict[QubitId, float]
    """Input drive amplitude. Same for all qubits."""
    fitted_parameters: dict[QubitId, list[float]]
    """Raw fitting output."""
    lo_frequencies: dict[QubitId, list[float | None]] = field(default_factory=dict)
    """LO frequency [Hz] used in each batch."""
    steps: dict[QubitId, list[float]] = field(default_factory=dict)
    """Signal step [a.u.] of each batch with respect to its inner neighbour."""
    flagged: dict[QubitId, list[bool]] = field(default_factory=dict)
    """Whether the step of each batch is significant."""


@dataclass
class QubitSpectroscopyData(Data):
    """QubitSpectroscopy acquisition outputs."""

    drive_frequencies: dict[QubitId, list[float]]
    """Frequencies."""
    amplitudes: dict[QubitId, float]
    """Amplitudes provided by the user."""
    data: dict[QubitId, npt.NDArray[np.float64]]
    """Raw data acquired."""
    batch_indices: dict[QubitId, list[int]] = field(default_factory=dict, kw_only=True)
    """Batch index of every point."""
    lo_frequencies: dict[QubitId, list[float | None]] = field(
        default_factory=dict, kw_only=True
    )
    """LO frequency [Hz] used in each batch."""
    anchor_batch: dict[QubitId, int] = field(default_factory=dict, kw_only=True)
    """Index of the batch the others are aligned to."""

    def signal(self, qubit: QubitId) -> npt.NDArray[np.float64]:
        return magnitude(self.data[qubit])

    def phase(self, qubit: QubitId) -> npt.NDArray[np.float64]:
        return phase(self.data[qubit])


def _acquisition(
    params: QubitSpectroscopyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> QubitSpectroscopyData:
    """Data acquisition for qubit spectroscopy.

    Handles wideband spectroscopy by splitting the sweep of each qubit in batches,
    each within the IF bandwidth of its LO.
    """

    sequences: dict[QubitId, PulseSequence] = {}
    frequencies: dict[QubitId, npt.NDArray[np.float64]] = {}
    for qubit in targets:
        drive_channel = platform.qubits[qubit].drive
        assert drive_channel is not None
        sequences[qubit] = drive_sequence(
            platform, qubit, params.drive_amplitude, params.drive_duration
        )
        frequencies[qubit] = np.arange(
            *params.frequency_range(
                cast(IqConfig, platform.config(drive_channel)).frequency
            )
        )

    sweeps = plan_drive_sweeps(platform, frequencies)
    results = acquire(
        platform,
        sequences,
        sweeps,
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
    )

    return QubitSpectroscopyData(
        drive_frequencies={q: results[q].frequencies.tolist() for q in targets},
        # TODO: remove, and propagate differently, since it is always the same for
        # all qubits
        amplitudes={q: params.drive_amplitude for q in targets},
        data={q: results[q].iq for q in targets},
        batch_indices={q: results[q].batch.tolist() for q in targets},
        lo_frequencies={q: sweeps[q].lo_frequencies for q in targets},
        anchor_batch={q: sweeps[q].anchor for q in targets},
    )


def _lorentzian_with_offset(frequency, amplitude, center, sigma, offset):
    return lorentzian(frequency, amplitude, center, sigma) + offset


def _guess_initial_parameters(signal, frequencies, is_peak):
    k = max(1, int(0.1 * len(signal)))
    guess_offset = np.mean(np.concatenate([signal[:k], signal[-k:]]))

    guess_offset = (signal[0] + signal[-1]) / 2
    signal_no_background = signal - guess_offset

    sign = 1 if is_peak else -1
    signal_flipped = signal_no_background * sign
    guess_center = frequencies[np.argmax(signal_flipped)]
    guess_peak_height = signal_flipped.max() * sign
    indices_beyond_half = np.where(signal_flipped > signal_flipped.max() / 2)[0]

    if len(indices_beyond_half) >= 1:
        guess_sigma = (
            frequencies[indices_beyond_half[-1]] - frequencies[indices_beyond_half[0]]
        ) / 2
    else:
        # if there is no clear peak, we give a high flexibility
        guess_sigma = frequencies[-1] - frequencies[0]

    guess_amp = guess_peak_height * guess_sigma * np.pi

    return [
        guess_amp,
        guess_center,
        guess_sigma,
        guess_offset,
    ]


def _lorentzian_fit(
    frequencies: np.ndarray, signal: np.ndarray
) -> tuple[float, list[float], list[float]]:
    freq_domain_size = frequencies[-1] - frequencies[0]

    # Try to fit both peak and dip, and pick the best one
    best_fit = None
    for is_peak in [True, False]:
        initial_parameters = _guess_initial_parameters(signal, frequencies, is_peak)
        bounds = (
            [0.0 if is_peak else -np.inf, frequencies[0], 0.0, -np.inf],
            [np.inf if is_peak else 0.0, frequencies[-1], freq_domain_size, np.inf],
        )

        # fit the model with the data and guessed parameters
        try:
            fit_parameters, parameters_cov = curve_fit(
                _lorentzian_with_offset,
                frequencies,
                signal,
                p0=initial_parameters,
                bounds=bounds,
            )
        except RuntimeError:
            continue

        sum_sq_residuals = float(
            np.sum(
                (signal - _lorentzian_with_offset(frequencies, *fit_parameters)) ** 2
            )
        )

        if best_fit is None or sum_sq_residuals < best_fit["sum_sq_residuals"]:
            best_fit = {
                "fit_parameters": fit_parameters,
                "parameters_cov": parameters_cov,
                "sum_sq_residuals": sum_sq_residuals,
            }

    if best_fit is None:
        raise RuntimeError("fit failed")

    # The output results are stored in a json, but ndarray is not JSON serializable,
    # so the parameters are converted to list.
    parameter_errors = np.sqrt(np.diag(best_fit["parameters_cov"])).tolist()
    model_parameters = best_fit["fit_parameters"].tolist()
    return (
        model_parameters[1],
        model_parameters,
        parameter_errors,
    )


def _stitch(
    data: QubitSpectroscopyData, qubit: QubitId, step_threshold: float = 3
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[float], list[bool]]:
    """Align the batches of ``qubit`` to its anchor batch, and merge them.

    See :func:`qibocal.protocols.lo_sweeps.stitch`.
    """
    frequencies = np.array(data.drive_frequencies[qubit])
    iq = np.asarray(data.data[qubit])
    if qubit not in data.batch_indices:
        return frequencies, iq, [], []

    return stitch(
        frequencies,
        iq,
        np.array(data.batch_indices[qubit]),
        data.anchor_batch[qubit],
        step_threshold,
    )


def _fit(data: QubitSpectroscopyData) -> QubitSpectroscopyResults:
    """Post-processing function for QubitSpectroscopy."""
    qubits = data.qubits
    frequency: dict[QubitId, float] = {}
    params: dict[QubitId, list[float]] = {}
    steps: dict[QubitId, list[float]] = {}
    flagged: dict[QubitId, list[bool]] = {}
    for qubit in qubits:
        frequencies, iq, steps[qubit], flagged[qubit] = _stitch(data, qubit)
        fit_result = _lorentzian_fit(frequencies, magnitude(iq))
        frequency[qubit], params[qubit], _ = fit_result
    return QubitSpectroscopyResults(
        frequency=frequency,
        fitted_parameters=params,
        amplitude=data.amplitudes,
        lo_frequencies={q: data.lo_frequencies.get(q, []) for q in qubits},
        steps=steps,
        flagged=flagged,
    )


def _plot(data: QubitSpectroscopyData, target: QubitId, fit: QubitSpectroscopyResults):
    """Plot QubitSpectroscopy.

    A single figure is built with two side-by-side panels: the signal
    magnitude and the signal phase, after stitching the batches. A Lorentzian
    fit is overlaid on the magnitude panel when a fit is available. For sweeps
    acquired in multiple batches, the raw data and the batch boundaries are
    shown as well.
    """
    fitting_report = ""
    frequencies, iq, *_ = _stitch(data, target)

    fig = make_subplots(
        rows=1,
        cols=2,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        subplot_titles=(
            "Signal",
            "Phase",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=frequencies * scipy.constants.nano,  # plotting in GHz
            y=magnitude(iq),
            opacity=1,
            name="Signal",
            showlegend=True,
            legendgroup="Signal",
            mode="markers",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frequencies * scipy.constants.nano,  # plotting in GHz
            y=phase(iq),
            opacity=1,
            name="Phase",
            showlegend=True,
            legendgroup="Phase",
            mode="markers",
        ),
        row=1,
        col=2,
    )

    if fit is not None:
        freqrange = np.linspace(
            min(frequencies),
            max(frequencies),
            2 * len(frequencies),
        )
        params = fit.fitted_parameters[target]
        fig.add_trace(
            go.Scatter(
                x=freqrange * scipy.constants.nano,  # plotting in GHz
                y=_lorentzian_with_offset(freqrange, *params),
                name="Fit",
                mode="lines",
            ),
            row=1,
            col=1,
        )

        if data.amplitudes[target] is not None:
            labels = ["Qubit Frequency [Hz]", "Drive Amplitude [a.u.]"]
            values = [fit.frequency[target], data.amplitudes[target]]

            fitting_report = table_html(
                table_dict(
                    target,
                    labels,
                    values,
                )
            )

        if len(fit.lo_frequencies[target]) > 0:
            fitting_report += table_html(
                {
                    "Batch": list(range(len(fit.lo_frequencies[target]))),
                    "LO Frequency [Hz]": fit.lo_frequencies[target],
                    "Step [a.u.]": fit.steps[target],
                    "Flagged": fit.flagged[target],
                }
            )

    fig.update_layout(
        showlegend=True,
        yaxis_title="Signal [a.u.]",
        xaxis_title="Frequency [GHz]",
        yaxis2_title="Phase [rad]",
        xaxis2_title="Frequency [GHz]",
    )

    return [fig], fitting_report


def _update(
    results: QubitSpectroscopyResults, platform: CalibrationPlatform, target: QubitId
):
    frequency = results.frequency[target]
    platform.calibration.single_qubits[target].qubit.frequency_01 = frequency
    drive_channel = platform.qubits[target].drive
    lo = platform.channels[drive_channel].lo
    if lo is not None:
        lo_frequency = platform.config(lo).frequency
        # the planner moves the LO only if the frequency is beyond its IF bandwidth
        [batch] = plan_batches(np.array([frequency]), lo_frequency)
        if batch.lo_frequency != lo_frequency:
            # move the LO, preserving the current intermediate frequency of the drive
            intermediate_frequency = (
                cast(IqConfig, platform.config(drive_channel)).frequency - lo_frequency
            )
            update.lo_frequency(
                frequency - intermediate_frequency, platform, target, "drive"
            )
            log.warning(
                f"Drive LO {lo} of qubit {target} moved to "
                f"{frequency - intermediate_frequency:.0f} Hz: its mixer calibration is "
                "now stale."
            )
    update.drive_frequency(frequency, platform, target)


qubit_spectroscopy = Protocol(_acquisition, _fit, _plot, _update)
"""Qubit Spectroscopy routine.
"""
