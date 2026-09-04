from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, Parameter, PulseSequence, Sweeper

from qibocal.calibration import CalibrationPlatform

from ... import update
from ...auto.operation import Data, Protocol, QubitId, Results
from ...config import log
from ...result import magnitude
from ..utils import (
    readout_frequency,
    table_dict,
    table_html,
)
from . import utils

__all__ = ["ResonatorFluxParameters", "resonator_flux"]


@dataclass
class ResonatorFluxParameters(utils.FluxFrequencySweepParameters):
    """ResonatorFlux runcard inputs."""

    bias_center: float | None = None
    freq_center: float | None = None


@dataclass
class ResonatorFluxResults(Results):
    """ResonatoFlux outputs."""

    frequency: dict[QubitId, float] = field(default_factory=dict)
    """Readout frequency."""
    coupling: dict[QubitId, float] = field(default_factory=dict)
    """Qubit-resonator coupling."""
    asymmetry: dict[QubitId, float] = field(default_factory=dict)
    """Asymmetry between junctions."""
    sweetspot: dict[QubitId, float] = field(default_factory=dict)
    """Sweetspot for each qubit."""
    matrix_element: dict[QubitId, float] = field(default_factory=dict)
    """Sweetspot for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]] = field(default_factory=dict)
    """Optimal parameters found from the fit."""
    successful_fit: dict[QubitId, bool] = field(default_factory=dict)
    """flag for each qubit to see whether the fit was successful."""
    # The attributes below are only for visualizing the inliers/outliers for debugging
    peak_biases: dict[QubitId, list[float]] = field(default_factory=dict)
    """Bias of extracted peaks (for visualization)."""
    peak_frequencies: dict[QubitId, list[int]] = field(default_factory=dict)
    """Frequency of extracted peaks (for visualization)."""
    inliers: dict[QubitId, list[bool]] = field(default_factory=dict)
    """Boolean mask indicating which peaks are inliers (for visualization)."""


ResFluxType = np.dtype(
    [
        ("freq", np.float64),
        ("bias", np.float64),
        ("signal", np.float64),
    ]
)
"""Custom dtype for resonator flux dependence."""


@dataclass
class ResonatorFluxData(Data):
    """ResonatorFlux acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    qubit_frequency: dict[QubitId, float] = field(default_factory=dict)
    """Qubit frequencies."""
    bare_resonator_frequency: dict[QubitId, int] = field(default_factory=dict)
    """Qubit bare resonator frequency power provided by the user."""
    charging_energy: dict[QubitId, float] = field(default_factory=dict)
    """Qubit charging energy in Hz."""
    data: dict[QubitId, npt.NDArray[ResFluxType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, bias, signal):
        """Store output for single qubit."""
        self.data[qubit] = utils.create_data_array(
            freq, bias, signal, dtype=ResFluxType
        )

    # TODO: fix this temporary solution
    @property
    def find_min(self) -> bool:
        """True when the resonator dip (rather than a peak) should be tracked.

        Only a 2D resonator readout shows a dip in transmission; everything
        else is treated as a peak. Used by ``utils.extract_trace``.
        """
        return self.resonator_type == "2D"


def _acquisition(
    params: ResonatorFluxParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ResonatorFluxData:
    """Data acquisition for ResonatorFlux experiment."""

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qubit_frequency = {}
    bare_resonator_frequency = {}
    charging_energy = {}
    matrix_element = {}
    offset = {}
    freq_sweepers = []
    offset_sweepers = []
    for q in targets:
        ro_sequence = platform.natives.single_qubit[q].MZ()
        ro_pulses[q] = ro_sequence[0][1]
        sequence += ro_sequence

        qubit = platform.qubits[q]
        offset0 = platform.config(qubit.flux).offset

        freq_sweepers.append(
            Sweeper(
                parameter=Parameter.frequency,
                range=params.frequency_range(readout_frequency(q, platform)),
                channels=[qubit.probe],
            )
        )
        offset_sweepers.append(
            Sweeper(
                parameter=Parameter.offset,
                range=params.bias_range(offset0),
                channels=[qubit.flux],
            )
        )

        qubit_frequency[q] = platform.config(qubit.drive).frequency
        bare_resonator_frequency[q] = platform.calibration.single_qubits[
            q
        ].resonator.bare_frequency
        matrix_element[q] = platform.calibration.get_crosstalk_element(q, q)
        offset[q] = -offset0 * matrix_element[q]
        charging_energy[q] = platform.calibration.single_qubits[q].qubit.charging_energy

    data = ResonatorFluxData(
        resonator_type=platform.resonator_type,
        qubit_frequency=qubit_frequency,
        bare_resonator_frequency=bare_resonator_frequency,
        charging_energy=charging_energy,
    )
    results = platform.execute(
        [sequence],
        [offset_sweepers, freq_sweepers],
        updates=[{platform.qubits[q].flux: {"offset": 0.0}} for q in targets],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    # retrieve the results for every qubit
    for i, qubit in enumerate(targets):
        result = results[ro_pulses[qubit].id]
        data.register_qubit(
            qubit,
            signal=magnitude(result),
            freq=freq_sweepers[i].values,
            bias=offset_sweepers[i].values,
        )
    return data


def _fit(data: ResonatorFluxData) -> ResonatorFluxResults:
    """PostProcessing for resonator_flux protocol.

    The fitting procedure requires the knowledge of the bare resonator frequency, the
    charging energy Ec and the maximum qubit frequency which is assumed to be the
    frequency at which the qubit is placed.

    The protocol aims at extracting the sweetspot, the flux coefficient, the coupling,
    the asymmetry and the dressed resonator frequency.
    """

    coupling = {}
    resonator_freq = {}
    asymmetry = {}
    fitted_parameters = {}
    sweetspot = {}
    matrix_element = {}
    successful_fit = {}
    peak_biases_dict = {}
    peak_frequencies_dict = {}
    inliers_dict = {}

    for qubit in data.qubits:
        successful_fit[qubit] = False
        qubit_data = data[qubit]

        try:
            peak_frequencies, peak_biases, inliers_mask = utils.extract_trace(
                qubit_data.freq, qubit_data.bias, qubit_data.signal, data.find_min
            )
            # Store peak coordinates and inliers/outliers for plotting, even if
            # the fit below ends up failing.
            peak_biases_dict[qubit] = peak_biases.tolist()
            peak_frequencies_dict[qubit] = peak_frequencies.tolist()
            inliers_dict[qubit] = inliers_mask.tolist()

            params, _ = utils.fit_qubit(
                qubit,
                w_max=data.qubit_frequency[qubit],
                bare_resonator_frequency=data.bare_resonator_frequency.get(qubit, 0),
                charging_energy=data.charging_energy.get(qubit, 0),
                frequencies=peak_frequencies[inliers_mask],
                biases=peak_biases[inliers_mask],
            )
            offset, normalization = params["offset"], params["normalization"]

            fitted_parameters[qubit] = {
                "w_max": data.qubit_frequency[qubit],
                "xj": 0,
                "d": params["d"],
                "normalization": normalization,
                "offset": offset,
                "crosstalk_element": 1,
                "charging_energy": params["charging_energy"],
                "resonator_freq": params["resonator_freq"],
                "g": params["g"],
            }

            bias_min = np.min(qubit_data.bias)
            bias_max = np.max(qubit_data.bias)
            sweetspot[qubit] = utils.select_sweetspot(
                offset, normalization, (bias_min, bias_max), max_distance=0.3
            )
            if not bias_min <= sweetspot[qubit] <= bias_max:
                log.warning(
                    f"[resonator_flux] qubit {qubit}: fitted sweetspot "
                    f"{sweetspot[qubit]:.4f} V is outside the swept range "
                    f"[{bias_min:.4f}, {bias_max:.4f}] V. The arc extremum was "
                    "not measured, so this value is an extrapolation - widen "
                    "bias_width and re-run."
                )

            resonator_freq[qubit] = utils.transmon_readout_frequency(
                xi=sweetspot[qubit], **fitted_parameters[qubit]
            )
            matrix_element[qubit] = normalization
            coupling[qubit] = params["g"]
            asymmetry[qubit] = params["d"]
            successful_fit[qubit] = True

        except (ValueError, RuntimeError) as e:
            successful_fit[qubit] = False
            log.error(f"Error in resonator_flux protocol fit: {e} ")

    return ResonatorFluxResults(
        frequency=resonator_freq,
        coupling=coupling,
        matrix_element=matrix_element,
        sweetspot=sweetspot,
        asymmetry=asymmetry,
        fitted_parameters=fitted_parameters,
        successful_fit=successful_fit,
        peak_biases=peak_biases_dict,
        peak_frequencies=peak_frequencies_dict,
        inliers=inliers_dict,
    )


def _plot(data: ResonatorFluxData, fit: ResonatorFluxResults, target: QubitId):
    """Plotting function for ResonatorFlux Experiment."""

    inliers_data = None
    outliers_data = None
    if fit is not None and target in fit.peak_biases:
        peak_biases = np.asarray(fit.peak_biases.get(target, []))
        peak_frequencies = np.asarray(fit.peak_frequencies.get(target, []))
        coordinates_all_peaks = np.column_stack([peak_biases, peak_frequencies])

        inliers_mask = np.asarray(fit.inliers.get(target, []), dtype=bool)

        inliers_data = coordinates_all_peaks[inliers_mask]
        outliers_data = coordinates_all_peaks[~inliers_mask]

    figures = utils.flux_dependence_plot(
        data,
        fit,
        target,
        fit_function=utils.transmon_readout_frequency,
        inliers=inliers_data,
        outliers=outliers_data,
    )

    if fit is not None and fit.successful_fit[target]:
        fitting_report = table_html(
            table_dict(
                target,
                [
                    "Coupling g [Hz]",
                    "Dressed resonator freq [Hz]",
                    "Asymmetry",
                    "Sweetspot [V]",
                    "Flux dependence [V]^-1",
                    "Chi [Hz]",
                ],
                [
                    np.round(fit.coupling[target], 2),
                    np.round(fit.frequency[target], 6),
                    np.round(fit.asymmetry[target], 3),
                    np.round(fit.sweetspot[target], 4),
                    np.round(fit.matrix_element[target], 4),
                    np.round(
                        (data.bare_resonator_frequency[target] - fit.frequency[target]),
                        2,
                    ),
                ],
            )
        )
        return figures, fitting_report
    return figures, ""


def _update(
    results: ResonatorFluxResults, platform: CalibrationPlatform, qubit: QubitId
):
    if results.successful_fit[qubit]:
        update.dressed_resonator_frequency(results.frequency[qubit], platform, qubit)
        update.readout_frequency(results.frequency[qubit], platform, qubit)
        update.readout_coupling(results.coupling[qubit], platform, qubit)
        update.flux_offset(results.sweetspot[qubit], platform, qubit)
        update.sweetspot(results.sweetspot[qubit], platform, qubit)


resonator_flux = Protocol(_acquisition, _fit, _plot, _update)
"""ResonatorFlux Protocol object."""
