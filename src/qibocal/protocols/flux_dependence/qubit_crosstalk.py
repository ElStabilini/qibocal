from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
)
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import Protocol, QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log

from ...result import magnitude
from ...update import replace
from ..utils import (
    GHZ_TO_HZ,
    HZ_TO_GHZ,
    format_error_single_cell,
    readout_frequency,
    round_report,
    table_dict,
    table_html,
)
from . import utils
from .qubit_flux_dependence import (
    QubitFluxData,
    QubitFluxParameters,
    QubitFluxResults,
    QubitFluxType,
)

__all__ = ["qubit_crosstalk"]

MINIMUM_SWEETSPOT_DISTANCE = 0.05
"""Minimum distance, in flux quanta, between the phase of the target qubit at its bias
point and the closest (anti-)sweetspot.

At a sweetspot the model is even in the bias of the neighbor qubit, therefore the sign of
the crosstalk element is not encoded in the data. Data acquired too close to one of them
is rejected instead of returning an element with a random sign."""

MAXIMUM_PHASE_DRIFT = 0.05
"""Maximum distance, in flux quanta, between the fitted and the calibrated phase of the
target qubit at its bias point before the flux calibration is reported as stale."""


@dataclass
class QubitCrosstalkParameters(QubitFluxParameters):
    """Crosstalk runcard inputs."""

    bias_point: dict[QubitId, float] | None = field(default_factory=dict)
    """Dictionary with {qubit_id: bias_point_qubit_id}."""
    flux_qubits: list[QubitId] | None = None
    """IDs of the qubits that we will sweep the flux on.
    If ``None`` flux will be swept on all qubits that we are running the routine on in a multiplex fashion.
    If given flux will be swept on the given qubits in a sequential fashion (n qubits will result to n different executions).
    Multiple qubits may be measured in each execution as specified by the ``qubits`` option in the runcard.
    """


@dataclass
class QubitCrosstalkData(QubitFluxData):
    """Crosstalk acquisition outputs when ``flux_qubits`` are given."""

    matrix_element: dict[QubitId, float] = field(default_factory=dict)
    """Diagonal flux element."""
    bias_point: dict[QubitId, float] = field(default_factory=dict)
    """Bias point for each qubit."""
    offset: dict[QubitId, float] = field(default_factory=dict)
    """Phase shift for each qubit."""
    qubit_frequency: dict[QubitId, float] = field(default_factory=dict)
    """Qubit frequency for each qubit."""
    data: dict[tuple[QubitId, QubitId], npt.NDArray[QubitFluxType]] = field(
        default_factory=dict
    )
    """Raw data acquired for (qubit, qubit_flux) pairs saved in nested dictionaries."""

    def register_qubit(self, qubit, flux_qubit, freq, bias, signal):
        """Store output for single qubit."""
        ar = utils.create_data_array(freq, bias, signal, dtype=QubitFluxType)
        if (qubit, flux_qubit) in self.data:
            self.data[qubit, flux_qubit] = np.rec.array(
                np.concatenate((self.data[qubit, flux_qubit], ar))
            )
        else:
            self.data[qubit, flux_qubit] = ar


@dataclass
class QubitCrosstalkResults(QubitFluxResults):
    """
    Qubit Crosstalk outputs.
    """

    qubit_frequency_bias_point: dict[QubitId, float] = field(default_factory=dict)
    """Expected qubit frequency at bias point."""
    crosstalk_matrix: dict[QubitId, dict[QubitId, float]] = field(default_factory=dict)
    """Crosstalk matrix element."""
    crosstalk_matrix_error: dict[QubitId, dict[QubitId, float]] = field(
        default_factory=dict
    )
    """Uncertainty on the crosstalk matrix element."""
    fitted_parameters: dict[tuple[QubitId, QubitId], dict] = field(default_factory=dict)
    """Fitted parameters for each couple target-flux qubit."""
    successful_fit: dict[tuple[QubitId, QubitId], bool] = field(default_factory=dict)
    """Flag for each couple target-flux qubit to see whether the fit was successful."""

    def __contains__(self, key: QubitId):
        """Checking if qubit is in crosstalk_matrix attribute."""
        return key in self.crosstalk_matrix


def _acquisition(
    params: QubitCrosstalkParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> QubitCrosstalkData:
    """Data acquisition for Crosstalk Experiment."""

    assert set(targets).isdisjoint(set(params.flux_qubits)), (
        "Flux qubits must be different from targets."
    )

    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    offset = {}
    charging_energy = {}
    matrix_element = {}
    maximum_frequency = {}
    freq_sweepers = []
    offset_sweepers = []

    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        charging_energy[qubit] = platform.calibration.single_qubits[
            qubit
        ].qubit.charging_energy
        qd_channel, qd_pulse = natives.RX()[0]
        ro_channel, ro_pulse = natives.MZ()[0]

        qd_pulse = replace(qd_pulse, duration=params.drive_duration)
        if params.drive_amplitude is not None:
            qd_pulse = replace(qd_pulse, amplitude=params.drive_amplitude)

        qd_pulses[qubit] = qd_pulse
        ro_pulses[qubit] = ro_pulse

        # store calibration parameters
        maximum_frequency[qubit] = platform.calibration.single_qubits[
            qubit
        ].qubit.maximum_frequency
        matrix_element[qubit] = platform.calibration.get_crosstalk_element(qubit, qubit)
        charging_energy[qubit] = platform.calibration.single_qubits[
            qubit
        ].qubit.charging_energy
        offset[qubit] = (
            -platform.calibration.single_qubits[qubit].qubit.sweetspot
            * matrix_element[qubit]
        )

        sequence.append((qd_channel, qd_pulse))
        sequence.append((ro_channel, Delay(duration=qd_pulse.duration)))
        sequence.append((ro_channel, ro_pulse))

        freq_sweepers.append(
            Sweeper(
                parameter=Parameter.frequency,
                range=params.frequency_range(platform.config(qd_channel).frequency),
                channels=[qd_channel],
            )
        )

    for q in params.flux_qubits:
        flux_channel = platform.qubits[q].flux
        offset0 = platform.config(flux_channel).offset
        offset_sweepers.append(
            Sweeper(
                parameter=Parameter.offset,
                range=params.bias_range(offset0),
                channels=[flux_channel],
            )
        )

    data = QubitCrosstalkData(
        resonator_type=platform.resonator_type,
        matrix_element=matrix_element,
        offset=offset,
        qubit_frequency=maximum_frequency,
        charging_energy=charging_energy,
        bias_point=params.bias_point,
    )

    options = {
        "nshots": params.nshots,
        "relaxation_time": params.relaxation_time,
        "acquisition_type": AcquisitionType.INTEGRATION,
        "averaging_mode": AveragingMode.CYCLIC,
    }

    updates = []
    for qubit in targets:
        if qubit in params.bias_point:
            channel = platform.qubits[qubit].flux
            updates.append({channel: {"offset": params.bias_point[qubit]}})

    updates += [
        {platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}}
        for q in targets
    ]

    for flux_qubit, offset_sweeper in zip(params.flux_qubits, offset_sweepers):
        results = platform.execute(
            [sequence], [[offset_sweeper], freq_sweepers], **options, updates=updates
        )

        # retrieve the results for every qubit
        for i, qubit in enumerate(targets):
            result = results[ro_pulses[qubit].id]
            data.register_qubit(
                qubit,
                flux_qubit,
                signal=magnitude(result),
                freq=freq_sweepers[i].values,
                bias=offset_sweeper.values,
            )
    return data


def _phase_bias_point(data: QubitCrosstalkData, target_qubit: QubitId) -> float:
    r"""Phase of the target qubit at its bias point, with the neighbor qubits grounded.

    It corresponds to :math:`V_{ii} (x_i - x_{ss})`, the quantity that fixes the sign of
    the measured crosstalk element through :math:`\sin(2 \pi \varphi_0)`.
    """
    return (
        data.bias_point[target_qubit] * data.matrix_element[target_qubit]
        + data.offset[target_qubit]
    )


def _fit_function(data: QubitCrosstalkData, target_qubit: QubitId):
    """Fit function in terms of the off-diagonal element and of the phase at zero bias.

    Fitting ``crosstalk_element`` and ``offset`` as independent parameters is degenerate:
    the bias of the target qubit only enters the model through the constant
    ``xi * normalization``, which ``offset`` absorbs entirely, and the pairs
    ``(c, offset)`` and ``(-c, n - 2 * xi * normalization - offset)`` describe the same
    curve. Here the whole constant term is fitted as ``phase``, which is restricted to a
    single half period in :func:`_fit` and therefore admits no mirror image.
    """

    def func(x, element, phase):
        return utils.transmon_frequency(
            xi=0,
            xj=x,
            d=0,
            w_max=data.qubit_frequency[target_qubit] * HZ_TO_GHZ,
            offset=phase,
            normalization=1,
            charging_energy=data.charging_energy[target_qubit] * HZ_TO_GHZ,
            crosstalk_element=element,
        )

    return func


def _element_guess(
    frequencies: npt.NDArray,
    biases: npt.NDArray,
    phase: float,
    w_max: float,
    charging_energy: float,
) -> float:
    r"""Initial guess for the off-diagonal element from the local slope of the feature.

    Over a narrow bias window the model is almost linear, with

    .. math::
        \frac{\partial f}{\partial x_j} = - \frac{\pi}{4} (w_{max} + E_c)
        \cos^2(\pi \varphi_0)^{-3/4} \sin(2 \pi \varphi_0) V_{ij}

    which is inverted here. Frequencies and energies are expected in GHz. Without a guess
    the fit starts from :math:`V_{ij} = 1`, orders of magnitude away from a typical
    crosstalk element.
    """
    slope = np.polyfit(biases, frequencies, 1)[0]
    return (
        -4
        * slope
        * (np.cos(np.pi * phase) ** 2) ** 0.75
        / (np.pi * (w_max + charging_energy) * np.sin(2 * np.pi * phase))
    )


def _fit(data: QubitCrosstalkData) -> QubitCrosstalkResults:
    crosstalk_matrix = {qubit: {} for qubit in data.qubit_frequency}
    crosstalk_matrix_error = {qubit: {} for qubit in data.qubit_frequency}
    fitted_parameters = {}
    qubit_frequency_bias_point = {}
    successful_fit = {}

    for target_flux_qubit, qubit_data in data.data.items():
        target_qubit, flux_qubit = target_flux_qubit
        successful_fit[target_flux_qubit] = False

        if target_qubit not in data.bias_point:
            log.error(
                f"Error in qubit_crosstalk protocol fit: no bias point provided "
                f"for qubit {target_qubit}."
            )
            continue

        phase = _phase_bias_point(data, target_qubit)
        # At a (anti-)sweetspot the model is even in the bias of the neighbor qubit and
        # the sign of the crosstalk element cannot be recovered from the data.
        if abs(phase - np.round(2 * phase) / 2) < MINIMUM_SWEETSPOT_DISTANCE:
            log.error(
                f"Error in qubit_crosstalk protocol fit: qubit {target_qubit} is biased "
                f"too close to a sweetspot (phase {phase:.3f}), the sign of the "
                f"crosstalk element is not determined. Repeat the acquisition with a "
                f"different bias point."
            )
            continue

        frequencies, biases = utils.flux_extract_feature(
            qubit_data.freq,
            qubit_data.bias,
            qubit_data.signal,
            data.resonator_type == "2D",
        )

        if frequencies is None or biases is None:
            continue

        qubit_frequency_bias_point[target_qubit] = (
            utils.transmon_frequency(
                xi=data.bias_point[target_qubit],
                xj=0,
                d=0,
                w_max=data.qubit_frequency[target_qubit] * HZ_TO_GHZ,
                offset=data.offset[target_qubit],
                normalization=data.matrix_element[target_qubit],
                charging_energy=data.charging_energy[target_qubit] * HZ_TO_GHZ,
                crosstalk_element=1,
            )
            * GHZ_TO_HZ
        )

        # The degeneracy is a reflection around the (anti-)sweetspots, so restricting
        # the phase to the half period selected by the calibration removes both the
        # mirror solution and the aliases, while still allowing the fit to absorb the
        # drift of the flux calibration.
        half_period = np.floor(2 * phase)
        try:
            popt, pcov = curve_fit(
                _fit_function(data, target_qubit),
                biases,
                frequencies * HZ_TO_GHZ,
                p0=(
                    _element_guess(
                        frequencies * HZ_TO_GHZ,
                        biases,
                        phase,
                        w_max=data.qubit_frequency[target_qubit] * HZ_TO_GHZ,
                        charging_energy=data.charging_energy[target_qubit] * HZ_TO_GHZ,
                    ),
                    phase,
                ),
                bounds=(
                    (-np.inf, half_period / 2),
                    (np.inf, (half_period + 1) / 2),
                ),
                maxfev=100000,
            )
            element, fitted_phase = float(popt[0]), float(popt[1])
            if abs(fitted_phase - phase) > MAXIMUM_PHASE_DRIFT:
                log.warning(
                    f"Fitted phase of qubit {target_qubit} deviates by "
                    f"{fitted_phase - phase:.3f} from the calibrated one: the flux "
                    f"calibration may be outdated and the sign of the crosstalk "
                    f"element unreliable."
                )
            fitted_parameters[target_qubit, flux_qubit] = {
                "xi": data.bias_point[target_qubit],
                "d": 0,
                "w_max": data.qubit_frequency[target_qubit] * HZ_TO_GHZ,
                # rewrite the fitted phase as an offset, so that the fitted parameters
                # can be fed back to utils.transmon_frequency when plotting
                "offset": fitted_phase
                - data.bias_point[target_qubit] * data.matrix_element[target_qubit],
                "normalization": data.matrix_element[target_qubit],
                "charging_energy": data.charging_energy[target_qubit] * HZ_TO_GHZ,
                "crosstalk_element": element / data.matrix_element[target_qubit],
            }
            crosstalk_matrix[target_qubit][flux_qubit] = element
            crosstalk_matrix_error[target_qubit][flux_qubit] = float(
                np.sqrt(pcov[0, 0])
            )
            successful_fit[target_flux_qubit] = True
        except (RuntimeError, ValueError) as e:  # pragma: no cover
            log.error(f"Error in qubit_crosstalk protocol fit: {e} ")

    return QubitCrosstalkResults(
        qubit_frequency_bias_point=qubit_frequency_bias_point,
        crosstalk_matrix=crosstalk_matrix,
        crosstalk_matrix_error=crosstalk_matrix_error,
        fitted_parameters=fitted_parameters,
        successful_fit=successful_fit,
    )


def _plot(data: QubitCrosstalkData, fit: QubitCrosstalkResults, target: QubitId):
    """Plotting function for Crosstalk Experiment."""
    figures, fitting_report = utils.flux_crosstalk_plot(
        data, target, fit, fit_function=utils.transmon_frequency
    )
    if fit is not None and any(
        success for (qubit, _), success in fit.successful_fit.items() if qubit == target
    ):
        labels = [
            "Qubit Frequency at Bias point [Hz]",
        ]
        values = [
            np.round(fit.qubit_frequency_bias_point[target], 4),
        ]
        for flux_qubit in fit.crosstalk_matrix[target]:
            if flux_qubit != target:
                labels.append(f"Crosstalk with qubit {flux_qubit} [V^-1]")
            else:
                labels.append("Flux dependence [V^-1]")
            values.append(
                format_error_single_cell(
                    round_report(
                        [
                            (
                                fit.crosstalk_matrix[target][flux_qubit],
                                fit.crosstalk_matrix_error[target][flux_qubit],
                            )
                        ]
                    )
                )
            )
        fitting_report = table_html(
            table_dict(
                target,
                labels,
                values,
            )
        )
    return figures, fitting_report


def _update(
    results: QubitCrosstalkResults, platform: CalibrationPlatform, qubit: QubitId
):
    """Update crosstalk matrix."""

    for flux_qubit, element in results.crosstalk_matrix[qubit].items():
        update.crosstalk_matrix(element, platform, qubit, flux_qubit)


qubit_crosstalk = Protocol(_acquisition, _fit, _plot, _update)
"""Qubit crosstalk Protocol object"""
