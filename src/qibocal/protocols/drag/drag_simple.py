from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Drag,
    Pulse,
    PulseSequence,
    Readout,
)
from qibolab._core.native import rotation
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import Protocol, QubitId, Results
from qibocal.calibration import CalibrationPlatform

from ..utils import table_dict, table_html
from .drag import (
    DragTuningData,
    DragTuningParameters,
    DragTuningType,
)

SEQUENCES = ["YpX9", "XpY9"]
"""Sequences used to fit drag parameter."""

__all__ = ["drag_simple"]


@dataclass
class DragTuningSimpleParameters(DragTuningParameters):
    """DragTuningSimple runcard inputs."""

    rx90: bool = False
    """If true the beta of the RX90 native pulse is calibrated, otherwise the one of RX."""

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.rx90, bool):
            raise TypeError(f"rx90 must be boolean, got {type(self.rx90).__name__}")


@dataclass
class DragTuningSimpleResults(Results):
    """DragTuningSimple outputs."""

    betas: dict[QubitId, float]
    """Optimal beta paramter for each qubit."""
    fitted_parameters: dict[tuple[QubitId, str], list[float]]
    """Raw fitting output: mapping qubit to setup to fit parameters [a, b] for ax+b."""
    rx90: bool = False
    """Whether the beta of RX90 (True) or RX (False) has been calibrated."""

    def __contains__(self, key):
        return key in self.betas


@dataclass
class DragTuningSimpleData(DragTuningData):
    """DragTuningSimple acquisition outputs."""

    rx90: bool = False
    """Whether the beta of RX90 (True) or RX (False) is being calibrated."""

    def __getitem__(self, key: tuple[QubitId, str]):
        # TODO: the input type breaks Liskov so probably the base class AbstractData
        # should be reconsidered.
        assert isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], str)
        qubit, setup = key
        if setup == "YpX9":
            return self.data[qubit][::2]
        return self.data[qubit][1::2]


def add_drag(pulse: Pulse, beta: float) -> Pulse:
    """Add DRAG component to Gaussian Pulse."""
    assert isinstance(pulse.envelope, Drag)
    return pulse.model_copy(
        update={"envelope": Drag(rel_sigma=pulse.envelope.rel_sigma, beta=beta)}
    )


def _rotation(natives, theta: float, phi: float, rx90: bool) -> PulseSequence:
    """Rotation built only from the native pulse whose beta is calibrated.

    ``natives.R`` would pick RX90 whenever available, which is not what we want when
    calibrating RX.
    """
    if rx90:
        return rotation(natives.RX90(), theta, phi, rx90=True)
    return rotation(natives.RX(), theta, phi)


def _acquisition(
    params: DragTuningSimpleParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> DragTuningSimpleData:
    """Acquisition function for DRAG experiments.

    We execute two sequences YpX9 and XpY9 following
    https://rsl.yale.edu/sites/default/files/2024-08/2011-RSL-Thesis-Matthew-Reed.pdf
    for different value of the DRAG parameter.
    """

    data = DragTuningSimpleData(rx90=params.rx90)
    beta_param_range = np.arange(*params.beta_range).tolist()

    sequences, all_ro_pulses = [], []
    for beta in beta_param_range:
        for setup in SEQUENCES:
            sequence = PulseSequence()
            ro_pulses = {}
            for q in targets:
                natives = platform.natives.single_qubit[q]
                ro_channel, ro_pulse = natives.MZ()[0]
                qd_channel = platform.qubits[q].drive
                if setup == "YpX9":
                    ry_sequence = _rotation(natives, np.pi, np.pi / 2, params.rx90)
                    rx90_sequence = _rotation(natives, np.pi / 2, 0.0, params.rx90)
                    for channel, pulse in ry_sequence:
                        assert isinstance(pulse, Pulse)
                        sequence.append((qd_channel, add_drag(pulse, beta=beta)))
                    for channel, pulse in rx90_sequence:
                        assert isinstance(pulse, Pulse)
                        sequence.append((qd_channel, add_drag(pulse, beta=beta)))
                    sequence.append(
                        (
                            ro_channel,
                            Delay(
                                duration=rx90_sequence.duration + ry_sequence.duration
                            ),
                        )
                    )
                else:
                    rx_sequence = _rotation(natives, np.pi, 0.0, params.rx90)
                    ry90_sequence = _rotation(
                        natives, np.pi / 2, np.pi / 2, params.rx90
                    )
                    for channel, pulse in rx_sequence:
                        assert isinstance(pulse, Pulse)
                        sequence.append((qd_channel, add_drag(pulse, beta=beta)))
                    for channel, pulse in ry90_sequence:
                        assert isinstance(pulse, Pulse)
                        sequence.append((qd_channel, add_drag(pulse, beta=beta)))
                    sequence.append(
                        (
                            ro_channel,
                            Delay(
                                duration=rx_sequence.duration + ry90_sequence.duration
                            ),
                        )
                    )
                sequence.append((ro_channel, ro_pulse))

            sequences.append(sequence)
            ro_pulses = {}
            for qubit in targets:
                acq_channel = platform.qubits[qubit].acquisition
                assert acq_channel is not None
                ro_pulse = list(sequence.channel(acq_channel))[-1]
                assert isinstance(ro_pulse, Readout)
                ro_pulses[qubit] = ro_pulse
            all_ro_pulses.append(ro_pulses)

    results = platform.execute(
        sequences,
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    for beta, ro_pulses in zip(np.repeat(beta_param_range, 2), all_ro_pulses):
        for qubit in targets:
            prob = results[ro_pulses[qubit].id]
            data.register_qubit(
                DragTuningType,
                (qubit),
                {
                    "prob": np.array([prob]),
                    "error": np.array([np.sqrt(prob * (1 - prob) / params.nshots)]),
                    "beta": np.array([beta]),
                },
            )

    return data


def _fit(data: DragTuningSimpleData) -> DragTuningSimpleResults:
    """Post-processing for DRAG protocol.

    A linear fit is applied for the probability of both sequences.
    The optimal is determined as the point in which the two lines met.
    """
    qubits = data.qubits
    fitted_parameters = {}
    betas_optimal = {}
    for qubit in qubits:
        for setup in SEQUENCES:
            qubit_data = data[qubit, setup]
            popt, _ = curve_fit(
                f=lambda x, a, b: a * x + b,
                xdata=qubit_data["beta"],
                ydata=qubit_data["prob"],
                p0=[
                    (qubit_data["prob"][-1] - qubit_data["prob"][0])
                    / (qubit_data["beta"][-1] - qubit_data["beta"][0]),
                    np.mean(qubit_data["prob"]),
                ],
                sigma=qubit_data["error"],
            )
            fitted_parameters[qubit, setup] = popt.tolist()
        try:
            betas_optimal[qubit] = -(
                fitted_parameters[qubit, "YpX9"][1]
                - fitted_parameters[qubit, "XpY9"][1]
            ) / (
                fitted_parameters[qubit, "YpX9"][0]
                - fitted_parameters[qubit, "XpY9"][0]
            )
        except ZeroDivisionError:
            pass
    return DragTuningSimpleResults(betas_optimal, fitted_parameters, data.rx90)


def _plot(
    data: DragTuningSimpleData, target: QubitId, fit: DragTuningSimpleResults
) -> tuple[list[go.Figure], str]:
    """Plotting function for DragTuning."""

    figures = []
    fitting_report = ""

    setup_colors = {SEQUENCES[0]: "red", SEQUENCES[1]: "blue"}

    fig = go.Figure()
    for setup in SEQUENCES:
        qubit_data = data[target, setup]
        fig.add_trace(
            go.Scatter(
                x=qubit_data["beta"],
                y=qubit_data["prob"],
                error_y={
                    "type": "data",
                    "array": qubit_data["error"],
                    "visible": True,
                    "color": setup_colors[setup],
                },
                mode="markers",
                name=f"{setup} data",
                showlegend=True,
                legendgroup=f"{setup} data",
                marker={"color": setup_colors[setup]},
            )
        )

    # add fitting traces
    if fit is not None:
        for setup in SEQUENCES:
            qubit_data = data[target, setup]
            betas = qubit_data["beta"]
            beta_range = np.linspace(
                min(betas),
                max(betas),
                2,
            )
            fig.add_trace(
                go.Scatter(
                    x=beta_range,
                    y=fit.fitted_parameters[target, setup][0] * beta_range
                    + fit.fitted_parameters[target, setup][1],
                    name=f"{setup} fit",
                    mode="lines",
                    line={"color": setup_colors[setup]},
                    showlegend=True,
                    legendgroup=f"{setup} fit",
                ),
            )
        fitting_report = table_html(
            table_dict(
                target,
                ["Beta"],
                [np.round(fit.betas[target], 4)],
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Beta parameter",
        yaxis_title="Excited state probability",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(
    results: DragTuningSimpleResults, platform: CalibrationPlatform, target: QubitId
):
    update.drag_pulse_beta(
        results.betas[target],
        platform,
        target,
        rx90=results.rx90,
    )


drag_simple = Protocol(_acquisition, _fit, _plot, _update)
"""DragTuning Protocol object."""
