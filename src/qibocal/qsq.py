from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, PulseSequence

from qibocal.auto.operation import Data, Parameters, Protocol, QubitId, Results
from qibocal.calibration import CalibrationPlatform

__all__ = ["qsq_1q"]


# Protocol for verification
@dataclass
class QSQParameters(Parameters):
    """QSQ (RX^n) runcard inputs."""

    # NOTE: missing guardrail for maximum number of pulses vs relaxation time and nshots
    counts_max: int
    """Maximum number of RX(pi/2) pulses n."""
    counts_step: int
    """RX(pi/2) pulse count step."""


@dataclass
class QSQResult(Results):
    """QSQ outputs."""


QSQType = np.dtype(
    [("rx_counts", np.float64), ("prob", np.float64), ("error", np.float64)]
)


@dataclass
class QSQData(Data):
    """QSQ acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    data: dict[QubitId, npt.NDArray[QSQType]] = field(default_factory=dict)
    """Raw data acquired."""


def qsq_sequence(
    platform: CalibrationPlatform,
    qubit: QubitId,
    iteration: int,
):
    """Pulse sequence for QSQ experiment, based on the n-qubit S-gate model

        S_n = (|+>^{otimes n}, {S^(k)}_{k=1}^{n}, {|J><J|}_{J in {+,-}^n})

    Args:
        platform: CalibrationPlatform
        qubit: QubitId
        iteration: Number of S gates applied.
    """

    natives = platform.natives.single_qubit[qubit]

    sequence = PulseSequence()

    prep_channel, prep_pulse = natives.R(theta=np.pi / 2, phi=np.pi / 2)[0]
    sequence.extend([(prep_channel, prep_pulse)])

    if iteration > 0:
        s_channel, s_pulse = natives.RZ(theta=np.pi / 2)[0]
        for _ in range(iteration):
            sequence.extend([(s_channel, s_pulse)])

    readout_channel, readout_pulse = natives.R(theta=-np.pi / 2, phi=np.pi / 2)[0]
    sequence.extend([(readout_channel, readout_pulse)])

    sequence |= natives.MZ()

    return sequence


def _acquisition(
    params: QSQParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> QSQData:
    r"""
    Data acquisition for the QSQ (S-gate model) experiment.

    Args:
        params: QSQParameters
        platform: CalibrationPlatform
        targets: list of QubitId
    """

    data = QSQData(resonator_type=platform.resonator_type)

    sequences: list[PulseSequence] = []
    iteration_sweep = range(0, params.counts_max, params.counts_step)
    for iteration in iteration_sweep:
        sequence = PulseSequence()
        for qubit in targets:
            sequence += qsq_sequence(
                platform=platform,
                qubit=qubit,
                iteration=iteration,
            )
        sequences.append(sequence)

    results = platform.execute(
        sequences,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
    )

    for iteration, sequence in zip(iteration_sweep, sequences):
        for qubit in targets:
            acq_channel = platform.qubits[qubit].acquisition
            ro_pulse = list(sequence.channel(acq_channel))[-1]
            prob = results[ro_pulse.id]
            error = np.sqrt(prob * (1 - prob) / params.nshots)
            data.register_qubit(
                QSQType,
                qubit,
                {
                    "s_counts": np.array([iteration]),
                    "prob": np.array([prob]),
                    "error": np.array([error]),
                },
            )
    return data


def _fit(data: QSQData) -> QSQResult:
    """Fitting function for QSQ experiment."""


def _plot(data: QSQData, target: QubitId, fit: QSQResult | None = None):
    """Plotting function for QSQ experiment."""


def _update(results: QSQResult, platform: CalibrationPlatform, target: QubitId):
    """Update function for QSQ experiment."""


qsq_1q = Protocol(_acquisition, _fit, _plot)
"""Single qubit QSQ Routine object."""
