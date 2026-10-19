"""Frequency sweeps wider than the IF bandwidth of an LO.

The sweep of each qubit is split in batches, each within the IF bandwidth of its LO,
which are executed separately and stitched back together in IQ.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from qibolab import (
    AcquisitionType,
    AveragingMode,
    ChannelId,
    Delay,
    Parameter,
    Pulse,
    PulseSequence,
    Rectangular,
    Sweeper,
)

from qibocal.auto.operation import QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log

__all__ = [
    "MAX_IF_BANDWIDTH",
    "Batch",
    "BatchedResults",
    "LoSweep",
    "acquire",
    "drive_sequence",
    "plan_batches",
    "plan_drive_sweeps",
    "stitch",
]

MAX_IF_BANDWIDTH = 300e6
"""Maximum intermediate frequency [Hz] reachable from an LO."""
BATCH_OVERLAP = 10
"""Number of points shared by adjacent batches, used to stitch them."""


@dataclass
class Batch:
    """Frequencies of a single qubit swept in a single execution."""

    frequencies: npt.NDArray[np.float64]
    """Absolute frequencies [Hz]."""
    lo_frequency: float | None
    """LO frequency [Hz] used to sweep :attr:`frequencies`."""
    anchor: bool
    """Whether the other batches are aligned to this one."""


def _split_outward(
    outward: npt.NDArray[np.int64], inner: npt.NDArray[np.int64], max_points: int
) -> list[npt.NDArray[np.int64]]:
    """Split the ``outward`` indices in equal batches, moving away from ``inner``.

    Both ``outward`` and ``inner`` are ordered in the outward direction.
    """
    if len(outward) == 0:
        return []
    overlap = min(BATCH_OVERLAP, max_points // 2)
    nbatches = int(np.ceil(len(outward) / (max_points - overlap)))
    batches = []
    for chunk in np.array_split(outward, nbatches):
        inner = np.concatenate([inner[max(len(inner) - overlap, 0) :], chunk])
        batches.append(inner)
    return batches


def plan_batches(
    frequencies: npt.NDArray[np.float64],
    lo_frequency: float | None,
    max_if_bandwidth: float = MAX_IF_BANDWIDTH,
) -> list[Batch]:
    """Split the sweep of a qubit in batches, sorted by frequency.

    The anchor batch contains the frequencies within ``max_if_bandwidth`` from the
    current LO, which is left untouched. The remaining frequencies are split in
    equal batches moving outward from the anchor, each with its LO at its centre.
    If no frequency is close enough to the current LO, the anchor is the batch
    nearest to it, with its LO moved as well.
    """
    if lo_frequency is None:
        return [Batch(frequencies, None, True)]

    indices = np.arange(len(frequencies))

    # largest number of points spanning at most twice the IF bandwidth
    max_points = int(
        np.searchsorted(
            frequencies, frequencies[0] + 2 * max_if_bandwidth, side="right"
        )
    )
    window = np.abs(frequencies - lo_frequency) <= max_if_bandwidth
    anchor_lo = lo_frequency
    if window.any():
        anchor = indices[window]
    else:
        anchor = (
            indices[:max_points]
            if frequencies[0] > lo_frequency
            else indices[-max_points:]
        )
        anchor_lo = float(frequencies[anchor[0]] + frequencies[anchor[-1]]) / 2

    lower = _split_outward(indices[: anchor[0]][::-1], anchor[::-1], max_points)
    upper = _split_outward(indices[anchor[-1] + 1 :], anchor, max_points)
    outer = [frequencies[np.sort(batch)] for batch in lower[::-1] + upper]
    batches = [Batch(freqs, float(freqs[0] + freqs[-1]) / 2, False) for freqs in outer]
    batches.insert(len(lower), Batch(frequencies[anchor], anchor_lo, True))
    return batches


@dataclass
class LoSweep:
    """Sweep of a channel, split in batches within the IF bandwidth of its LO."""

    channel: ChannelId
    """Swept channel."""
    lo: ChannelId | None
    """LO of :attr:`channel`."""
    lo_frequency: float | None
    """Calibrated frequency [Hz] of :attr:`lo`."""
    batches: list[Batch]
    """Batches, sorted by frequency."""

    @property
    def anchor(self) -> int:
        """Index of the batch the others are aligned to."""
        return next(k for k, batch in enumerate(self.batches) if batch.anchor)

    @property
    def lo_frequencies(self) -> list[float | None]:
        """LO frequency [Hz] used in each batch."""
        return [batch.lo_frequency for batch in self.batches]

    def updates(self, k: int) -> dict[ChannelId, dict[str, float]]:
        """Platform updates required to sweep batch ``k``.

        The LO is moved together with the channel, to pass qibolab's IF validation.
        """
        lo_frequency = self.batches[k].lo_frequency
        if lo_frequency == self.lo_frequency:
            return {}
        assert self.lo is not None and lo_frequency is not None
        return {
            self.lo: {"frequency": lo_frequency},
            self.channel: {"frequency": lo_frequency},
        }


def plan_drive_sweeps(
    platform: CalibrationPlatform,
    frequencies: dict[QubitId, npt.NDArray[np.float64]],
) -> dict[QubitId, LoSweep]:
    """Split the drive sweep of each qubit in batches, see :func:`plan_batches`."""
    sweeps: dict[QubitId, LoSweep] = {}
    for qubit, freqs in frequencies.items():
        drive_channel = platform.qubits[qubit].drive
        assert drive_channel is not None
        lo = platform.channels[drive_channel].lo
        lo_frequency = platform.config(lo).frequency if lo is not None else None
        sweep = sweeps[qubit] = LoSweep(
            channel=drive_channel,
            lo=lo,
            lo_frequency=lo_frequency,
            batches=plan_batches(freqs, lo_frequency),
        )
        if sweep.batches[sweep.anchor].lo_frequency != lo_frequency:
            log.warning(
                f"No frequency swept for qubit {qubit} lies within the IF bandwidth "
                f"of {lo}: no batch runs on a calibrated LO."
            )

    # moving an LO shared by multiple qubits would set it to multiple values at once
    shared_los = defaultdict(list)
    for qubit, sweep in sweeps.items():
        if sweep.lo is not None:
            shared_los[sweep.lo].append(qubit)
    for lo, qubits in shared_los.items():
        if len(qubits) > 1 and any(
            frequency != sweeps[q].lo_frequency
            for q in qubits
            for frequency in sweeps[q].lo_frequencies
        ):
            raise NotImplementedError(
                f"Qubits {qubits} share the drive LO {lo}, which would need to be moved "
                "to sweep the requested frequencies."
            )
    return sweeps


def drive_sequence(
    platform: CalibrationPlatform, qubit: QubitId, amplitude: float, duration: float
) -> PulseSequence:
    """Rectangular drive pulse, followed by the measurement of ``qubit``."""
    natives = platform.natives.single_qubit[qubit]
    drive_channel = platform.qubits[qubit].drive
    assert natives.MZ is not None and drive_channel is not None
    [(ro_channel, ro_pulse)] = natives.MZ()

    drive_pulse = Pulse(amplitude=amplitude, duration=duration, envelope=Rectangular())
    return PulseSequence(
        [
            (drive_channel, drive_pulse),
            (ro_channel, Delay(duration=drive_pulse.duration)),
            (ro_channel, ro_pulse),
        ]
    )


@dataclass
class BatchedResults:
    """Results of all the batches of a qubit, in execution order."""

    frequencies: npt.NDArray[np.float64]
    """Frequency [Hz] of every point, repeated where batches overlap."""
    batch: npt.NDArray[np.int64]
    """Batch index of every point."""
    iq: npt.NDArray[np.float64]
    """Acquired IQ, with the points on the second to last axis."""


def acquire(
    platform: CalibrationPlatform,
    sequences: dict[QubitId, PulseSequence],
    sweeps: dict[QubitId, LoSweep],
    *,
    nshots: int,
    relaxation_time: int,
    outer: dict[QubitId, Sweeper] | None = None,
    updates: dict[QubitId, dict] | None = None,
) -> dict[QubitId, BatchedResults]:
    """Execute the batches of all qubits, in parallel where possible.

    Each qubit plays its own entry of ``sequences``, which contains a single
    acquisition, while its frequencies are swept as planned in ``sweeps``.
    Optionally, a further ``outer`` sweeper and ``updates`` are applied per qubit.
    """
    # parallel sweepers are iterated together, thus they need the same length:
    # execution ``k`` is split by the length of the batches and outer sweepers
    executions = defaultdict(list)
    for qubit, sweep in sweeps.items():
        for k, batch in enumerate(sweep.batches):
            outer_length = len(outer[qubit].values) if outer is not None else 0
            executions[k, len(batch.frequencies), outer_length].append(qubit)

    values = {qubit: defaultdict(list) for qubit in sweeps}
    for (k, *_), qubits in sorted(executions.items()):
        sequence = PulseSequence()
        freq_sweepers = []
        execution_updates = [sweeps[qubit].updates(k) for qubit in qubits]
        for qubit in qubits:
            sequence.extend(sequences[qubit])
            freq_sweepers.append(
                Sweeper(
                    parameter=Parameter.frequency,
                    values=sweeps[qubit].batches[k].frequencies,
                    channels=[sweeps[qubit].channel],
                )
            )
        if updates is not None:
            execution_updates += [updates[qubit] for qubit in qubits]
        sweepers = (
            [freq_sweepers]
            if outer is None
            else [[outer[qubit] for qubit in qubits], freq_sweepers]
        )

        results = platform.execute(
            [sequence],
            sweepers,
            updates=execution_updates,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
            nshots=nshots,
            relaxation_time=relaxation_time,
        )

        for qubit in qubits:
            frequencies = sweeps[qubit].batches[k].frequencies
            [(_, acquisition)] = sequences[qubit].acquisitions
            values[qubit]["frequencies"].append(frequencies)
            values[qubit]["batch"].append(np.full(len(frequencies), k))
            values[qubit]["iq"].append(results[acquisition.id])

    return {
        qubit: BatchedResults(
            frequencies=np.concatenate(value["frequencies"]),
            batch=np.concatenate(value["batch"]),
            iq=np.concatenate(value["iq"], axis=-2),
        )
        for qubit, value in values.items()
    }


def _batch_step(
    frequencies: npt.NDArray[np.float64],
    signal: npt.NDArray[np.complex128],
    inner: npt.NDArray[np.bool_],
    outer: npt.NDArray[np.bool_],
) -> tuple[complex, float]:
    """Signal step of the ``outer`` batch with respect to the ``inner`` one.

    The step is the mean complex difference over the shared frequencies, on the
    last axis of ``signal``, and over any leading axis. It is returned together
    with its standard error.
    """
    shared = np.isin(
        frequencies, np.intersect1d(frequencies[inner], frequencies[outer])
    )
    difference = signal[..., outer & shared] - signal[..., inner & shared]
    return difference.mean(), difference.std(ddof=1) / np.sqrt(difference.size)


def stitch(
    frequencies: npt.NDArray[np.float64],
    iq: npt.NDArray[np.float64],
    batch: npt.NDArray[np.int64],
    anchor: int,
    step_threshold: float = 3,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[float], list[bool]]:
    """Align the batches to the ``anchor`` one, and merge them.

    Returns the merged frequencies and IQ, together with the step of each batch and
    whether it is larger than ``step_threshold`` times its standard error.
    """
    signal = iq[..., 0] + 1j * iq[..., 1]
    nbatches = int(batch.max()) + 1

    steps = np.zeros(nbatches, dtype=complex)
    errors = np.zeros(nbatches)
    offsets = np.zeros(nbatches, dtype=complex)

    # moving outward from the anchor, the inner neighbour is always aligned first
    for k in [*range(anchor + 1, nbatches), *range(anchor - 1, -1, -1)]:
        inner = k - 1 if k > anchor else k + 1
        steps[k], errors[k] = _batch_step(
            frequencies, signal, batch == inner, batch == k
        )
        offsets[k] = offsets[inner] + steps[k]

    # average the corrected points shared by adjacent batches
    merged, inverse, counts = np.unique(
        frequencies, return_inverse=True, return_counts=True
    )
    stitched = np.zeros((*signal.shape[:-1], len(merged)), dtype=complex)
    # transposed, to accumulate along the last axis
    np.add.at(stitched.T, inverse, (signal - offsets[batch]).T)
    stitched /= counts
    return (
        merged,
        np.stack([stitched.real, stitched.imag], axis=-1),
        np.abs(steps).tolist(),
        (np.abs(steps) > step_threshold * errors).tolist(),
    )
