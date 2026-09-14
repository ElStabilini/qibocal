from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import TypeAdapter
from qibolab import Custom, Envelope, Pulse, Readout

from qibocal.auto.operation import (
    Data,
    Parameters,
    Protocol,
    QubitId,
    QubitPairId,
    Results,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log

from .utils import table_dict, table_html

__all__ = ["set_envelope"]

SINGLE_QUBIT_GATES = ("RX", "RX90", "RX12", "MZ")
"""Native gates looked up under ``native_gates.single_qubit``."""
COUPLER_GATES = ("CP",)
"""Native gates looked up under ``native_gates.coupler``."""
TWO_QUBIT_GATES = ("CZ", "CNOT", "iSWAP")
"""Native gates looked up under ``native_gates.two_qubit``."""
SUPPORTED_GATES = SINGLE_QUBIT_GATES + COUPLER_GATES + TWO_QUBIT_GATES
"""Native gates whose envelope this protocol can replace."""

DEFAULT_AMPLITUDE = 0.0
"""Amplitude assigned by qibolab's ``initialize_parameters`` to a fresh pulse."""
DEFAULT_DURATION = 0.0
"""Duration assigned by qibolab's ``initialize_parameters`` to a fresh pulse."""

SAMPLES = 100
"""Number of samples used to draw an envelope in the report."""

Target = QubitId | QubitPairId
"""Either a qubit (or coupler) name, or a qubit pair."""


def _key(target: Target) -> Target:
    """Normalize a runcard target into a hashable dictionary key."""
    return tuple(target) if isinstance(target, (list, tuple)) else target


def _envelope_path(target: Target, gate: str) -> str:
    """Dotted path of the pulse whose envelope has to be replaced.

    The pair key is formatted explicitly as ``f"{pair[0]}-{pair[1]}"``, which is
    the format qibolab actually uses when dumping ``native_gates.two_qubit``.
    Interpolating the raw tuple instead produces a key that does not exist, and
    the resulting :class:`KeyError` is silently swallowed by
    :meth:`qibocal.auto.task.Completed.update_platform`.
    """
    if gate in TWO_QUBIT_GATES:
        pair = _key(target)
        prefix = f"native_gates.two_qubit.{pair[0]}-{pair[1]}.{gate}.0.1"
    elif gate in COUPLER_GATES:
        prefix = f"native_gates.coupler.{target}.{gate}.0.1"
    else:
        prefix = f"native_gates.single_qubit.{target}.{gate}.0.1"
    # MZ natives are `Readout` objects: the envelope lives on their probe pulse.
    return f"{prefix}.probe" if gate == "MZ" else prefix


def _stored_target(platform: CalibrationPlatform, target: Target, gate: str) -> Target:
    """Resolve a target to the key actually stored in the platform.

    Two-qubit natives may be declared for the reversed pair only (which
    ``TwoQubitContainer`` transparently accepts when the gate is symmetric), but
    the dotted path has to name the key that is really present.
    """
    if gate not in TWO_QUBIT_GATES:
        return target
    pair = _key(target)
    if pair in platform.natives.two_qubit:
        return pair
    reverse = (pair[1], pair[0])
    if reverse in platform.natives.two_qubit:
        return reverse
    raise KeyError(f"No two-qubit native gates defined for pair {pair}.")


def _pulse(platform: CalibrationPlatform, target: Target, gate: str) -> Pulse:
    """Pulse currently implementing ``gate`` on ``target``."""
    natives = platform.natives
    if gate in TWO_QUBIT_GATES:
        container = natives.two_qubit[_stored_target(platform, target, gate)]
    elif gate in COUPLER_GATES:
        container = natives.coupler[target]
    else:
        container = natives.single_qubit[target]

    pulse = container.ensure(gate)[0][1]
    if isinstance(pulse, Readout):
        pulse = pulse.probe
    if not isinstance(pulse, Pulse):
        raise TypeError(
            f"First element of native gate {gate} on {target} is a "
            f"{type(pulse).__name__}, which carries no envelope."
        )
    return pulse


def _current_envelope(platform: CalibrationPlatform, target: Target, gate: str) -> dict:
    """Envelope currently assigned to ``gate``, as a JSON-able dict.

    Envelopes are never stored as pydantic objects on :class:`Data` or
    :class:`Results`, since those are dumped through ``dataclasses.asdict`` plus
    ``json.dumps``, which cannot flatten a nested ``BaseModel``.
    """
    return _pulse(platform, target, gate).envelope.model_dump(mode="json")


def _apply(
    platform: CalibrationPlatform, target: Target, gate: str, envelope: dict
) -> None:
    """Replace the envelope of ``gate`` and reset amplitude and duration.

    The three values are set in a single :meth:`qibolab.Platform.update` call,
    so that the whole swap goes through one validation of the parameters tree.
    Resetting amplitude and duration is what makes the gate inert, i.e. "as if
    never calibrated", rather than merely reshaped.
    """
    prefix = _envelope_path(_stored_target(platform, target, gate), gate)
    platform.update(
        {
            f"{prefix}.envelope": dict(envelope),
            f"{prefix}.amplitude": DEFAULT_AMPLITUDE,
            f"{prefix}.duration": DEFAULT_DURATION,
        }
    )


@dataclass
class SetEnvelopeParameters(Parameters):
    """SetEnvelope runcard inputs."""

    target_gate: str
    """Native gate whose envelope is replaced (e.g. ``RX``, ``MZ``, ``CZ``)."""
    envelope: dict
    """New envelope, as a raw runcard mapping (e.g. ``{kind: gaussian, rel_sigma: 0.2}``)."""


@dataclass
class SetEnvelopeData(Data):
    """SetEnvelope acquisition outputs."""

    target_gate: str
    """Native gate whose envelope was replaced."""
    new_envelope: dict
    """Envelope assigned to the gate."""
    old_envelopes: dict[Target, dict] = field(default_factory=dict)
    """Envelope each target carried before the swap, kept for the report."""


@dataclass
class SetEnvelopeResults(Results):
    """SetEnvelope outputs.

    No fitting is performed: this only carries the acquisition payload across
    the ``qq acquire``/``qq fit`` boundary, so that :func:`_update` can re-apply
    the swap to whichever platform is live at fit time.
    """

    target_gate: str
    """Native gate whose envelope was replaced."""
    new_envelope: dict
    """Envelope assigned to the gate."""
    old_envelopes: dict[Target, dict] = field(default_factory=dict)
    """Envelope each target carried before the swap."""

    def __contains__(self, key: Target) -> bool:
        return _key(key) in self.old_envelopes


def _acquisition(
    params: SetEnvelopeParameters,
    platform: CalibrationPlatform,
    targets: list[Target],
) -> SetEnvelopeData:
    """Replace the envelope of ``params.target_gate`` on every target.

    No pulse is played. The platform is mutated here as well as in
    :func:`_update`, so that later actions running in the same process already
    see the new envelope.
    """
    old_envelopes = {}
    for target in targets:
        old_envelopes[_key(target)] = _current_envelope(
            platform, target, params.target_gate
        )
        _apply(platform, target, params.target_gate, params.envelope)
        log.info(
            f"Replaced {params.target_gate} envelope of {target} with "
            f"{params.envelope}, resetting amplitude and duration."
        )

    return SetEnvelopeData(
        target_gate=params.target_gate,
        new_envelope=dict(params.envelope),
        old_envelopes=old_envelopes,
    )


def _fit(data: SetEnvelopeData) -> SetEnvelopeResults:
    """Passthrough: there is nothing to fit."""
    return SetEnvelopeResults(
        target_gate=data.target_gate,
        new_envelope=data.new_envelope,
        old_envelopes=data.old_envelopes,
    )


def _waveforms(envelope: dict) -> tuple[np.ndarray, np.ndarray]:
    """Sample the i and q components of an envelope."""
    model = TypeAdapter(Envelope).validate_python(envelope)
    samples = len(model.i_) if isinstance(model, Custom) else SAMPLES
    return model.i(samples), model.q(samples)


def _description(envelope: dict) -> str:
    """Human readable one-liner for an envelope."""
    parameters = {key: value for key, value in envelope.items() if key != "kind"}
    if not parameters:
        return envelope["kind"]
    joined = ", ".join(f"{key}={value}" for key, value in parameters.items())
    return f"{envelope['kind']} ({joined})"


def _plot(data: SetEnvelopeData, target: Target, fit: SetEnvelopeResults):
    """Compare the old and the new envelope of the swapped gate."""
    old = data.old_envelopes.get(_key(target))
    if old is None:
        return [], ""

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("In-phase", "Quadrature"),
    )
    for name, envelope, color in [
        ("Old", old, "#6597aa"),
        ("New", data.new_envelope, "#aa6464"),
    ]:
        try:
            i, q = _waveforms(envelope)
        except (AssertionError, ValueError) as error:
            # e.g. `Snz` constrains the number of samples, `Custom` its length
            log.warning(f"Cannot sample {name.lower()} envelope {envelope}: {error}.")
            continue
        for column, component in [(1, i), (2, q)]:
            figure.add_trace(
                go.Scatter(
                    x=np.arange(len(component)),
                    y=component,
                    name=name,
                    legendgroup=name,
                    showlegend=column == 1,
                    line={"color": color},
                ),
                row=1,
                col=column,
            )

    figure.update_layout(
        title=f"{data.target_gate} envelope on {target}",
        xaxis_title="Sample",
        xaxis2_title="Sample",
        yaxis_title="Amplitude [a.u.]",
    )

    report = table_html(
        table_dict(
            str(target),
            ["Gate", "Old envelope", "New envelope", "Amplitude", "Duration"],
            [
                data.target_gate,
                _description(old),
                _description(data.new_envelope),
                DEFAULT_AMPLITUDE,
                DEFAULT_DURATION,
            ],
        )
    )
    return [figure], report


def _update(
    results: SetEnvelopeResults, platform: CalibrationPlatform, target: Target
) -> None:
    """Re-apply the swap on the platform live at fit time."""
    # Raises `KeyError` for a target that was not acquired, which
    # `Completed.update_platform` turns into a per-target skip.
    results.old_envelopes[_key(target)]
    _apply(platform, target, results.target_gate, results.new_envelope)


set_envelope = Protocol(_acquisition, _fit, _plot, _update)
"""Set envelope Protocol object."""
