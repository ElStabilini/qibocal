from dataclasses import dataclass, fields

import numpy as np
import plotly.graph_objects as go

from qibocal.auto.operation import Protocol, QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log, raise_error
from qibocal.protocols.randomized_benchmarking.standard_rb import (
    StandardRBParameters,
    survival_figure,
)

from ..utils import table_dict, table_html
from .utils import (
    INTERLEAVED_1Q_GATES,
    RBInterData,
    StandardRBResult,
    fit,
    rb_acquisition,
)

__all__ = ["standard_rb_inter"]

DIMENSION = 2
"""Hilbert space dimension of a single qubit."""


@dataclass
class StandardRBInterParameters(StandardRBParameters):
    """Parameters for the single qubit interleaved randomized benchmarking protocol."""

    interleave: str = "X"
    """Single qubit Clifford gate to interleave.

    One of ``I``, ``Z``, ``X``, ``Y``, ``SX``, ``SXdg``, ``SY``, ``SYdg``, ``H``.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.interleave not in INTERLEAVED_1Q_GATES:
            raise_error(
                ValueError,
                f"Gate {self.interleave} cannot be interleaved, "
                f"choose one of {list(INTERLEAVED_1Q_GATES)}.",
            )


@dataclass
class StandardRBInterResult(StandardRBResult):
    """Single qubit interleaved RB outputs."""

    fidelity_gate: dict[QubitId, list] = None
    """The fidelity of the interleaved gate and its uncertainty."""
    interleave: str | None = None
    """Name of the interleaved gate."""

    def __contains__(self, value: QubitId):
        return all(
            value in getattr(self, field.name)
            for field in fields(self)
            if isinstance(getattr(self, field.name), dict)
            and field.name != "fidelity_gate"
        )


def _acquisition(
    params: StandardRBInterParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RBInterData:
    """Data acquisition for single qubit Interleaved Randomized Benchmarking.

    The reference Clifford fidelity is read from the calibration, so
    ``standard_rb`` should be run (with update) before this protocol.
    """
    data = rb_acquisition(params, platform, targets, interleave=params.interleave)
    fidelity = {}
    for target in targets:
        reference = platform.calibration.single_qubits[target].rb_fidelity
        if reference is None:
            log.warning(
                "Qubit %s has no reference RB fidelity: run standard_rb before "
                "the interleaved version to extract the gate fidelity.",
                target,
            )
            continue
        fidelity[target] = reference
    data.fidelity = fidelity
    data.interleave = params.interleave
    return data


def _reference_decay(reference) -> tuple[float, float] | None:
    """Reference decay and its uncertainty from the stored fidelity, or ``None``
    when the reference is missing or the decay ratio would be undefined."""
    if reference is None:
        return None
    f_ref, sigma_f_ref = reference
    sigma_f_ref = sigma_f_ref if sigma_f_ref is not None else 0.0
    # F = 1/d + (d - 1)/d * p
    p_ref = (DIMENSION * f_ref - 1) / (DIMENSION - 1)
    sigma_p_ref = DIMENSION * sigma_f_ref / (DIMENSION - 1)
    if not np.isfinite(p_ref) or p_ref <= 0:
        return None
    return p_ref, sigma_p_ref


def _fit(data: RBInterData) -> StandardRBInterResult:
    """Fits the survival probability with y = Ap^x+B and extracts the fidelity of
    the interleaved gate.

    Args:
        data: Data from the data acquisition stage.

    Returns:
        StandardRBInterResult: Aggregated and processed data.
    """
    results = fit(data, single_qubit=True)
    fidelity_gate = {}
    for qubit in data.qubits:
        # Interleaved RB fidelity as defined in Eq. (4) of
        # https://arxiv.org/abs/1203.4550, with d = 2:
        #   F_gate = 1/d + (d - 1)/d * p_int / p_ref
        #
        # Only the reference fidelity is stored in the calibration, so p_ref and
        # sigma_p_ref are reconstructed by inverting F_ref = 1/d + (d - 1)/d * p_ref.
        # The reference and interleaved runs are assumed to be statistically
        # independent, so their errors are added in quadrature.
        reference = _reference_decay(data.fidelity.get(qubit))
        if reference is None:
            continue
        p_ref, sigma_p_ref = reference
        p_int = results.fit_parameters[qubit][1]
        sigma_p_int = results.fit_uncertainties[qubit][1]
        prefactor = (DIMENSION - 1) / DIMENSION
        fid_gate = 1 / DIMENSION + prefactor * p_int / p_ref
        uncertainty_gate = prefactor * np.sqrt(
            (sigma_p_int / p_ref) ** 2 + (p_int * sigma_p_ref / p_ref**2) ** 2
        )
        fidelity_gate[qubit] = [fid_gate, uncertainty_gate]

    return StandardRBInterResult(
        **vars(results), fidelity_gate=fidelity_gate, interleave=data.interleave
    )


def _plot(
    data: RBInterData, fit: StandardRBInterResult | None, target: QubitId
) -> tuple[list[go.Figure], str]:
    """Survival probability figure and interleaved RB table."""
    fig = survival_figure(data, fit, target, single_qubit=True)
    metadata = table_html(
        table_dict(
            str(target),
            ["niter", "nshots", "interleaved gate"],
            [data.niter, data.nshots, data.interleave],
        )
    )
    if fit is None:
        return [fig], metadata

    reference = _reference_decay(data.fidelity.get(target))
    gate = (fit.fidelity_gate or {}).get(target)

    names = [f"F_step (Clifford + {data.interleave})"]
    values = [(fit.fidelity[target], fit.fidelity_uncertainty[target])]
    if reference is not None:
        names.append("Reference decay p_ref")
        values.append(reference)
    if gate is not None:
        names.append(f"{data.interleave} fidelity (IRB estimate)")
        values.append((gate[0], gate[1]))

    fitting_report = table_html(
        table_dict(str(target), names, values, display_error=True)
    )

    return [fig], metadata + fitting_report


def _update(
    results: StandardRBInterResult, platform: CalibrationPlatform, target: QubitId
):
    """Write the interleaved gate fidelity in calibration."""
    if target not in (results.fidelity_gate or {}):
        return
    platform.calibration.single_qubits[target].gate_fidelity[results.interleave] = (
        tuple(results.fidelity_gate[target])
    )


standard_rb_inter = Protocol(_acquisition, _fit, _plot, _update)
