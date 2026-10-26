"""ORBIT protocol, wrapping :mod:`qibocal.orbit` to run it through ``qq``."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal import orbit as core
from qibocal.auto.operation import Data, Parameters, Protocol, QubitId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log

from .randomized_benchmarking.utils import number_to_str
from .utils import table_dict, table_html

__all__ = ["orbit"]


@dataclass
class OrbitParameters(Parameters):
    """ORBIT runcard inputs."""

    knobs: dict[str, float]
    """Map from platform update path to initial step of the optimization.

    Paths may contain the ``{qubit}``, ``{drive}`` and ``{native}`` placeholders,
    cf. :func:`qibocal.orbit.resolve`.
    """
    depth: int | None = None
    """Number of Cliffords per sequence.

    If ``None``, it is derived from the RB fidelity stored in the calibration.
    """
    niter: int = 20
    """Number of random sequences per evaluation."""
    maxfev: int = 60
    """Maximum number of evaluations."""
    xatol: float = 0.05
    """Convergence tolerance on the knobs, in units of their steps."""
    max_deviation: float = 5.0
    """Maximum deviation of the knobs from their initial values, in units of their steps."""
    confirm_niter: int = 40
    """Number of fresh random sequences to compare initial and best values."""
    min_improvement_sigma: float = 2.0
    """Improvement required to update the platform, in units of its uncertainty."""
    seed: int | None = None
    """Seed for the generation of the random sequences."""


OrbitType = np.dtype([("fidelity", np.float64), ("error", np.float64)])
"""Sequence fidelity of each evaluation."""


@dataclass
class OrbitData(Data):
    """ORBIT acquisition outputs."""

    min_improvement_sigma: float
    """Improvement required to update the platform."""
    depth: dict[QubitId, int] = field(default_factory=dict)
    """Depth of the sequences."""
    paths: dict[QubitId, list[str]] = field(default_factory=dict)
    """Resolved update paths of the knobs."""
    steps: dict[QubitId, list[float]] = field(default_factory=dict)
    """Initial steps of the knobs."""
    initial: dict[QubitId, list[float]] = field(default_factory=dict)
    """Initial values of the knobs."""
    best: dict[QubitId, list[float]] = field(default_factory=dict)
    """Best values of the knobs."""
    values: dict[QubitId, list[list[float]]] = field(default_factory=dict)
    """Values of the knobs at each evaluation."""
    confirmation: dict[QubitId, list[float]] = field(default_factory=dict)
    """Initial and best fidelities, with errors, on fresh sequences."""
    data: dict[QubitId, npt.NDArray[OrbitType]] = field(default_factory=dict)
    """Sequence fidelity of each evaluation."""


@dataclass
class OrbitResults(Results):
    """ORBIT outputs."""

    best: dict[QubitId, dict[str, float]]
    """Best values of the knobs."""
    fidelity_initial: dict[QubitId, tuple[float, float]]
    """Confirmed sequence fidelity with the initial values."""
    fidelity_best: dict[QubitId, tuple[float, float]]
    """Confirmed sequence fidelity with the best values."""
    improved: dict[QubitId, bool]
    """Whether the best values are a significant improvement."""


def _depth(params: OrbitParameters, platform: CalibrationPlatform, target: QubitId):
    """Depth from parameters, or from the calibrated RB fidelity."""
    if params.depth is not None:
        return params.depth
    calibration = platform.calibration.single_qubits.get(target)
    if calibration is None or calibration.rb_fidelity is None:
        raise ValueError(
            f"No RB fidelity calibrated for qubit {target}: "
            "run standard_rb first, or set the ORBIT depth explicitly."
        )
    return core.optimal_depth(calibration.rb_fidelity[0])


def _acquisition(
    params: OrbitParameters, platform: CalibrationPlatform, targets: list[QubitId]
) -> OrbitData:
    """Optimize the knobs of each target qubit in turn."""
    data = OrbitData(min_improvement_sigma=params.min_improvement_sigma)
    rng = np.random.default_rng(params.seed)
    for target in targets:
        depth = _depth(params, platform, target)
        res = core.optimize(
            platform,
            target,
            params.knobs,
            core.rb_objective(
                platform, target, depth, params.niter, params.nshots, rng
            ),
            maxfev=params.maxfev,
            xatol=params.xatol,
            max_deviation=params.max_deviation,
            confirm=core.rb_objective(
                platform, target, depth, params.confirm_niter, params.nshots, rng
            ),
        )
        data.depth[target] = depth
        for key in ("paths", "steps", "initial", "best", "values", "confirmation"):
            getattr(data, key)[target] = res[key]
        data.register_qubit(
            OrbitType, target, {"fidelity": res["fidelity"], "error": res["error"]}
        )
    return data


def _fit(data: OrbitData) -> OrbitResults:
    """Check whether the best values significantly improve the sequence fidelity."""
    best, fidelity_initial, fidelity_best, improved = {}, {}, {}, {}
    for target in data.qubits:
        f0, err0, f1, err1 = data.confirmation[target]
        best[target] = dict(zip(data.paths[target], data.best[target]))
        fidelity_initial[target] = (f0, err0)
        fidelity_best[target] = (f1, err1)
        improved[target] = bool(
            f1 - f0 > data.min_improvement_sigma * np.hypot(err0, err1)
        )
    return OrbitResults(best, fidelity_initial, fidelity_best, improved)


def _plot(data: OrbitData, target: QubitId, fit: OrbitResults | None = None):
    """Sequence fidelity and knobs evolution along the optimization."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Sequence fidelity", "Knobs deviation"),
    )
    trace = data[target]
    evaluations = np.arange(len(trace))
    fig.add_trace(
        go.Scatter(
            x=evaluations,
            y=trace.fidelity,
            error_y={"type": "data", "array": trace.error},
            mode="lines+markers",
            name="sequence fidelity",
        ),
        row=1,
        col=1,
    )
    best = int(np.argmax(trace.fidelity))
    fig.add_trace(
        go.Scatter(
            x=[best],
            y=[trace.fidelity[best]],
            mode="markers",
            marker={"size": 14, "symbol": "circle-open", "color": "red"},
            name="best",
        ),
        row=1,
        col=1,
    )

    deviations = (
        np.array(data.values[target]) - np.array(data.initial[target])
    ) / np.array(data.steps[target])
    for path, deviation in zip(data.paths[target], deviations.T):
        fig.add_trace(
            go.Scatter(x=evaluations, y=deviation, mode="lines", name=path),
            row=2,
            col=1,
        )
    fig.update_layout(showlegend=True, xaxis2_title="Nelder-Mead evaluations")
    fig.update_yaxes(title_text="Survival probability", row=1, col=1)
    fig.update_yaxes(title_text="Deviation [steps]", row=2, col=1)

    fitting_report = ""
    if fit is not None:
        names = ["depth", "evaluations"] + data.paths[target]
        values = [data.depth[target], len(trace)] + [
            f"{v0:.6g} &rarr; {v1:.6g}"
            for v0, v1 in zip(data.initial[target], data.best[target])
        ]
        names += ["fidelity initial", "fidelity best", "updated"]
        values += [
            number_to_str(*fit.fidelity_initial[target]),
            number_to_str(*fit.fidelity_best[target]),
            fit.improved[target],
        ]
        fitting_report = table_html(table_dict(target, names, values))

    return [fig], fitting_report


def _update(results: OrbitResults, platform: CalibrationPlatform, target: QubitId):
    """Apply the best values, only if they are a significant improvement."""
    if not results.improved[target]:
        log.info(f"ORBIT did not improve qubit {target}: platform not updated.")
        return
    platform.update(results.best[target])


orbit = Protocol(_acquisition, _fit, _plot, _update)
