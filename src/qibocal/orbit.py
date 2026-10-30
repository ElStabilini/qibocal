"""Optimized randomized benchmarking for immediate tune-up (ORBIT).

Closed-loop optimization of arbitrary platform parameters, using the sequence
fidelity of randomized benchmarking at a fixed depth as objective, following
Kelly et al., Phys. Rev. Lett. 112, 240504 (2014), arXiv:1403.0035.

The functions in this module do not depend on any qibocal protocol type, so that
they can be used directly from Python. The :func:`qibocal.protocols.orbit` protocol
is a thin wrapper around them, which allows running ORBIT with ``qq``.
"""

from collections.abc import Callable

import numpy as np
from qibolab import AveragingMode
from scipy.optimize import minimize

from qibocal.auto.operation import QubitId
from qibocal.auto.transpile import (
    _execute_circuits,
    _transpile_circuits,
    build_native_gate_compiler,
    build_native_gate_transpiler,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.randomized_benchmarking.standard_rb import StandardRBParameters
from qibocal.protocols.randomized_benchmarking.utils import (
    RBGenerator,
    _generate_indexed_circuits,
)

__all__ = ["get_value", "optimal_depth", "optimize", "rb_objective", "resolve"]

Knobs = dict[str, float]
"""Map from platform update path (possibly templated) to initial simplex step.

The step is expressed in the same units of the parameter.
"""
Objective = Callable[[CalibrationPlatform], tuple[float, float]]
"""Function evaluating the sequence fidelity, and its error, on the current platform."""


def resolve(path: str, platform: CalibrationPlatform, target: QubitId) -> str:
    """Fill the placeholders of a templated update path."""

    natives = platform.natives.single_qubit[target]
    return path.format(
        qubit=target,
        drive=platform.qubits[target].drive,
        native="RX90" if natives.RX90 is not None else "RX",
    )


def get_value(platform: CalibrationPlatform, path: str) -> float:
    """Read the platform parameter addressed by an update path."""
    current = platform.parameters.model_dump()
    for step in path.split("."):
        try:
            current = current[int(step)]
        except ValueError:
            current = current[step]
    return float(current)


def optimal_depth(fidelity: float, min_depth: int = 5, max_depth: int = 1000) -> int:
    """Depth maximizing the sensitivity of the sequence fidelity to the gate error.

    It corresponds to the characteristic decay of the sequence fidelity
    :math:`m = -1/\\ln p`, with :math:`p = 2F - 1` for a single qubit Clifford fidelity
    :math:`F` (cf. Eq. S1 of arXiv:1403.0035).
    """
    p = 2 * fidelity - 1
    if not 0 < p < 1:
        return max_depth if p >= 1 else min_depth
    return int(np.clip(round(-1 / np.log(p)), min_depth, max_depth))


def rb_objective(
    platform: CalibrationPlatform,
    target: QubitId,
    depth: int,
    niter: int,
    nshots: int,
    seed: int | np.random.Generator | None = None,
) -> Objective:
    """Single qubit RB sequence fidelity at fixed ``depth``.

    The ``niter`` random sequences are generated and transpiled once, while they are
    compiled at every evaluation, in order to pick up the current platform
    parameters.
    """
    params = StandardRBParameters(depths=[depth], niter=niter, nshots=nshots)
    circuits = [
        indexed.circuit
        for indexed in _generate_indexed_circuits(params, RBGenerator(seed), [target])
    ]
    qubit_maps = [[target]] * len(circuits)
    transpiled = _transpile_circuits(
        circuits, qubit_maps, platform, build_native_gate_transpiler(platform)
    )

    def objective(platform: CalibrationPlatform) -> tuple[float, float]:
        counts = _execute_circuits(
            platform,
            build_native_gate_compiler(platform),
            transpiled,
            qubit_maps,
            nshots=nshots,
            averaging_mode=AveragingMode.CYCLIC,
        )
        survival = np.array([c["0"] / nshots for c in counts])
        return float(np.mean(survival)), float(np.std(survival) / np.sqrt(niter))

    return objective


def optimize(
    platform: CalibrationPlatform,
    target: QubitId,
    knobs: Knobs,
    objective: Objective,
    maxfev: int = 60,
    xatol: float = 0.05,
    max_deviation: float = 5.0,
    confirm: Objective | None = None,
) -> dict:
    """Maximize ``objective`` over ``knobs`` with the Nelder-Mead algorithm.

    If ``confirm`` is provided, it is evaluated on both the initial and best values,
    to compare them on an independent set of sequences.

    Returns:
        dict with the resolved ``paths``, their ``steps``, ``initial`` and ``best``
        values, the evaluated ``values`` with their ``fidelity`` and ``error``, and
        the ``confirmation`` as ``[f_initial, err_initial, f_best, err_best]``
        (``None`` if ``confirm`` is not provided).
    """
    paths = [resolve(path, platform, target) for path in knobs]
    steps = np.array(list(knobs.values()), dtype=float)
    base = platform.parameters
    initial = np.array([get_value(platform, path) for path in paths])
    trace = {"values": [], "fidelity": [], "error": []}

    def evaluate(x: np.ndarray, objective: Objective) -> tuple[float, float]:
        values = initial + x * steps
        platform.parameters = base
        platform.update(dict(zip(paths, values.tolist())))
        return objective(platform)

    def cost(x: np.ndarray) -> float:
        fidelity, error = evaluate(x, objective)
        trace["values"].append((initial + x * steps).tolist())
        trace["fidelity"].append(fidelity)
        trace["error"].append(error)
        return -fidelity

    n = len(paths)
    try:
        res = minimize(
            cost,
            x0=np.zeros(n),
            method="Nelder-Mead",
            bounds=[(-max_deviation, max_deviation)] * n,
            options={
                "maxfev": maxfev,
                "xatol": xatol,
                "initial_simplex": np.vstack([np.zeros(n), np.eye(n)]),
            },
        )
        confirmation = (
            None
            if confirm is None
            else [*evaluate(np.zeros(n), confirm), *evaluate(res.x, confirm)]
        )
    finally:
        platform.parameters = base

    return {
        "paths": paths,
        "steps": steps.tolist(),
        "initial": initial.tolist(),
        "best": (initial + res.x * steps).tolist(),
        "confirmation": confirmation,
        **trace,
    }
