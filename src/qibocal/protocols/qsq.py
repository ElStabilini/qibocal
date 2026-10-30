"""QSQ (quantum system quizzing) routine for qibocal: acquisition + plot, no fit.

Gate labels:
    sx -> RX(90)  implemented as a single native GPI2(phi=0) pulse
    tx -> RX(45)  (unrolled by the transpiler into native pulses)
    h  -> Hadamard (unrolled by the transpiler into native pulses)
Sequences are read left to right (first label = first gate applied).
Initial state |0>, measurement in Z.
"""

from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibo import Circuit, gates
from qibo.backends import NumpyBackend

from qibocal.auto.operation import Data, Parameters, Protocol, QubitId, Results
from qibocal.auto.transpile import (
    build_native_gate_compiler,
    build_native_gate_transpiler,
    execute_circuits,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import marginalize_qubit_counts, table_dict, table_html

DEFAULT_SEQUENCES = (
    ["e"]
    + [
        f"sx*{n}"
        for n in (
            2,
            4,
            6,
            8,
            10,
            12,
            20,
            22,
            50,
            52,
            100,
            102,
            150,
            152,
            200,
            202,
            400,
            402,
        )
    ]
    + ["sxhsx", "sx*3hsx", "hh", "hsh", "htxh", "sx*2htxh", "tx*2sx"]
)


def parse_sequence(seq: str) -> list[str]:
    """'sx*3hsx' -> ['sx','sx','sx','h','sx']; 'e' -> []; 'hsh' -> h sx h."""
    if seq == "e":
        return []
    out, i = [], 0
    while i < len(seq):
        if seq.startswith(("sx", "tx"), i):
            label, i = seq[i : i + 2], i + 2
        elif seq[i] == "h":
            label, i = "h", i + 1
        elif seq[i] == "s":  # 's' alone means sx (e.g. "hsh")
            label, i = "sx", i + 1
        else:
            raise ValueError(f"Cannot parse '{seq}' at position {i}")
        rep = 1
        if i < len(seq) and seq[i] == "*":
            j = i + 1
            while j < len(seq) and seq[j].isdigit():
                j += 1
            rep, i = int(seq[i + 1 : j]), j
        out += [label] * rep
    return out


def _add_gate(circuit: Circuit, label: str, q: int):
    if label == "sx":
        circuit.add(gates.GPI2(q, 0.0))  # one pi/2 pulse (RX(90) up to phase)
    elif label == "tx":
        circuit.add(gates.RX(q, np.pi / 4))
    elif label == "h":
        circuit.add(gates.H(q))
    else:
        raise ValueError(label)


def build_circuit(labels: list[str], nqubits: int) -> Circuit:
    """Same sequence applied in parallel on all qubits, then measured."""
    c = Circuit(nqubits)
    for label in labels:
        for q in range(nqubits):
            _add_gate(c, label, q)
    for q in range(nqubits):
        c.add(gates.M(q))
    return c


def ideal_outcome(labels: list[str]) -> float:
    """Ideal P(1) from a noiseless simulation (0 or 1 for valid quizzes)."""
    c = build_circuit(labels, 1)
    state = NumpyBackend().execute_circuit(c.copy(deep=True)).state()
    return float(np.round(abs(state[1]) ** 2, 6))


@dataclass
class QSQParameters(Parameters):
    """QSQ input parameters."""

    sequences: list[str] = field(default_factory=lambda: list(DEFAULT_SEQUENCES))
    """Gate sequences (strings, e.g. 'sx*4', 'sx*3hsx', 'e')."""


QSQType = np.dtype(
    [("index", int), ("prob1", float), ("expected", float), ("error", float)]
)


@dataclass
class QSQData(Data):
    sequences: list[str] = field(default_factory=list)
    nshots: int = 0
    data: dict[QubitId, np.ndarray] = field(default_factory=dict)


@dataclass
class QSQResults(Results):
    """No fit: empty results."""


def _acquisition(
    params: QSQParameters, platform: CalibrationPlatform, targets: list[QubitId]
) -> QSQData:
    transpiler = build_native_gate_transpiler(platform)
    compiler = build_native_gate_compiler(platform)

    labels = [parse_sequence(s) for s in params.sequences]
    expected = np.array([ideal_outcome(lab) for lab in labels])
    circuits = [build_circuit(lab, len(targets)) for lab in labels]

    results = execute_circuits(
        circuits,
        [targets] * len(circuits),
        platform,
        transpiler,
        compiler,
        nshots=params.nshots,
    )

    data = QSQData(sequences=list(params.sequences), nshots=params.nshots)
    for i, target in enumerate(targets):
        prob1 = np.array(
            [marginalize_qubit_counts(r, [i])["1"] / params.nshots for r in results]
        )
        error = np.where(expected >= 0.5, 1 - prob1, prob1)
        data.register_qubit(
            QSQType,
            target,
            {
                "index": np.arange(len(labels)),
                "prob1": prob1,
                "expected": expected,
                "error": error,
            },
        )
    return data


def _fit(data: QSQData) -> QSQResults:
    return QSQResults()


def _plot(data: QSQData, fit: QSQResults, target: QubitId):
    """Plotting function for QSQ experiment."""

    target_data = data[target]
    sequences = [data.sequences[i] for i in target_data.index]
    errors = np.sqrt(target_data.error * (1 - target_data.error) / data.nshots)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Excited state probability", "Error probability"),
    )
    fig.add_trace(
        go.Scatter(x=sequences, y=target_data.prob1, mode="markers", name="Measured"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=sequences, y=target_data.expected, mode="markers", name="Ideal"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sequences,
            y=target_data.error,
            error_y={"array": errors},
            mode="markers",
            name="P(wrong outcome)",
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Sequence", row=2, col=1)
    fig.update_yaxes(title_text="P(1)", row=1, col=1)
    fig.update_yaxes(title_text="P(wrong outcome)", row=2, col=1)

    fitting_report = table_html(
        table_dict(
            target,
            ["Mean error probability", "Shots"],
            [np.round(np.mean(target_data.error), 4), data.nshots],
        )
    )
    return [fig], fitting_report


qsq = Protocol(_acquisition, _fit, _plot)
