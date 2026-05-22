from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibo import gates
from qibolab import AcquisitionType, AveragingMode, PulseSequence, Readout
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibo.models import Circuit
from qibocal.calibration.calibration import QubitPairId
from qibocal.config import log
from qibocal.protocols.utils import COLORBAND, COLORBAND_LINE
from qibolab import Delay


# Protocol for verification
@dataclass
class QSQParameters(Parameters):
    """QSQ exeriment runcard inputs."""

    nrepetitions: int


@dataclass
class QSQResult(Results):
    """QSQ experiment outputs."""


QSQType = np.dtype([])


@dataclass
class QSQData(Data):
    """QSQ experiment acquisition outputs."""

    data: npt.NDArray[QSQType] = field(default_factory=dict)


def generate_sequence(
    platform: CalibrationPlatform, target: QubitId | QubitPairId, params: QSQParameters
) -> Circuit:
    """Generates the pulse sequence for the QSQ experiment.

    drive --- RY90 ---- {gate sequence} ----- RY90 -----------
    flux -----------------------------------------------------
    readout ------------------------------------------- MZ ---

    The set of possible gate sequences is {empty, SS, SSinv, SinvS, SinvSinv}.
    expected + state: {empty, SSinv, SinvS}
    expected - state: {SS, SinvSinv}
    """


def _acquisition(
    params: QSQParameters, platform: CalibrationPlatform, targets: list[QubitId]
) -> QSQData:
    """Acquisition function for QSQ experiment.

    The experiment is performed using a standard mass probability function with form
    """


def _fit(data: QSQData) -> QSQResult:
    """Fitting function for QSQ experiment."""


def _plot(data: QSQData, target: QubitId, fit: QSQResult | None = None):
    """Plotting function for QSQ experiment."""


def _update(results: QSQResult, platform: CalibrationPlatform, target: QubitId):
    """Update function for QSQ experiment."""


qsq_1q = Routine(_acquisition, _fit, _plot)
"""Single qubit QSQ Routine object."""
