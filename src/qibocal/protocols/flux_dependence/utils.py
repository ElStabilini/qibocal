import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import ndimage
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.signal import medfilt

from ...auto.operation import Parameters
from ...config import log
from ..utils import (
    DISTANCE_XY,
    DISTANCE_Z,
    GHZ_TO_HZ,
    HZ_TO_GHZ,
    FeatExtractionError,
    Range,
    RangeLike,
    clustering,
    merging,
    minmax_scaling,
    peaks_finder,
    reshaping_raw_signal,
    to_range,
    zca_whiten,
)


@dataclass(kw_only=True)
class FluxFrequencySweepParameters(Parameters):
    """Parameters to define flux DC sweep."""

    freq_width: int | None = None
    """Width for frequency sweep relative to the readout frequency [Hz]."""
    freq_step: int | None = None
    """Frequency step for sweep [Hz]."""
    frequency: RangeLike | None = None
    """Frequency [Hz] range for sweep."""
    bias_width: float | None = None
    """Width for bias sweep [a.u.]."""
    bias_step: float | None = None
    """Bias step for sweep [a.u.]."""
    bias: RangeLike | None = None
    """Bias [a.u.] range for sweep."""

    def frequency_range(self, center: float = 0.0) -> Range:
        def legacy_range() -> Range:
            assert self.freq_width is not None and self.freq_step is not None
            return (
                center - self.freq_width / 2,
                center + self.freq_width / 2,
                self.freq_step,
            )

        return (
            to_range(self.frequency, center=center)
            if self.frequency is not None
            else legacy_range()
        )

    def bias_range(self, center: float = 0.0) -> Range:
        def legacy_range() -> Range:
            assert self.bias_width is not None and self.bias_step is not None
            return (
                center - self.bias_width / 2,
                center + self.bias_width / 2,
                self.bias_step,
            )

        return (
            to_range(self.bias, center=center)
            if self.bias is not None
            else legacy_range()
        )


def create_data_array(freq, bias, signal, dtype):
    """Create custom dtype array for acquired data."""
    size = len(freq) * len(bias)
    ar = np.empty(size, dtype=dtype)
    frequency, biases = np.meshgrid(freq, bias)
    ar["freq"] = frequency.ravel()
    ar["bias"] = biases.ravel()
    ar["signal"] = signal.ravel()
    return np.rec.array(ar)


def flux_dependence_plot(
    data,
    fit,
    qubit,
    inliers,
    outliers,
    fit_function,
):
    figures = []
    qubit_data = data[qubit]
    frequencies = qubit_data.freq * HZ_TO_GHZ

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=qubit_data.freq * HZ_TO_GHZ,
            y=qubit_data.bias,
            z=qubit_data.signal,
            colorbar={"title": "Signal [a.u.]"},
            colorscale="Viridis",
        ),
    )

    # TODO: This fit is for frequency, can it be reused here, do we even want the fit ?
    if (
        fit is not None
        and fit_function is not None
        and data.__class__.__name__ != "CouplerSpectroscopyData"
        and fit.successful_fit[qubit]
    ):
        params = fit.fitted_parameters[qubit]
        bias = np.unique(qubit_data.bias)
        fig.add_trace(
            go.Scatter(
                x=fit_function(bias, **params) * HZ_TO_GHZ,
                y=bias,
                showlegend=True,
                name="Fit",
                marker={"color": "rgb(248, 248, 248)"},
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=[
                    fit.frequency[qubit] * HZ_TO_GHZ,
                ],
                y=[
                    fit.sweetspot[qubit],
                ],
                mode="markers",
                marker={
                    "size": 8,
                    "color": "red",
                },
                name="Sweetspot",
                showlegend=True,
            ),
        )

        # Inliers and outliers plotting for debugging purposes
        if inliers is not None and len(inliers) > 0:
            fig.add_trace(
                go.Scatter(
                    x=inliers[:, 1] * HZ_TO_GHZ,  # frequency
                    y=inliers[:, 0],  # bias
                    mode="markers",
                    marker={
                        "size": 6,
                        "color": "white",
                    },
                    name="Inliers",
                    showlegend=True,
                    visible="legendonly",
                ),
            )

        if outliers is not None and len(outliers) > 0:
            fig.add_trace(
                go.Scatter(
                    x=outliers[:, 1] * HZ_TO_GHZ,  # frequency
                    y=outliers[:, 0],  # bias
                    mode="markers",
                    marker={
                        "size": 6,
                        "color": "green",
                    },
                    name="Outliers",
                    showlegend=True,
                    visible="legendonly",
                ),
            )

    fig.update_xaxes(
        title_text="Frequency [GHz]",
    )
    fig.update_yaxes(title_text="Bias [a.u.]")

    fig.update_layout(xaxis1={"range": [np.min(frequencies), np.max(frequencies)]})

    fig.update_layout(
        showlegend=True,
        legend={"orientation": "h"},
    )

    figures.append(fig)

    return figures


def flux_crosstalk_plot(data, qubit, fit, fit_function):
    figures = []
    fitting_report = ""
    all_qubit_data = {
        index: data_qubit
        for index, data_qubit in data.data.items()
        if index[0] == qubit
    }
    fig = make_subplots(
        rows=1,
        cols=len(all_qubit_data),
        horizontal_spacing=0.3 / len(all_qubit_data),
        vertical_spacing=0.1,
        subplot_titles=len(all_qubit_data) * ("Signal [a.u.]",),
    )
    for col, (flux_qubit, qubit_data) in enumerate(all_qubit_data.items()):
        frequencies = qubit_data.freq * HZ_TO_GHZ
        fig.add_trace(
            go.Heatmap(
                x=frequencies,
                y=qubit_data.bias,
                z=qubit_data.signal,
                showscale=False,
            ),
            row=1,
            col=col + 1,
        )
        if fit is not None and fit.successful_fit[qubit] and flux_qubit[1] != qubit:
            fig.add_trace(
                go.Scatter(
                    x=fit_function(
                        xj=qubit_data.bias, **fit.fitted_parameters[flux_qubit]
                    ),
                    y=qubit_data.bias,
                    showlegend=not any(
                        isinstance(trace, go.Scatter) for trace in fig.data
                    ),
                    legendgroup="Fit",
                    name="Fit",
                    marker={"color": "green"},
                ),
                row=1,
                col=col + 1,
            )

        fig.update_xaxes(
            title_text="Frequency [GHz]",
            row=1,
            col=col + 1,
        )

        fig.update_yaxes(
            title_text=f"Qubit {flux_qubit[1]}: Bias [a.u.]", row=1, col=col + 1
        )

    fig.update_layout(xaxis1={"range": [np.min(frequencies), np.max(frequencies)]})
    fig.update_layout(xaxis2={"range": [np.min(frequencies), np.max(frequencies)]})
    fig.update_layout(xaxis3={"range": [np.min(frequencies), np.max(frequencies)]})
    fig.update_layout(
        showlegend=True,
    )
    figures.append(fig)

    return figures, fitting_report


def G_f_d(xi, xj, offset, d, crosstalk_element, normalization):
    """Auxiliary function to calculate qubit frequency as a function of bias.

    It also determines the flux dependence of :math:`E_J`,:math:`E_J(\\phi)=E_J(0)G_f_d`.
    For more details see: https://arxiv.org/pdf/cond-mat/0703002.pdf

    Args:
        xi (float): bias of target qubit
        xj (float): bias of neighbor qubit
        offset (float): phase_offset [a.u.].
        d (float): asymmetry between the two junctions of the transmon.
                   Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
        crosstalk_element(float): off-diagonal crosstalk matrix element
        normalization(float): diagonal crosstalk matrix element
    Returns:
        (float)
    """
    return (
        d**2
        + (1 - d**2)
        * np.cos(
            np.pi
            * (xi * normalization + normalization * xj * crosstalk_element + offset)
        )
        ** 2
    ) ** 0.25


def transmon_frequency(
    xi, xj, w_max, d, normalization, offset, crosstalk_element, charging_energy
):
    r"""Approximation to transmon frequency.

    The formula holds in the transmon regime Ej / Ec >> 1.

    See  https://arxiv.org/pdf/cond-mat/0703002.pdf for the complete formula.

    Args:
        xi (float): bias of target qubit
        xj (float): bias of neighbor qubit
        w_max (float): maximum frequency  :math:`w_{max} = \sqrt{8 E_j E_c}
        d (float): asymmetry between the two junctions of the transmon.
                   Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
        normalization(float): diagonal crosstalk matrix element
        offset (float): phase_offset [a.u.].
        crosstalk_element(float): off-diagonal crosstalk matrix element
        charging_energy (float): Ec / h

     Returns:
         (float): qubit frequency as a function of bias.
    """
    return (w_max + charging_energy) * G_f_d(
        xi,
        xj,
        offset=offset,
        d=d,
        normalization=normalization,
        crosstalk_element=crosstalk_element,
    ) - charging_energy


def transmon_readout_frequency(
    xi,
    xj,
    w_max,
    d,
    normalization,
    crosstalk_element,
    offset,
    resonator_freq,
    g,
    charging_energy,
):
    r"""Approximation to flux dependent resonator frequency.

    The formula holds in the transmon regime Ej / Ec >> 1.

    See  https://arxiv.org/pdf/cond-mat/0703002.pdf for the complete formula.

    Args:
         xi (float): bias of target qubit
         xj (float): bias of neighbor qubit
         w_max (float): maximum frequency  :math:`w_{max} = \sqrt{8 E_j E_c}
         d (float): asymmetry between the two junctions of the transmon.
                    Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
         normalization(float): diagonal crosstalk matrix element
         offset (float): phase_offset [a.u.].
         crosstalk_element(float): off-diagonal crosstalk matrix element
         resonator_freq (float): bare resonator frequency
         g (float): readout coupling.
         charging_energy (float): Ec / h

     Returns:
         (float): resonator frequency as a function of bias.
    """

    qubit_frequency = transmon_frequency(
        xi=xi,
        xj=xj,
        w_max=w_max,
        d=d,
        normalization=normalization,
        offset=offset,
        crosstalk_element=crosstalk_element,
        charging_energy=charging_energy,
    )
    return resonator_freq + g**2 * (
        1 / (resonator_freq - qubit_frequency)
        - 1 / (resonator_freq - qubit_frequency + charging_energy)
    )


def filter_data(matrix_z: np.ndarray):
    """Filter data with a ZCA transformation and then a unit-variance Gaussian."""

    # adding zca filter for filtering out background noise gradient
    zca_z = zca_whiten(matrix_z)
    # adding gaussian fliter with unitary variance for blurring the signal and reducing noise
    return ndimage.gaussian_filter(zca_z, 1)


def flux_extract_feature(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    find_min: bool,
    min_points: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features of the signal by filtering out background noise.

    It first applies a custom filter mask (see `custom_filter_mask`)
    and then finds the biggest peak for each DC bias value;
    the masked signal is then clustered (see `clustering`) in order to classify the relevant signal for the experiment.
    If `find_min` is set to `True` it finds minimum peaks of the input signal;
    `min_points` is the minimum number of points for a cluster to be considered relevant signal.
    Position of the relevant signal is returned.
    """

    reshaped_x, reshaped_y, reshaped_z = reshaping_raw_signal(x, y, z)
    reshaped_z = -reshaped_z if find_min else reshaped_z

    z_masked = filter_data(reshaped_z)

    # renormalizing
    z_masked_norm = minmax_scaling(z_masked, axis=1)

    # filter data using find_peaks
    peaks_dict = peaks_finder(reshaped_x, reshaped_y, z_masked_norm)
    if len(peaks_dict.keys()) == 0:  # if find_peaks fails
        """
        Peaks Detection Failed:
        no peaks found in peaks_finder routine.
        """
        return None, None

    peaks, labels = clustering(peaks_dict, z_masked)

    # merging close clusters
    try:
        signal_clusters = merging(
            peaks,
            labels,
            min_points,
            distance_xy=DISTANCE_XY,
            distance_z=DISTANCE_Z,
        )

    except FeatExtractionError:
        return None, None

    medians = np.array(
        [[lab, np.median(cl["cluster"][2, :])] for lab, cl in signal_clusters.items()]
    )

    signal_labels = np.zeros(labels.size, dtype=bool)
    signal_idx = medians[np.argmax(medians[:, 1]), 0]
    signal_labels[signal_clusters[signal_idx]["cluster"][-1, :].astype(int)] = True

    return peaks_dict["x"]["val"][signal_labels], peaks_dict["y"]["val"][signal_labels]


def _function_dof(fit_function) -> int:
    sig = inspect.signature(fit_function)

    # Filter for positional parameters without defaults
    params = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect.Parameter.empty
    ]

    # Subtract 1 for the independent variable
    return len(params) - 1


def select_sweetspot(
    offset: float,
    normalization: float,
    bias_window: npt.ArrayLike,
    max_distance: float = 0,
):
    """Select the closest flux sweetspot that lies in the acquired bias window.

    The fitted model is periodic in ``offset + normalization * bias``. There is a
    sweetspot for every integer ``n`` at ``bias = (n - offset) / normalization``.

    If no sweetspot lies inside the acquired bias window, the closest sweetspot outside
    the window is selected unless it is farther than ``max_distance``.
    """
    low, high = np.sort(np.asarray(bias_window, dtype=float))

    reduced_bias_low = offset + normalization * low
    reduced_bias_high = offset + normalization * high

    n_low = int(np.ceil(reduced_bias_low))
    n_high = int(np.floor(reduced_bias_high))
    candidates = []
    for n in range(min(n_low, n_high), max(n_low, n_high) + 1):
        candidate = (n - offset) / normalization
        if low <= candidate <= high:
            candidates.append(candidate)

    if candidates:
        # If there is at least one sweetspot in the bias window, take the bias with abs
        # closest to 0.0
        return min(candidates, key=abs)

    # If candidates is empty then n_low == n_high + 1. These are the two integers
    # immediately outside the bias acquisition window.
    sweetspot_below = (n_low - offset) / normalization
    sweetspot_above = (n_high - offset) / normalization

    def distance_to_window(c):
        return max(low - c, 0, c - high)

    sweetspot = min(sweetspot_below, sweetspot_above, key=distance_to_window)

    if distance_to_window(sweetspot) > max_distance:
        raise ValueError("No fitted sweetspot lies inside the acquired bias window.")

    return sweetspot


def _continuity_score(
    xvals: npt.NDArray[np.floating],
    yvals: npt.NDArray[np.floating],
    inliers: npt.NDArray[np.bool_],
) -> int:
    """Score consecutive inlier runs quadratically, counting each y-value once."""
    order = np.argsort(xvals)  # in practice they are already ordered
    ordered_yvals = np.asarray(yvals)[order]
    ordered_inliers = np.asarray(inliers, dtype=bool)[order]

    # pad with 0s to ensure a change if the arc starts at the boundary
    padded = np.pad(ordered_inliers.astype(int), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)

    # If multiple points are at the same y-value, count them only once. This is to
    # deweight clusters/lines of peaks that are part of the background.
    unique_y_counts = [
        len(np.unique(ordered_yvals[start:stop])) for start, stop in zip(starts, stops)
    ]

    return int(np.sum(np.square(unique_y_counts)))


def ransac_fit(
    xvals: npt.NDArray[np.floating],
    yvals: npt.NDArray[np.floating],
    fit_function: Callable[..., np.ndarray],
    residual_threshold: float,
    min_trials: int = 100,
    max_trials: int = 5000,
    stop_probability: float = 0.999,
    random_state: int = 0,
    bounds: tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
    p0: npt.ArrayLike | None = None,
    min_samples: int | None = None,
    loss: str = "soft_l1",
    f_scale: float | None = None,
    return_covariance: bool = False,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
    """Fit a model to data using RANSAC, ignoring outliers.

    Repeatedly fits ``fit_function`` to minimal random subsets of the data (sized to the
    function's degrees of freedom). Points with residual below ``residual_threshold``
    are inliers. Candidate models are scored by summing the squared lengths of
    consecutive inlier runs along the x-axis, favoring a continuous feature over
    scattered noise.

    The number of trials adapts dynamically based on the current inlier
    ratio, following the standard RANSAC stopping criterion, and is bounded by
    ``min_trials`` and ``max_trials``. A final least-squares refit is performed on the
    best inlier set.

    Returns:
        Tuple of (fit parameters, boolean array of inliers).
    """
    # A fixed seed makes debugging and regression tests reproducible. RandomState's
    # output is guaranteed to be stable across numpy versions:
    # https://numpy.org/doc/2.5/reference/random/legacy.html#numpy.random.RandomState
    rng = np.random.RandomState(random_state)

    # The fit is much faster using Levenberg-Marquardt instead of Trust Region
    # Reflective, but without bounds non of the RANSAC iterations fall within the bounds
    # (even after correcting for symmetries). Possibly this is also because there
    # poorly/non constrained parameters.
    method = "lm" if bounds is None else "trf"
    fit_kwargs: dict[str, Any] = {} if bounds is None else {"bounds": bounds}

    function_dof = _function_dof(fit_function)
    if min_samples is None:
        if function_dof <= 0:
            raise ValueError(
                "Cannot introspect the degrees of freedom of the fit function "
                "(does it take *args?). Pass min_samples explicitly."
            )
        n_samples = function_dof
    else:
        n_samples = max(int(min_samples), function_dof)

    function_dof = _function_dof(fit_function)
    if len(xvals) < function_dof:
        raise ValueError(
            f"Not enough datapoints for RANSAC: got {len(xvals)}, but the fit function"
            f"has {function_dof} degrees of freedom, meaning at least that many"
            f"datapoints are needed to determine its parameters."
        )

    # Parameters of this model span several orders of magnitude even in GHz, so let the
    # optimizer rescale them from the Jacobian rather than treating them as comparable.
    trial_kwargs: dict[str, Any] = dict(fit_kwargs)
    if method != "lm":
        trial_kwargs["x_scale"] = "jac"

    # N_needed is the adaptively-updated trial budget; start at infinity so the loop is
    # initially bounded only by min_trials/max_trials.
    N_needed = np.inf
    ransac_iterations = 0
    best_inliers = np.zeros(len(xvals), dtype=bool)
    best_params = None
    best_score = (0, 0)
    # Track already-sampled subsets so we don't waste a trial refitting the exact same
    # minimal sample twice.
    tried_subsets = set()

    # Standard adaptive RANSAC loop: keep going until we've hit max_trials, or until the
    # estimated number of iterations needed to find an all-inlier sample (with
    # probability stop_probability) drops below where we already are, although we never
    # stop before attempting at least min_trials.
    # https://en.wikipedia.org/wiki/Random_sample_consensus#Parameters
    while (
        ransac_iterations < min(N_needed, max_trials) or ransac_iterations < min_trials
    ):
        ransac_iterations += 1

        # Small samples maximize the probability that every drawn point sits on the
        # feature of interest, but an exactly determined sample interpolates whatever it
        # is given and cannot discriminate between degenerate parameter combinations.
        # n_samples trades one against the other.
        subset = rng.choice(len(xvals), n_samples, replace=False)
        subset_ = tuple(sorted(subset))
        if subset_ in tried_subsets:
            continue
        tried_subsets.add(subset_)

        try:
            with warnings.catch_warnings():
                # Poor fits are expected for random subsets, so suppress these warnings.
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = curve_fit(
                    fit_function,
                    xvals[subset],
                    yvals[subset],
                    p0=p0,
                    method=method,
                    **trial_kwargs,
                )
        except RuntimeError:
            continue

        # compute the number of inliers based on all datapoints, not just those in the
        # subset.
        residuals_all = np.abs(yvals - fit_function(xvals, *popt))
        inliers = residuals_all < residual_threshold

        # if all points are inliers, we cannot improve and can break the ransac loop
        if inliers.sum() == len(xvals):
            best_inliers = inliers
            best_params = popt
            break

        # Use a continuity score to reduce the sensitivity to a noisy background.
        # Especially if the arc is only in a small part of the bias range.
        unique_y_count = len(np.unique(yvals[inliers]))
        score = (
            _continuity_score(xvals, yvals, inliers),
            unique_y_count,
        )

        # Prefer fits with long consecutive inlier runs (high continuity) over many
        # scattered inliers, making noisy fits less likely to be selected.
        if inliers.sum() >= n_samples and score > best_score:
            best_inliers = inliers
            best_params = popt
            best_score = score
            # Re-estimate how many trials are needed to have `stop_probability`
            # confidence of drawing an all-inlier sample.
            denom = np.log(1 - (best_inliers.sum() / len(xvals)) ** n_samples)
            N_needed = np.log(1 - stop_probability) / denom

    if best_params is None:
        raise RuntimeError(
            f"RANSAC failed to find a valid fit after {ransac_iterations} iterations: "
            f"no sample of size {n_samples} produced at least {n_samples} "
            f"inliers (residual_threshold={residual_threshold})."
        )

    # Finally optimize by doing a least-squares fit to the best set of inliers. A robust
    # loss is used here, and only here, to absorb the borderline points that sit just
    # inside the residual threshold.
    refit_kwargs = dict(trial_kwargs)
    if method != "lm" and loss is not None:
        refit_kwargs["loss"] = loss
        if f_scale is not None:
            refit_kwargs["f_scale"] = f_scale

    popt, pcov = curve_fit(
        fit_function,
        xvals[best_inliers],
        yvals[best_inliers],
        p0=best_params,
        method=method,
        maxfev=100000,
        **refit_kwargs,
    )

    # These are for visualization in the plot only
    inliers_mask = best_inliers

    if return_covariance:
        return popt, inliers_mask, pcov
    return popt, inliers_mask


ARC_PARAMETERS = (
    "g",
    "d",
    "offset",
    "normalization",
    "resonator_freq",
    "charging_energy",
)
"""Free parameters of the resonator arc model, in the order used by ``curve_fit``."""

DEFAULT_FIXED_PARAMETERS = ("charging_energy",)
"""Parameters held at their guess value during the fit.
"""

ARC_BOUNDS = {
    "g": (5e-3, 0.4),  # GHz
    "d": (0.0, 0.9),  # above ~0.9 the derivative of G_f_d collapses
    "offset": (-1.0, 1.0),  # a.u., the model is periodic with unit period
    "resonator_freq": (-0.1, 0.1),  # GHz, relative to the guess
    "charging_energy": (0.10, 0.35),  # GHz
}
# ``normalization`` is absent because its bounds are derived from the
# bias sweep itself (see :func:`normalization_grid`), and ``resonator_freq`` is stored as
# an interval around the value returned by the grid scan."""

MIN_SAMPLES_FACTOR = 2
# Trial subsets are this many times the number of free parameters, so that each RANSAC
# trial is overdetermined and cannot interpolate an arbitrary subset.


def scaling_slice(sig: np.ndarray, axis: int | None) -> np.ndarray:
    """Min-max scaling over a specific axis of the np.ndarray."""

    def expand(a):
        return np.expand_dims(a, axis) if axis is not None else a

    sig_min = expand(np.min(sig, axis=axis))
    return (sig - sig_min) / (expand(np.max(sig, axis=axis)) - sig_min)


# Sobstitute the extract oeak coordinates
def extract_trace(
    freq: npt.NDArray[np.float64],
    bias: npt.NDArray[np.float64],
    signal: npt.NDArray[np.float64],
    find_min: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Extract one candidate (frequency, bias) peak per bias row."""

    freqs = np.unique(freq)
    biases = np.unique(bias)
    grid = signal.reshape(len(biases), len(freqs))

    z = -grid if find_min else grid
    z = scaling_slice(filter_data(z), axis=1)

    peaks = peaks_finder(freqs, biases, z)
    return peaks["x"]["val"], peaks["y"]["val"]


def smooth_trace_mask(
    bias: npt.NDArray[np.float64],
    frequencies: npt.NDArray[np.float64],
    freq_step: float,
    kernel_size: int = 5,
    tolerance_factor: float = 4.0,
) -> npt.NDArray[np.bool_]:
    """Mask of trace points consistent with a locally smooth arc."""

    # The initial guess scores candidate models by their correlation with the extracted
    # trace, and that score is not robust: the flux-dependent swing of a readout resonator
    # is often only a couple of MHz while a misextracted point can land anywhere in the
    # acquisition window, so a handful of outliers dominates the correlation and the scan
    # locks onto an aliased solution. Removing the obviously discontinuous points before
    # the scan, but not before the fit, resolves this.

    bias = np.asarray(bias, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    if len(frequencies) < kernel_size:
        return np.ones(len(frequencies), dtype=bool)

    order = np.argsort(bias)
    ordered = frequencies[order]
    deviation = np.abs(ordered - medfilt(ordered, kernel_size=kernel_size))
    # Empirically, four times the median deviation separates the arc from stray points;
    # the frequency step is a floor for the case of an essentially noiseless trace.
    tolerance = max(
        tolerance_factor * float(np.median(deviation)), 3.0 * float(freq_step)
    )

    keep = np.zeros(len(frequencies), dtype=bool)
    keep[order] = deviation < tolerance
    return keep


def normalization_grid(
    bias: npt.NDArray[np.float64], size: int = 64
) -> npt.NDArray[np.float64]:
    """Candidate flux-to-bias coefficients implied by the bias sweep."""

    # The limits come from the acquisition rather than from the chip, so nothing here is
    # device specific. Above ``0.5 / bias_step`` there is less than half a period between
    # adjacent bias points and the arc is aliased; below ``0.05 / bias_span`` the arc is
    # indistinguishable from a parabola over the whole window and its period is
    # unmeasurable. The grid is logarithmic because it spans orders of magnitude and the
    # sensitivity to the normalization is multiplicative.

    bias = np.unique(np.asarray(bias, dtype=float))
    span = float(bias[-1] - bias[0])
    step = float(np.min(np.diff(bias)))
    if span <= 0 or step <= 0:
        raise ValueError("bias values must span a finite range")

    low, high = 0.05 / span, 0.5 / step
    if low >= high:
        raise ValueError("bias sampling is too coarse to resolve any flux period")
    return np.geomspace(low, high, size)


def asymmetry_grid(size: int = 16, d_max: float = 0.9) -> npt.NDArray[np.float64]:
    """Candidate junction asymmetries, spaced uniformly in ``sqrt(d)``."""
    # The qubit frequency at the anti-sweetspot goes as ``sqrt(d)``, so uniform spacing in
    # ``d`` wastes resolution exactly where a nominally symmetric SQUID lives. Only
    # ``d >= 0`` is needed since the model depends on ``d**2`` and ``1 - d**2``.
    return np.linspace(0.0, np.sqrt(d_max), size) ** 2


def _dispersive_shape(
    bias: npt.NDArray[np.float64],
    d: float,
    offsets: npt.NDArray[np.float64],
    normalizations: npt.NDArray[np.float64],
    w_max: float,
    resonator_freq: float,
    charging_energy: float,
    min_detuning_factor: float = 3.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Coupling-independent part of the dispersive shift over the parameter grid.

    Returns an array of shape ``(len(normalizations), len(offsets), len(bias))`` and a
    mask flagging grid points where the qubit approaches the resonator and the dispersive
    approximation, along with the model itself, breaks down.
    """
    phase = np.pi * (
        normalizations[:, None, None] * bias[None, None, :] + offsets[None, :, None]
    )
    g_fd = (d**2 + (1 - d**2) * np.cos(phase) ** 2) ** 0.25
    qubit_frequency = (w_max + charging_energy) * g_fd - charging_energy

    detuning = resonator_freq - qubit_frequency
    with np.errstate(divide="ignore", invalid="ignore"):
        shape = 1.0 / detuning - 1.0 / (detuning + charging_energy)

    invalid = ~np.all(np.isfinite(shape), axis=-1) | np.any(
        np.abs(detuning) < min_detuning_factor * charging_energy, axis=-1
    )
    return shape, invalid


def arc_initial_guess(
    bias: npt.NDArray[np.float64],
    frequencies: npt.NDArray[np.float64],
    w_max: float,
    resonator_freq: float,
    charging_energy: float,
    offset_size: int = 81,
    normalization_size: int = 64,
    asymmetry_size: int = 16,
    tie_tolerance: float = 0.01,
) -> tuple[dict[str, float], float]:
    """Global initial guess for the resonator arc, by grid scan."""
    bias = np.asarray(bias, dtype=float)
    measured = np.asarray(frequencies, dtype=float)
    centered_measured = measured - measured.mean()
    measured_ss = float(centered_measured @ centered_measured)
    if measured_ss <= 0:
        raise ValueError("the extracted trace has no frequency variation to fit")

    offsets = np.linspace(-0.5, 0.5, offset_size)  # one period, exact by symmetry
    normalizations = normalization_grid(bias, normalization_size)
    asymmetries = asymmetry_grid(asymmetry_size)

    grid_shape = (len(asymmetries), len(normalizations), len(offsets))
    correlation = np.full(grid_shape, -np.inf)
    slope = np.zeros(grid_shape)
    shape_mean = np.zeros(grid_shape)

    for k, d in enumerate(asymmetries):
        shape, invalid = _dispersive_shape(
            bias, d, offsets, normalizations, w_max, resonator_freq, charging_energy
        )
        shape_mean[k] = shape.mean(axis=-1)
        centered = shape - shape_mean[k][..., None]
        shape_ss = np.einsum("ijk,ijk->ij", centered, centered)
        with np.errstate(divide="ignore", invalid="ignore"):
            covariance = np.einsum("ijk,k->ij", centered, centered_measured)
            rho = covariance / np.sqrt(shape_ss * measured_ss)
            slope[k] = covariance / shape_ss
        rejected = invalid | (shape_ss <= 0) | ~np.isfinite(rho) | (rho <= 0)
        correlation[k] = np.where(rejected, -np.inf, rho)

    if not np.any(np.isfinite(correlation)):
        raise RuntimeError("no feasible initial guess on the parameter grid")

    # Break near-ties toward the smallest normalization. With the normalization free
    # over two decades, an aliased high-frequency solution threads a sparse trace about
    # as well as the true one, and the correlation cannot tell them apart. The lowest
    # frequency consistent with the data is the physically sensible choice, and it is
    # what one would pick by eye. This has to happen here rather than in the refit,
    # because an optimizer started on an alias stays there.
    threshold = np.max(correlation) - tie_tolerance
    candidates = np.any(correlation >= threshold, axis=(0, 2))
    i = int(np.flatnonzero(candidates)[0])
    k, j = np.unravel_index(
        int(np.argmax(correlation[:, i, :])), correlation[:, i, :].shape
    )

    coupling_squared = slope[k, i, j]
    guess = {
        "g": float(np.sqrt(coupling_squared)),
        "d": float(asymmetries[k]),
        "offset": float(offsets[j]),
        "normalization": float(normalizations[i]),
        "resonator_freq": float(
            measured.mean() - coupling_squared * shape_mean[k, i, j]
        ),
        "charging_energy": float(charging_energy),
    }
    return guess, float(correlation[k, i, j])


def _validated_inputs(
    qubit,
    w_max: float,
    bare_resonator_frequency: float,
    charging_energy: float,
    frequencies: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    """Sanity-check the platform values the model is anchored to, all in GHz."""
    if not np.isfinite(w_max) or w_max <= 0:
        raise ValueError("qubit frequency is missing from the platform configuration")

    if not 0.05 < charging_energy < 0.5:
        log.warning(
            f"[resonator_flux] qubit {qubit}: charging energy is {charging_energy:.3f} "
            "GHz, which is outside the transmon range; falling back to 0.2 GHz"
        )
        charging_energy = 0.2

    if abs(bare_resonator_frequency - np.mean(frequencies)) > 0.5:
        log.warning(
            f"[resonator_flux] qubit {qubit}: bare resonator frequency "
            f"({bare_resonator_frequency:.4f} GHz) is inconsistent with the acquired "
            "window; using the measured mean instead"
        )
        bare_resonator_frequency = float(np.mean(frequencies))

    return w_max, bare_resonator_frequency, charging_energy


def arc_diagnostics(
    parameters: dict[str, float],
    covariance: np.ndarray,
    free_parameters: tuple[str, ...],
    bias: npt.NDArray[np.float64],
    bounds: tuple[list[float], list[float]],
    at_bound_rtol: float = 1e-2,
) -> dict[str, Any]:
    """Assess which fitted parameters the data actually constrains."""

    # ``coverage`` is the fraction of a flux period spanned by the sweep. Near the
    # sweetspot the model expands as ``1 - (1 - d**2) * (pi * phi)**2 / 4``, so the
    # asymmetry enters only through the product ``(1 - d**2) * normalization**2`` and is
    # degenerate with the flux coefficient; it is resolved only by the flattening of the
    # arc further out. Below roughly a third of a period the asymmetry is not measurable
    # at all, whatever the fit reports.

    # ``at_bound`` lists parameters resting against a bound, which is the usual sign of an
    # optimizer that has run away along a degenerate direction.

    errors = np.sqrt(np.diag(covariance))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / np.outer(errors, errors)
    off_diagonal = np.abs(correlation - np.eye(len(free_parameters)))
    worst = np.unravel_index(int(np.nanargmax(off_diagonal)), off_diagonal.shape)

    span = float(np.max(bias) - np.min(bias))
    uncertainties = dict(zip(free_parameters, errors))

    at_bound = [
        name
        for name, low, high in zip(free_parameters, bounds[0], bounds[1])
        if min(abs(parameters[name] - low), abs(parameters[name] - high))
        <= at_bound_rtol * max(abs(high - low), 1e-12)
    ]

    asymmetry_error = uncertainties.get("d")
    return {
        "uncertainties": uncertainties,
        "coverage": parameters["normalization"] * span,
        "worst_correlation": float(off_diagonal[worst]),
        "worst_correlated_pair": (
            free_parameters[worst[0]],
            free_parameters[worst[1]],
        ),
        "at_bound": at_bound,
        "asymmetry_identifiable": bool(
            asymmetry_error is not None
            and np.isfinite(asymmetry_error)
            and asymmetry_error < 0.5 * max(parameters["d"], 0.05)
            and not at_bound
        ),
    }


def fit_arc(
    qubit,
    bias: npt.NDArray[np.float64],
    frequencies: npt.NDArray[np.float64],
    w_max: float,
    bare_resonator_frequency: float,
    charging_energy: float,
    residual_threshold: float,
    fixed_parameters: tuple[str, ...] = DEFAULT_FIXED_PARAMETERS,
    freq_step: float | None = None,
    random_state: int = 0,
) -> tuple[dict[str, float], npt.NDArray[np.bool_], dict[str, Any]]:
    """Fit the flux dependence of a readout resonator."""
    bias = np.asarray(bias, dtype=float)
    frequencies_ghz = np.asarray(frequencies, dtype=float) * HZ_TO_GHZ

    free_parameters = tuple(n for n in ARC_PARAMETERS if n not in fixed_parameters)
    min_samples = MIN_SAMPLES_FACTOR * len(free_parameters)
    if len(bias) < min_samples:
        raise RuntimeError(
            f"only {len(bias)} usable points extracted, but {min_samples} are needed "
            f"for a {len(free_parameters)}-parameter model; check the signal to noise "
            "ratio and the frequency window"
        )

    w_max_ghz, f_bare_ghz, charging_energy_ghz = _validated_inputs(
        qubit,
        w_max * HZ_TO_GHZ,
        bare_resonator_frequency * HZ_TO_GHZ,
        charging_energy * HZ_TO_GHZ,
        frequencies_ghz,
    )

    # The grid scan is not robust to outliers, so it is fed a smoothness-filtered subset
    # of the trace. The fit itself still sees every point.
    step = (
        residual_threshold * HZ_TO_GHZ / 5
        if freq_step is None
        else freq_step * HZ_TO_GHZ
    )
    smooth = smooth_trace_mask(bias, frequencies_ghz, step)
    if smooth.sum() < min_samples:
        smooth = np.ones(len(bias), dtype=bool)

    guess, correlation = arc_initial_guess(
        bias[smooth],
        frequencies_ghz[smooth],
        w_max=w_max_ghz,
        resonator_freq=f_bare_ghz,
        charging_energy=charging_energy_ghz,
    )
    log.info(
        f"[resonator_flux] qubit {qubit}: initial guess reached a correlation of "
        f"{correlation:.4f} on {int(smooth.sum())}/{len(bias)} trace points"
    )

    normalizations = normalization_grid(bias)
    limits = dict(ARC_BOUNDS)
    limits["normalization"] = (float(normalizations[0]), float(normalizations[-1]))
    limits["resonator_freq"] = (
        guess["resonator_freq"] + ARC_BOUNDS["resonator_freq"][0],
        guess["resonator_freq"] + ARC_BOUNDS["resonator_freq"][1],
    )
    bounds = (
        [limits[name][0] for name in free_parameters],
        [limits[name][1] for name in free_parameters],
    )

    def model(x, *values):
        parameters = dict(guess)
        parameters.update(zip(free_parameters, values))
        return transmon_readout_frequency(
            xi=x,
            xj=0,
            w_max=w_max_ghz,
            crosstalk_element=1,
            **{name: parameters[name] for name in ARC_PARAMETERS},
        )

    p0 = [float(np.clip(guess[name], *limits[name])) for name in free_parameters]
    popt, inliers, pcov = ransac_fit(
        bias,
        frequencies_ghz,
        fit_function=model,
        residual_threshold=residual_threshold * HZ_TO_GHZ,
        bounds=bounds,
        p0=p0,
        min_samples=min_samples,
        f_scale=step,
        random_state=random_state,
        return_covariance=True,
    )

    parameters = dict(guess)
    parameters.update(zip(free_parameters, (float(value) for value in popt)))
    diagnostics = arc_diagnostics(parameters, pcov, free_parameters, bias, bounds)
    diagnostics["initial_guess_correlation"] = correlation
    diagnostics["rms"] = (
        float(
            np.sqrt(
                np.mean((model(bias[inliers], *popt) - frequencies_ghz[inliers]) ** 2)
            )
        )
        * GHZ_TO_HZ
    )

    if not diagnostics["asymmetry_identifiable"]:
        log.warning(
            f"[resonator_flux] qubit {qubit}: the junction asymmetry is not constrained "
            f"by this dataset (d = {parameters['d']:.3f} +- "
            f"{diagnostics['uncertainties'].get('d', float('nan')):.3f}, flux coverage "
            f"{diagnostics['coverage']:.2f} periods). Treat it as a shape parameter of "
            "the fit rather than a measurement, and widen the bias sweep to cover a "
            "full period if the asymmetry is wanted."
        )
    if diagnostics["at_bound"]:
        log.warning(
            f"[resonator_flux] qubit {qubit}: {', '.join(diagnostics['at_bound'])} "
            "ended the fit against a bound, which usually means the fit drifted along a "
            "degenerate direction; the other parameters may still be sound but this one "
            "is not a measurement"
        )

    for name in ("g", "resonator_freq", "charging_energy"):
        parameters[name] *= GHZ_TO_HZ

    return parameters, inliers, diagnostics
