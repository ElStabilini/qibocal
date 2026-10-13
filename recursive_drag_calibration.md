# Roadmap: recursive-DRAG (R1D/R2D) fast π-pulse calibration for qibocal

Goal: implement the fast-π-pulse scheme of **arXiv:2512.19919** — Da Costa
Jesus, Li, Gao, Barends, Cárdenas-López, Motzoi, *"Analytical blueprint for
99.999% fidelity X-gates on present superconducting hardware under strong
driving"* — as a `recursive_drag` envelope in qibolab plus a family of qibocal
protocols that calibrate its prefactors, following the exact
`Parameters`/`Data`/`Results`/`_acquisition`/`_fit`/`_plot`/`_update`/`Protocol`
structure used by every other protocol in `src/qibocal/protocols/`.

Versions assumed: qibolab ≥ 0.2.16, qibocal 0.2.7 (`pyproject.toml`).
Formulas below were read from the **arXiv LaTeX source**, not the PDF text
layer, and equation numbers refer to the published numbering.

---

## Phase 0 — What the paper does and does not give us

Two points to settle before any code is written, because they determine the
whole shape of the work.

**The paper contains no experimental calibration protocol.** It is theory plus
numerics. There is no methods section, no pulse sequence, no error-amplification
scheme. What it *does* give:

- **(a)** a closed-form envelope ansatz built by applying the DRAG frame
  transformation recursively — Eqs. (36), (37);
- **(b)** a five-parameter family `{β, α₁₂, α₀₂, α₁₃, δ_c}` layered on top of
  it — Eqs. (26), (38);
- **(c)** analytic near-optimal values for `β`, `α`, `δ_c` — Eqs. (32), (34),
  (35) — explicitly offered as *"a warm start for further calibration in an
  experimental setting"*;
- **(d)** the statement that these five parameters are orthogonal only to
  `O(Ω_x²/Δ₂²)` and couple at higher order, with the admission that *"analytical
  prediction of the three DRAG prefactors becomes intractable"*.

Point **(d)** is what *forces* a round-robin iterative calibration. That loop is
**our design**, constructed from the paper's parameterization — it is not a
transcription of anything the paper prescribes. Worth stating plainly in the
eventual PR description so nobody reads more authority into it than it has.

**Vocabulary correction to the original framing.** This is *recursive* DRAG —
nested frame transformations (`S₁`…`S₉`) collapsing into a single analytic
waveform — and the calibration of the prefactors sitting on top of that waveform
is *iterative*. The two are separate concerns and should stay separate in the
code: **the recursion lives in the envelope, the iteration lives in the
runcard.** Conflating them is the main way this implementation could go wrong
structurally.

**What the recursions actually suppress** (Fig. 1b, Figs. 3–5):

| Pulse | Suppresses | Mechanism |
|---|---|---|
| linear DRAG | \|1⟩↔\|2⟩ (single-photon) | `Ω_y = −Ω̇_x/(2Δ₂)` |
| **R1D** | + \|0⟩↔\|2⟩ (two-photon) | first recursion, Eq. (8)/(37) |
| **R2D** | + \|1⟩↔\|3⟩ (two-photon) | second recursion, Eq. (36) |

Reported performance: 10⁻⁵ infidelity at ~11.8 ns with **no** calibration, and
at **~6.75 ns** once the five prefactors are calibrated. The calibration is
therefore not a nicety — it is where roughly half the speedup comes from.

---

## Phase 1 — The physics, reduced to what the code must compute

### 1.1 The equations

Base shape, Eq. (16) — the pulse of Ref. [10]. Chosen over `sin⁴(πt/T)` because
it satisfies the R1D **and** R2D boundary conditions while having a lower peak
amplitude:

```
Ω₂(x) = A [ 1/2 − (9/16)·cos(2πx) + (1/16)·cos(6πx) ],    x = t/T ∈ [0,1]
```

Verified by hand: `Ω₂(0) = Ω₂(1) = 0`, `Ω̇₂(0) = Ω̇₂(1) = 0`, and
`Ω̈₂(0) = Ω̈₂(1) = 0` — that last one is the condition R1D adds beyond Eq. (5b)
and is exactly why the Hann pulse is not usable here (Sec. II C).

Second recursion (Eq. 36), then first (Eq. 37):

```
Ω₁  = sqrt( Ω₂² + (2·α₁₃/Δ₃²)·( Ω̇₂² + Ω̈₂·Ω₂ ) )
Ω_x = sqrt( Ω₁² + (2·α₀₂/Δ₂²)·( Ω̇₁² + Ω̈₁·Ω₁ ) )
```

Quadrature, superlinear correction (Eq. 18, with `λ₂ = √2`), and the
parameterization (Eq. 38):

```
Ω_y  = −Ω̇_x / (2·Δ₂)
Ω_x ← Ω_x − Ω_x³/(4·Δ₂²)
Ω    = β·( Ω_x + i·α₁₂·Ω_y ),   plus constant detuning δ_c
```

Constants: `Δ₂` is the anharmonicity, `Δ₃ = 3·Δ₂` for a transmon ladder
(`Δⱼ = j(j−1)/2·α_anh`), and `λⱼ = √j`.

**One generator covers all three pulse families**, which is the single most
useful structural fact here:

```
α₀₂ = α₁₃ = 0   →  linear DRAG
α₁₃ = 0         →  R1D
both ≠ 0        →  R2D
```

So there is no need for three envelope classes, three protocols, or any
dispatch — and every comparison figure in the paper is reachable from one code
path.

### 1.2 Make everything dimensionless in units of the pulse duration

The recursion is **not** scale-free: `Ω̇²/Δ²` carries physical time. But
qibolab's `BaseEnvelope.i(samples)` receives only a sample count, and
duplicating `duration` as an envelope field would go stale the instant anyone
edits `Pulse.duration`.

The escape is qibolab's own established convention — `Gaussian.rel_sigma`,
documented as *"in units of the interval duration"*. Write `ω(x) ≡ Ω(t)·T` and
`Δ̃ ≡ Δ·T`; then with `′ = d/dx`:

```
ω₁² = ω₂² + (2·α₁₃/Δ̃₃²)·( ω₂′² + ω₂″·ω₂ )
ω_x² = ω₁² + (2·α₀₂/Δ̃₂²)·( ω₁′² + ω₁″·ω₁ )
ω_y  = −ω_x′ / (2·Δ̃₂)
```

Every equation closes with no reference to `T`. The envelope fields become
`rel_anharmonicity = Δ₂·T` and `rel_detuning = δ_c·T`, both pure numbers.

*(Derivation check: with `Ω = ω/T` and `x = t/T`, we get `Ω̇ = ω′/T²` and
`Ω̈ = ω″/T³`; substituting into Eq. (37) and multiplying through by `T²` gives
the line above. Same for `Ω_y`.)*

### 1.3 The overall scale A is fixed, not free

Eq. (5a) requires `∫₀ᵀ Ω_x dt = π`, i.e. `∫₀¹ ω_x dx = π`. The recursions are
**nonlinear** in `A`, so this cannot be done by normalizing afterwards — solve
for `A` with a 1-D root find (`scipy.optimize.brentq`; scipy is already a
qibolab dependency via `scipy.signal.windows.gaussian`).

This is not pedantry: the analytic `β` warm start (Eq. 34) is a perturbation
*around* `β = 1`, so if `A` is set by naive peak normalization instead, Eq. (34)
predicts a correction to a quantity that does not mean what it is supposed to
mean. `Pulse.amplitude` then carries `β` times the hardware's volts-per-Ω
conversion, calibrated as usual by Rabi/flipping.

### 1.4 Positivity gate → minimum gate time

For large `α₀₂`/`α₁₃` the square-root arguments go negative. This is not a
numerical accident, it is the paper's **minimum gate time**, Eqs. (24), (25) and
(39):

```
T_min = π·sqrt( n·(α₁₃Δ₂² + α₀₂Δ₃²) + ν ) / (Δ₂·Δ₃)
ν     = sqrt( n²(α₁₃²Δ₂⁴ + α₀₂²Δ₃⁴) + 2(2−5n)·n·α₁₃·α₀₂·Δ₂²Δ₃² )
```

Raise a `ValueError` naming `T_min` rather than emitting NaNs into a waveform
that then gets played at a qubit.

### 1.5 δ_c leaves a frame offset — the main hazard

A constant detuning during the pulse is equivalent to keeping the carrier on
resonance and multiplying the envelope by `e^{i·δ_c·t}` — but the frame at
`t = T` is then rotated by `δ_c·T`, which must be undone with a `VirtualZ`.

*(Check: in the qubit frame `H = δ_c·σ_z/2 + (Ω_x σ_x + Ω_y σ_y)/2`. Moving to a
frame rotating at `δ_c` removes the `σ_z` term and rotates the drive as
`Ω → Ω·e^{iδ_c t}`; the frames differ by `R_z(δ_c T)` at the end.)*

See Phase 4 for where that lands. **This is the single most likely source of a
silent, hard-to-diagnose calibration error in the whole plan.**

---

## Phase 2 — qibolab: the `recursive_drag` envelope

One upstream PR against `qibolab`, touching `qibolab/_core/pulses/envelope.py`
(class + `__all__` + the `Envelope` discriminated union) and its tests. This is
the only cross-repo dependency and should be opened first, since everything
downstream imports it.

```python
class RecursiveDrag(BaseEnvelope):
    """Recursive DRAG (R1D/R2D) envelope, arXiv:2512.19919."""

    kind: Literal["recursive_drag"] = "recursive_drag"

    rel_anharmonicity: float  # Δ₂·T, dimensionless (negative for a transmon)
    delta3_ratio: float = 3.0  # Δ₃/Δ₂; 3 for a transmon ladder
    alpha12: float = 1.0  # Y-quadrature prefactor
    alpha02: float = 0.0  # first recursion  (|0⟩↔|2⟩)
    alpha13: float = 0.0  # second recursion (|1⟩↔|3⟩)
    rel_detuning: float = 0.0  # δ_c·T, applied as a phase ramp
    superlinear: bool = True  # Eq. (18) cubic correction
```

Implementation notes for `i(samples)` / `q(samples)`:

- Build on an **oversampled** grid (~16× `samples`) over `x ∈ [0,1]`; take
  `ω₂′` and `ω₂″` **analytically** (the base shape is two cosines), take the
  recursion derivatives with `np.gradient`, then decimate. Differentiating a
  `sqrt` of a sampled array at hardware resolution is the obvious accuracy
  trap — `ω₁` is a square root and its second derivative is what feeds the
  outer recursion.
- Run the `∫ω_x dx = π` root-solve (§1.3) first, then normalize the final
  complex waveform to unit peak so `Pulse.amplitude` keeps its usual `[−1, 1]`
  meaning.
- `Snz.i()` is the existing precedent for an envelope that asserts on invalid
  geometry — follow it for the §1.4 positivity gate.

qibolab-side tests:

- boundary conditions `Ω(0) = Ω(T) = Ω̇(0) = Ω̇(T) = 0`;
- `α₀₂ = α₁₃ = 0` reproduces plain `Drag` to float tolerance;
- sub-`T_min` parameters raise rather than returning NaN;
- round-trip through `TypeAdapter(Envelope)` (the union is discriminated on
  `kind`, and all pulse models are `extra="forbid", frozen=True`).

> **Fallback, explicitly not the merge target.** The same generator can be driven
> through qibolab's existing `Custom` envelope with zero qibolab changes. But
> `Custom.i()` raises unless `len(i_) == int(duration * sampling_rate)`, so a
> stored `Custom` RX is welded to one sampling rate and one duration and breaks
> every duration sweep (`rabi_length`, etc.), and `parameters.json` ends up
> holding an opaque base64 sample blob. Use it as a local bridge if the qibolab
> PR stalls; never ship it.

---

## Phase 3 — qibocal: bootstrapping the envelope onto RX

New package `src/qibocal/protocols/recursive_drag/`.

### `recursive_drag_init`

A bootstrap `Protocol` in the `calibrate_mixers.py` mould — no real measurement,
`_update` mutates platform config through dotted paths. It:

- reads `platform.calibration.single_qubits[q].qubit.anharmonicity`
  (`src/qibocal/calibration/calibration.py:72`, computed as
  `frequency_12 − frequency_01`, so **`qubit_spectroscopy_ef` must have run
  first**) and forms `rel_anharmonicity = 2π · anh · T · 1e-9`;
- takes `duration`, `alpha12`, `alpha02`, `alpha13` from the runcard;
- computes the analytic warm starts — `δ_c` from Eq. (32),

  ```
  δ_c ≈ 0.712 · (λ₂² − 4·α₁₂)/Δ₂ · π²/T²      with λ₂ = √2
  ```

  and `β` from Eq. (34) — in a small **pure** module `warm_start.py` with no
  platform dependency, so it is unit-testable on its own;
- writes envelope + amplitude into `native_gates.single_qubit.{q}.RX.0.1`.

This is the first action of every runcard and is what makes the paper's "warm
start" concrete.

**Relation to `set_envelope`.** This is deliberately narrower than the generic
`set_envelope` routine in `pulse_envelope_switch.md` — that routine remains
worth building on its own terms. Its Phase 0 findings apply **verbatim** here
and should be treated as already-established constraints, not re-derived:

- the whole `envelope` sub-object must be replaced as one dict-valued leaf, not
  patched field-by-field (`extra="forbid"`);
- durable mutation belongs in `_update`, not `_acquisition`, to survive the
  `qq acquire` → `qq fit` platform reload;
- the cross-invocation staleness gap is real and undetected, and applies to this
  work too.

### `update.py` helper

One addition next to `drag_pulse_beta` (`src/qibocal/update.py:217`):

```python
def recursive_drag_prefactor(
    field: str, value: float, platform, qubit, rx90: bool = False
):
    key = "RX90" if rx90 else "RX"
    platform.update(
        {f"native_gates.single_qubit.{qubit}.{key}.0.1.envelope.{field}": value}
    )
```

Single-field dotted paths are safe **here** because every prefactor lives inside
an envelope that is already `recursive_drag` — unlike `drag_pulse_beta`, which
only works by the accident of `Gaussian` and `Drag` sharing `rel_sigma`.

Note `drag_pulse_beta` has no `RX90` variant today; if π/2 pulses are wanted
later, mirror the `rx90: bool` switch of `update.drive_amplitude`.

---

## Phase 4 — qibocal: `utils.py` and the VirtualZ hazard

`recursive_drag/utils.py` gets one helper used by every protocol below:

```python
def repeat_rx(platform, qubit, n, **envelope_overrides) -> PulseSequence
```

It must append the **whole** `natives.RX()` sequence `n` times, **not**
`natives.RX()[0]`. Reason: once `rel_detuning ≠ 0`, the native RX has to carry a
trailing `VirtualZ(phase=−rel_detuning)` to undo the frame rotation the phase
ramp induces (§1.5). `Native` *is* a `PulseSequence`, so a two-element RX
`[(qd, Pulse), (qd, VirtualZ)]` is legal and leaves `natives.RX()[0]` and the
`RX.0.1` dotted update paths working for every existing protocol.

**Known consequence — document it, do not paper over it.** `flipping_sequence`
(`src/qibocal/protocols/flipping.py:44`) does

```python
sequence.extend([(qd_channel, qd_detuned)] * (flips * (4 if rx90 else 2)))
```

re-emitting only the bare pulse. It would therefore **silently drop the
VirtualZ** and mis-calibrate at `rel_detuning ≠ 0`. Two options:

1. patch `flipping` to extend over the full native (preferred — it also fixes
   the same latent issue for any future multi-element native); or
2. state the restriction explicitly in the docs page and in the protocol
   docstring.

Either way it must be resolved before hardware time is spent, because the
failure mode is a plausible-looking but wrong amplitude calibration.

---

## Phase 5 — qibocal: the per-parameter protocols

Four parameters, four observables. Each is an ordinary protocol building **one
`PulseSequence` per swept value** and submitting the list to `platform.execute`
— there is no qibolab `Sweeper` parameter for an envelope field, the same
constraint `drag_tuning` already lives with (`Parameter` enum is
`frequency, amplitude, duration, duration_interpolated, relative_phase, phase,
offset`).

| Parameter | Protocol | Sequence | Observable | Fit |
|---|---|---|---|---|
| **β** (amplitude) | *reuse existing* `flipping` | `R(π/2)` then `(X_π)^2N` | P₁ | already implemented |
| **α₁₂** (DRAG prefactor) | `recursive_drag_alpha12` | `(X_π X_−π)^N` | P₁ | cosine, minimum |
| **δ_c** (detuning) | `recursive_drag_detuning` | YpX9 / XpY9 pair | P₁ | two lines, intersection |
| **α₀₂, α₁₃** (leakage) | `recursive_drag_leakage` | `(X_π)^2N` | P₂ (and P₃) | parabola, minimum |

**β needs no new code.** It is literally `Pulse.amplitude`, and `flipping`
already amplifies over/under-rotation and calls `update.drive_amplitude`.
Reuse it, subject to the Phase 4 caveat.

**`recursive_drag_alpha12`** is structurally a near-clone of `drag_tuning`
(`src/qibocal/protocols/drag/drag.py`) with `Drag(beta=…)` swapped for
`RecursiveDrag(alpha12=…)`. Subclass `DragTuningParameters`/`DragTuningData` the
way `drag_simple` already subclasses them.

**`recursive_drag_detuning`** reuses `drag_simple`'s YpX9/XpY9 construction and
its two-line-intersection fit (`src/qibocal/protocols/drag/drag_simple.py`),
sweeping `rel_detuning` instead of `beta` — those sequences are built to be
sensitive to exactly the phase error `δ_c` controls.

> ⚠️ **This is the step with the least support from the paper and the most need
> for validation.** The paper prescribes no sequence for `δ_c`; the YpX9/XpY9
> choice is ours. Treat a disagreement between the fitted `δ_c` and the Eq. (32)
> prediction as a bug in the sequence choice until proven otherwise — Eq. (32)
> is reported to agree well with both numerics (Fig. 6a) and experiment
> (Ref. [30]), so it is the more trustworthy of the two.
>
> Alternative worth trying if YpX9/XpY9 disappoints: a hardware
> `Parameter.frequency` sweep on the drive channel is much faster, but it shifts
> the frame for the whole sequence rather than only during the pulse, so it is
> not self-consistent with how `δ_c` will actually be applied. Prefer
> self-consistency over speed here.

**`recursive_drag_leakage`** is the genuinely new protocol. `(X_π)^2N` returns
to |0⟩ ideally, so residual P₂ is amplified leakage — directly the quantity
`α₀₂` is designed to null. Requires `qutrit_classification`
(`src/qibocal/protocols/classification/qutrit_classification.py`) to have run so
the discriminator emits states 0/1/2; then `probability(result, state=2)` from
`src/qibocal/result.py:79`. Sweep `α₀₂` and `α₁₃` independently (one
`prefactor: str` field selecting which, validated in `__post_init__` the way
`FlippingParameters` validates its ints), fit a parabola, take the minimum.

### House conventions all four must follow

Confirmed against the existing protocols — deviating from any of these will look
wrong in review:

- module-level `__all__` listing the public names **and** the protocol object;
- `@dataclass` `Parameters` that never redeclare `nshots`/`relaxation_time`
  (`Parameters.load` sets them post-construction), with validation in
  `__post_init__`;
- a module-level `np.dtype` constant for the `Data` record, and
  `data: dict[QubitId, npt.NDArray[XType]] = field(default_factory=dict)`;
- `_acquisition(params, platform, targets)` and `_fit(data)` — the **type
  annotations are load-bearing**, `Protocol` derives `parameters_type`/
  `data_type`/`results_type` by `inspect.signature`;
- `_fit` wrapped in `try/except Exception` + `log.warning`, so a failed qubit
  simply drops out of the results dict and `Completed.update_platform`'s
  `except KeyError` skips it;
- `_plot(data, target, fit) -> (list[go.Figure], str)` using
  `table_html(table_dict(...))`;
- `<name> = Protocol(_acquisition, _fit, _plot, _update)` at the bottom.

Registration: `recursive_drag/__init__.py` re-exports, then
`from . import recursive_drag` / `from .recursive_drag import *` /
`__all__ += recursive_drag.__all__` in `src/qibocal/protocols/__init__.py`,
which builds `PROTOCOLS = {name: globals()[name] for name in __all__}`.

---

## Phase 6 — The iterative runcard

The loop **is** the runcard, because that is already qibocal's iteration
mechanism: `Runcard.actions` is a flat list run in order against one live
platform, and `update: true` feeds each fit forward. There is no loop construct
in the runcard schema and none should be invented.

```yaml
platform: mock
targets: [0]
actions:
  - {id: init,   operation: recursive_drag_init,     parameters: {duration: 8, alpha02: 1.3, alpha13: 1.7}}

  # ---- round 1 ----
  - {id: amp1,   operation: flipping,                parameters: {nflips_max: 20, nflips_step: 2, nshots: 2000}}
  - {id: a12_1,  operation: recursive_drag_alpha12,  parameters: {...}}
  - {id: det1,   operation: recursive_drag_detuning, parameters: {...}}
  - {id: leak1,  operation: recursive_drag_leakage,  parameters: {prefactor: alpha02, ...}}
  - {id: leak1b, operation: recursive_drag_leakage,  parameters: {prefactor: alpha13, ...}}

  # ---- rounds 2 and 3: repeat the five actions verbatim with new ids ----
```

Ship it as `doc/source/protocols/recursive_drag/recursive_drag.yml`.

**Ordering rationale:** amplitude first (it is the largest and most robustly
measured error), then `α₁₂` (which per Eq. (38) and Fig. 9 largely determines
both `β` and `δ_c`), then `δ_c`, then the two leakage prefactors last (weakest
signal, most amplification needed).

**Convergence criterion:** round-to-round prefactor changes fall below the fit
error bars, readable straight off the per-action report pages. Three rounds is a
starting guess, not a result.

---

## Phase 7 — Tests and docs

- **Fit regression**, mirroring `tests/test_drag.py`: store
  `tests/tests_data/<id>-0/` folders (`action.yml`, `data.npz`,
  `results.json`), reload with `Data.load`/`Results.load`, re-run `_fit`,
  compare with `approx_for_regression`.
- **Integration**: add one action per new protocol to
  `tests/runcards/protocols.yml`. That alone gives acquisition, fit, plot,
  update and report-rendering coverage through `tests/test_protocols.py`
  against the mock platform (whose RX is already `drag`, `duration: 40`).
- **Waveform unit tests** (no platform needed): boundary conditions; `α → 0`
  limits reduce to DRAG; `∫ω_x dx = π` holds after the root-solve; sub-`T_min`
  raises.
- **Warm-start unit tests**: Eq. (32) must reproduce the paper's reported
  `δ_c = 0` near `α ≈ 0.5` for `λ₂ ≈ √2`, and the `δ_c ∝ 1/T²` scaling — both
  are stated in the text as matching the Ref. [4] experiment, so they are real
  external checks rather than self-consistency.
- **VirtualZ regression**: a test that `repeat_rx(..., n=3)` emits three
  `VirtualZ` instructions when `rel_detuning ≠ 0` — guards the Phase 4 hazard.
- **Docs**: `doc/source/protocols/recursive_drag/recursive_drag.rst` with
  Method/Parameters/Example/Requirements sections and
  `.. autoclass:: ... :noindex:`, `:cite:p:` entries in `doc/source/refs.bib`,
  a toctree entry in `doc/source/protocols/index.rst`, and an `automodule` stub
  in `doc/source/api-reference/`. The **Requirements** section must state the
  prerequisites explicitly: calibrated `frequency_12` (for the anharmonicity)
  and a working qutrit discriminator.

---

## Verification

1. **Waveform, offline — do this first.** Run the qibolab envelope tests, then
   plot `|FFT(Ω)|` and `|FFT(Ω_x²)|` for the calibrated parameters and confirm
   the spectral holes the paper shows in Figs. 3–5: a hole at `Δ₂` in `Ω`
   (|1⟩↔|2⟩), and holes at **both** `Δ₂` and `Δ₃` in `Ω_x²` (|0⟩↔|2⟩ and
   |1⟩↔|3⟩). This is the sharpest available check that the recursion is
   correctly implemented and it needs no hardware at all.
2. **Mock platform.** `qq run` the Phase 6 runcard against `mock`; confirm it
   completes, the report renders, and `new_platform/parameters.json` carries a
   `recursive_drag` envelope with updated prefactors.
3. **Split invocation.** `qq acquire` then `qq fit` on the same runcard, per the
   durability analysis already done in `pulse_envelope_switch.md` Phase 0.
4. **Hardware.** Run the three-round runcard. Check the prefactors converge and
   land near Fig. 8's values (`α₁₂ ≈ 1.0–1.3`, `α₀₂ ≈ 1.3`, `α₁₃ ≈ 1.7` at
   7–8 ns), and that the fitted `δ_c` tracks Eq. (32). Then the actual
   deliverable: `standard_rb` before and after, at progressively shorter RX
   durations, against a Gaussian-DRAG baseline.

---

## Risks to watch throughout

- **The VirtualZ / `flipping` interaction (§1.5, Phase 4)** is the most likely
  source of a silent wrong answer in this plan. It produces a clean-looking fit
  and a wrong amplitude. Resolve it before hardware time.
- **The `δ_c` sequence choice (Phase 5)** is ours, not the paper's. It is the
  weakest link in the chain and the first thing to question if the loop fails to
  converge.
- **Anharmonicity provenance.** `Δ₂` comes from `frequency_12 − frequency_01`
  and the whole waveform depends on it. A stale or wrong `frequency_12` silently
  produces a subtly wrong pulse shape rather than an error. Consider logging the
  `rel_anharmonicity` actually used in `recursive_drag_init`'s report.
- **`Δ₃ = 3Δ₂` is a transmon approximation.** It is exposed as
  `delta3_ratio` rather than hard-coded precisely so it can be challenged; the
  paper notes (Eq. 25 discussion) that `T_min` is genuinely sensitive to `Δ₃`.
- **Cross-invocation platform staleness** — the pre-existing, undetected gap
  documented in `pulse_envelope_switch.md` Phase 0/6 — applies here too. A
  multi-round calibration split across separate `qq acquire` invocations without
  `qq update` in between will silently use stale prefactors.
- **Scope creep toward numerical optimization.** The paper's best results come
  from full black-box numerical optimization of the final unitary. That is *not*
  what this plan builds, and it does not fit the qibocal protocol model. Keep
  the deliverable as single-parameter error-amplification protocols composed by
  a runcard.

---

## Dependency order

```
Phase 2 (qibolab PR)  ──▶ Phase 3 ──▶ Phase 4 ──▶ Phase 5 ──▶ Phase 6 ──▶ Phase 7
```

The qibolab PR is the only cross-repo dependency and blocks everything
downstream, so open it first. Phase 1 (§1.1–1.5) is analysis and is already
done — it is written down here so it can be checked rather than re-derived.
