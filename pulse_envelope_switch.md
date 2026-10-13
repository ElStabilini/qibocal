# Roadmap: pulse-envelope switching routine for qibocal

Goal: a `set_envelope` routine that lets an experimenter replace the envelope of
a native gate from a runcard, invalidating the platform's calibration state for
that gate exactly as if it had never been calibrated — without hand-editing
`parameters.json` or `calibration.json`, and without touching qibolab unless a
genuinely new envelope shape is needed.

Versions confirmed against the actual installed environment: qibolab 0.2.16,
qibocal 0.2.6 (`.venv` at the repo root). **Feasibility confirmed** — every
claim below was checked against the real source, not assumed.

---

## Phase 0 — Reconnaissance (done)

All checklist items from the original draft have been resolved by reading the
real source:

- **Envelope union** (`qibolab/_core/pulses/envelope.py`): `Envelope` is a
  pydantic discriminated union on `kind` — `Rectangular`, `Exponential`,
  `Gaussian`, `GaussianSquare`, `Drag`, `Snz`, `Custom`. All envelope/pulse
  models use `ConfigDict(extra="forbid", frozen=True)`
  (`qibolab/_core/serialize.py`), so a mismatched field for a given `kind`
  raises `pydantic.ValidationError` immediately at construction/validation,
  not later.
- **Native gate structure**: `Native` (`qibolab/_core/native.py`) is literally
  a `PulseSequence` — a list of `(channel_id, PulseLike)` tuples.
  `platform.natives.single_qubit[q].RX[0][1]` is the `Pulse`.
  `MZ[0][1]` is a `Readout` with a `.probe: Pulse` (the actual drive/readout
  pulse) and a separate `.acquisition: Acquisition` with no envelope — so MZ
  envelope swaps must target `.probe.envelope`, not a top-level `.envelope`.
- **Update signature**: confirmed via `qibocal/protocols/rabi/amplitude.py` —
  `_update(results, platform, target)`, called once per target.
- **`Parameters` base dataclass** (`qibocal/auto/operation.py`): `nshots` and
  `relaxation_time` have no defaults (set post-construction by
  `Parameters.load`), so subclasses can freely add required fields.
- **`Results.__contains__`**: checks a key against every dict-valued field of
  the concrete `Results` subclass — but this is *not* what gates whether
  `_update` runs. `Completed.update_platform` (`qibocal/auto/task.py`) calls
  `operation.update(results, platform, target)` unconditionally per target,
  wrapped in `try/except KeyError`; a missing key in a dict-valued `Results`
  field is what actually causes the per-target skip.
- **`initialize_parameters` + `Hardware`** (qibolab
  `_core/platform/parameters.py`): the default envelope assigned to *every*
  native pulse, including RX, is `Rectangular()` with `amplitude=0,
  duration=0` — i.e. "reset to default" means a fully inert, zero pulse. This
  is exactly what "invalidated" should mean for this routine.
- **`Calibration` model** (`qibocal/calibration/calibration.py`): per-qubit
  fields `resonator`, `qubit` (frequency/sweetspot), `readout`
  (fidelity/IQ-blob data), `t1`/`t2`/`t2_spin_echo`, `rb_fidelity`; per-pair
  `rb_fidelity`, `cz_fidelity`, `coupling`. `rb_fidelity` is directly
  envelope-dependent (built from RX/RX90 Cliffords); `readout.fidelity`
  depends on the MZ pulse shape; `t1`/`t2`/`resonator`/`qubit` are physically
  independent of pulse shape. **There is no invalidation/staleness concept
  anywhere in qibocal** (`grep -rn "invalidat" src/qibocal/` returns
  nothing) — this has to be hand-built.
- **Protocol registration** (`qibocal/protocols/__init__.py`): each protocol
  submodule exports its module-level `Protocol(...)` instance name via
  `__all__`; these are star-imported and collected into a flat
  `PROTOCOLS: dict[str, Protocol]`. A runcard's `operation: <name>` string is
  looked up in this dict.
- **When updates run** (`qibocal/auto/mode.py`, `task.py`, `execute.py`,
  `output.py`): `ExecutionMode` is a `Flag` (`ACQUIRE`, `FIT`,
  `AUTOCALIBRATION = ACQUIRE | FIT`). `Task.run` calls `acquisition()` iff
  `ACQUIRE in mode` and `fit()` iff `FIT in mode`; it never calls `update()`
  itself. `update()` runs from exactly two call sites, both gated on
  `FIT in mode`: `Executor.run_protocol` (`qq run`) and `Output.process`
  (`qq fit`). Crucially, `acquisition()` receives the live, shared platform
  object regardless of the update gate, so a `platform.update(...)` call
  made directly inside `_acquisition` takes effect immediately.
- **Platform dump timing** (`Output.dump`/`update_platform`,
  `qibocal/auto/output.py`): the platform is dumped once per CLI invocation,
  at the very end, to `<path>/platform/parameters.json` (or
  `<path>/new_platform/...` if `platform/` already exists).

**Resolved risk** (this was the "stop and reconsider" trigger in the original
draft): the ordering hazard is real for the *split* `qq acquire` + `qq fit`
workflow. Traced `Output.process` directly: `qq fit` reconstructs the
platform **fresh from its canonical location**, not from the `qq acquire`
output folder, before running `update()` against it. This means:

- **Single-process `qq run` (or a single `qq acquire` with multiple
  actions)**: one live platform object flows through the whole action list in
  order — a mutation from `set_envelope`'s acquisition is immediately visible
  to a later action's acquisition in the same process. Works correctly.
- **Split `qq acquire` then `qq fit`**: an acquisition-side-only mutation
  would be invisible to the `qq fit` process's freshly reloaded platform.
  **Design decision**: put the durable mutation in `_update` (re-applying the
  same swap using data carried on `Data`/`Results`, not `Parameters`, since
  `fit(data)` never receives the original runcard parameters). `_update` runs
  against whatever platform object is actually live at fit time in *both*
  the single- and split-process cases, so this is what survives the process
  boundary.
- **Genuine, permanent limitation** (documented, not solved): if
  `set_envelope` and a later hardware-dependent routine run in two
  *separate* `qq acquire` invocations (not one `qq run`, not one `qq acquire`
  action list), the second invocation reloads the platform fresh from its
  canonical location and silently uses the **old** envelope for real
  pulses — `qq update <path>` must be run in between, and nothing in the
  framework detects or warns about this gap. This is a pre-existing
  framework-level issue (it also silently affects `calibrate_mixers`, and a
  confirmed live bug in the existing two-qubit dotted-path code — see Phase 6)
  — **decision: document this clearly in the routine's docstring, do not
  attempt to guard against it in code.**

---

## Phase 1 — Prototype (superseded by direct source reading)

Instead of prototyping on the dummy platform, the swap mechanism was
confirmed to already exist and be used in production: `platform.update({...})`
→ `Parameters.replace` (qibolab `_core/parameters.py`) dumps the whole
`Parameters` tree, walks dotted paths, sets the leaf, then `model_validate`s
the entire tree. `qibocal/update.py:217-224` (`drag_pulse_beta`) already does
exactly this for an RX envelope kind swap. Two things this confirmed that
change the design from the original draft:

- **Must replace the whole `envelope` sub-object as one dict-valued leaf**
  (`{prefix}.envelope`: a full `{"kind": ..., ...}` dict), not patch `kind`
  plus a couple of fields. `drag_pulse_beta` only works because Gaussian and
  Drag happen to share the `rel_sigma` field — that approach breaks
  immediately for the roadmap's own headline example, Gaussian→Rectangular
  (`Rectangular` has zero extra fields, so a leftover `rel_sigma` trips
  `extra="forbid"`).
- **Driver dispatch check deferred to hardware validation (Phase 5)** — no
  driver-level envelope dispatch issue was found in the parts of the qibolab
  source read so far; this remains to be checked against real Zurich/QICK
  hardware as originally planned.

---

## Phase 2 — The routine design (finalized)

New file: `src/qibocal/protocols/set_envelope.py`, following the
`calibrate_mixers.py` precedent (a `Protocol` whose acquisition does not run
a standard qubit pulse-sequence measurement, and whose `update()` mutates
platform config directly via dotted paths).

- **Scope** (decision): single-qubit gates (`RX`/`RX90`/`RX12`/`MZ`), the
  coupler gate (`CP`), *and* two-qubit gates (`CZ`/`CNOT`/`iSWAP`) are all in
  v1, handled by **one** registered `Protocol` — confirmed via a full-tree
  grep that `Protocol.two_qubit_gates` has zero runtime consumers in qibocal
  0.2.6 (it's declared and set by existing two-qubit routines but nothing
  reads it), so target-type dispatch (qubit vs. coupler vs. pair) can happen
  purely at runtime on the `target`'s type plus the `gate` string.
- **`_envelope_path(target, gate)`**: builds the dotted-path prefix:
  - single qubit: `native_gates.single_qubit.{q}.{gate}.0.1`
  - coupler: `native_gates.coupler.{q}.CP.0.1`
  - two-qubit: `native_gates.two_qubit.{q0}-{q1}.{gate}.0.1` — **must**
    format the pair explicitly as `f"{pair[0]}-{pair[1]}"` (confirmed real
    key format via `qibocal/calibration/calibration.py`'s `QubitPairId`
    serializer). Existing code (`snz_optimize.py`,
    `snz_optimize_t_idle.py`, `snz_optimize_t_idle_vs_t_tot.py`) instead
    interpolates the raw Python tuple repr, producing a key that doesn't
    match and is silently swallowed by `Completed.update_platform`'s blanket
    `except KeyError` — three existing routines whose two-qubit update
    silently no-ops today. This routine's helper must not repeat that bug.
  - `gate == "MZ"`: append `.probe` before `.envelope` (and before
    `.amplitude`/`.duration`).
- **`_apply(platform, target, gate, envelope_dict)`**: one
  `platform.update({...})` call setting `{prefix}.envelope`,
  `{prefix}.amplitude: 0.0`, `{prefix}.duration: 0.0` together — atomic,
  idempotent, and matches the decision below.
- **Amplitude/duration on swap** (decision): **reset to qibolab's
  `initialize_parameters` defaults (`amplitude=0, duration=0`)** alongside
  the new envelope. The gate becomes fully inert — exactly "as if never
  calibrated" — rather than merely swapping shape while keeping the old
  amplitude/duration.
- **Where the mutation happens**: both `_acquisition` (immediate, makes
  same-process downstream acquisitions correct right away) and `_update`
  (durable — re-applies the same swap from data carried on `Results`, which
  is what actually survives the `qq fit` platform-reload case).
- **`SetEnvelopeParameters(Parameters)`**: `target_gate: str`,
  `envelope: dict` (raw runcard dict). `Parameters` are plain dataclasses,
  not pydantic — typing a field as `qibolab.Envelope` does **not** trigger
  validation on its own. `__post_init__` explicitly calls
  `TypeAdapter(Envelope).validate_python(self.envelope)`, which then runs for
  free at `Task.__post_init__` — i.e. before any hardware time is spent.
- **`SetEnvelopeData`/`SetEnvelopeResults`**: envelopes are stored as plain
  JSON-able `dict`s (`envelope.model_dump(mode="json")`), never as raw
  pydantic `Envelope` instances — confirmed that `AbstractData._to_json` uses
  `dataclasses.asdict()` plus `json.dumps`, which cannot flatten a nested
  pydantic `BaseModel` and would raise `TypeError` the first time a real
  `Envelope` object sat inside a saved dict field. `Data` carries `new`/`old`
  envelope dicts (`old` for the diff plot) since `_fit`/`_update`
  structurally cannot see the original runcard `Parameters`
  (`operation.fit(data)` takes only `data`).
- **`_fit`**: trivial passthrough, no numerics — same shape as
  `calibrate_mixers._fit`.
- **`_plot`**: samples `envelope.i(samples)`/`.q(samples)` for old vs. new,
  plotly comparison figure, in the style of `drag/drag_simple.py`.
- Registered as `set_envelope = Protocol(_acquisition, _fit, _plot, _update)`,
  wired into `qibocal/protocols/__init__.py` the same way `calibrate_mixers`
  is.

**Check** (to run once implemented): two-action runcard (`set_envelope` →
`rabi_amplitude`) under `qq run`; confirm the Rabi used the new shape.

**Check**: same runcard as `qq acquire` then `qq fit` separately; per the
Phase 0 trace this is expected to work because the durable mutation lives in
`_update`, which runs against whatever platform `qq fit` reconstructs.

**Check**: confirm the dumped `parameters.json` carries the new envelope
(and the reset amplitude/duration).

---

## Phase 3 — Invalidation (finalized)

In `_update`, after re-applying the envelope/amplitude/duration reset:

- **`RX`/`RX90`/`RX12`**: clear
  `platform.calibration.single_qubits[target].rb_fidelity = None`
  unconditionally — `rb_fidelity` is built from RX/RX90 Cliffords regardless
  of which single-qubit gate changed.
- **`MZ`**: additionally clear
  `platform.calibration.single_qubits[target].readout.fidelity = None`.
- **`CZ`/`CNOT`/`iSWAP`**: clear both `rb_fidelity` and `cz_fidelity` on
  `platform.calibration.two_qubits.setdefault(target, TwoQubitCalibration())`
  for that pair.
- **Two-qubit pairs affected by a single-qubit gate change**: found via
  linear scan `[pair for pair in platform.pairs if target in pair]`
  (confirmed no reverse qubit→pairs lookup exists in qibolab or qibocal) —
  clear `rb_fidelity` (not `cz_fidelity`, which depends on CZ's own envelope,
  invalidated separately when `set_envelope` targets `CZ` directly) for each
  such pair.
- **`CP` (coupler)**: envelope/amplitude/duration reset only, **no**
  calibration invalidation — there is no structural coupler→pair mapping
  available in qibocal (it lives in platform-specific topology, outside
  qibocal's source). Documented as a known gap, not silently ignored.
- **Left untouched**: `t1`, `t2`, `t2_spin_echo`, `resonator`, `qubit`
  (frequencies, sweetspot) — confirmed physically independent of pulse
  shape.

**Check**: after running `set_envelope` alone, reload the resulting platform
fresh and confirm a protocol depending on a calibrated π pulse fails loudly
(zero amplitude) rather than running silently on stale values.

**Check**: couplers and two-qubit natives — confirmed a CZ involving a
changed target qubit gets its pair's `rb_fidelity` invalidated via the
linear scan above.

**Decided**: `gate: MZ` invalidates `readout.fidelity`; it leaves `t1`/`t2`
and frequency/sweetspot fields alone.

---

## Phase 4 — Tests (planned)

- Bad input: an envelope spec with a mismatched field (e.g. `rel_sigma` on a
  `Rectangular`) raises at `SetEnvelopeParameters.__post_init__`, both
  directly and via `Task.__post_init__` — before any hardware time is spent.
- Round-trip: `SetEnvelopeData`/`Results` survive `.save()`/`.load()` with
  dict-valued envelope fields (regression test for the confirmed
  JSON-serialization pitfall).
- Idempotence: calling `_apply` twice with the same envelope leaves platform
  state identical.
- `Readout` path: `gate: MZ` changes `probe.envelope` (and
  `probe.amplitude`/`probe.duration`), not a nonexistent top-level field.
- Two-qubit pair-key formatting: explicit regression test that
  `_envelope_path((0, 1), "CZ")` produces `"native_gates.two_qubit.0-1.CZ.0.1"`
  — guards against the tuple-repr bug found in the existing SNZ routines.
- Invalidation: named fields cleared per Phase 3; unaffected qubits/fields
  untouched; amplitude/duration reset to 0 regardless of chosen envelope
  kind.
- Dummy-platform integration test: `set_envelope` → `rabi_amplitude` in one
  `Executor` script (sibling to the existing `test_rx_calibration` in
  `tests/test_executor.py`).
- Two-process ordering test: `qq acquire` then `qq fit` separately (modeled
  on `tests/test_cli.py`'s existing `test_fit_command`), asserting the
  dumped `new_platform/parameters.json` and `new_platform/calibration.json`
  both reflect the swap — the concrete check that the `_update`-based
  durability design actually works across the process boundary.

---

## Phase 5 — Hardware validation

- Change RX from Gaussian to the truncated Gaussian on real hardware.
- Recalibrate amplitude and DRAG beta.
- Run RB and compare against a pre-change RB run.
- Check driver-level envelope dispatch on real hardware (Zurich, QICK/qibosoq)
  now that the platform-side mechanism is confirmed — this was deferred from
  Phase 1 since it requires real instruments.
- If the error rate doesn't move in an explainable way, suspect the plumbing
  before the physics.

---

## Phase 6 — Upstream

Two framework-level issues were found during investigation that are worth
surfacing to `qiboteam/qibocal` independently of this feature, since they
predate it and affect existing code:

- **Two-qubit dotted-path bug**: `snz_optimize.py`, `snz_optimize_t_idle.py`,
  and `snz_optimize_t_idle_vs_t_tot.py` all build `platform.update()` paths
  by interpolating the raw pair tuple (`f"...{pair}..."` →
  `"...(0, 1)..."`) instead of the `"0-1"` string format qibolab actually
  uses, so their two-qubit `_update` calls raise `KeyError` internally and
  are silently swallowed by `Completed.update_platform`'s blanket
  `except KeyError: log.warning(...)` — these updates no-op today without
  any visible error.
- **Acquire/fit platform-reload semantics**: `qq fit` reconstructs the
  platform fresh from its canonical location rather than from the preceding
  `qq acquire`'s dumped output, so any acquisition-side platform mutation
  (this routine's, and existing ones like `calibrate_mixers`) is invisible
  across a `qq acquire` → `qq acquire` invocation boundary unless `qq update`
  is run in between, with no detection or warning when it isn't. Worth
  raising as a design question — is this the intended contract, and if so,
  should the CLI warn when a runcard mutates the platform but no `qq update`
  followed before the next `qq acquire`?

Both should be opened as issues before or alongside the `set_envelope` PR,
since they're independent, pre-existing bugs this investigation surfaced
rather than something introduced by this feature.

- PR with tests and a docs page under the protocols section, once
  `set_envelope` itself is implemented per Phase 2-4 above.

---

## Risks to watch throughout

- **Routine abstraction stretch.** Confirmed precedent (`calibrate_mixers.py`)
  shows a `Protocol` with no real acquisition and an `update()` that mutates
  config directly is an accepted, if unusual, pattern in this codebase — not
  a novel workaround invented for this feature. The remaining stretch is
  `_fit` doing no fitting purely to carry data from `Data` to `Results` so
  `_update` can see it; this is functionally sound (confirmed by tracing
  `Output.process`) but is exactly the kind of workaround worth naming
  explicitly rather than treating as a fully natural fit for `Protocol`.
- **Invalidation is destructive and irreversible in-session.** Keep a copy of
  a known-good `parameters.json` outside the output folder before the first
  hardware run. Trust the invalidation path on the dummy platform first.
- **Cross-invocation staleness is a permanent, undetected gap** (see Phase 0
  and Phase 6) — accepted as documented behavior for v1, not fixed.
