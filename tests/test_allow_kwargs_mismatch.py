"""Behavioural tests for the allow_kwargs_mismatch opt-in on the
Wang-Landau and canonical resume/measure entry points.

allow_kwargs_mismatch relaxes ONLY the ensemble-kwargs hash guard: a real
hash mismatch (both sides hash cleanly and differ) becomes a UserWarning
instead of a ValueError, so a run can be resumed or measured across software
environments where the pickle of identical move objects differs. CE
identity and ensemble_cls stay strict. Default False preserves the hard
error.

The mismatch is forced by tampering the on-disk ensemble_kwargs_hash to a
bogus value, then resuming with ensemble_kwargs=None (which hashes cleanly,
so reconstruction proceeds and the only failure is the kwargs-hash guard).
"""

from __future__ import annotations

import h5py
import pytest


def _tamper_kwargs_hash(path):
    """Overwrite /meta/ensemble_kwargs_hash with a bogus non-empty value."""
    with h5py.File(path, "a") as f:
        f["meta"].attrs["ensemble_kwargs_hash"] = "tampered-not-a-real-hash"


def _tamper_ce_identity(path):
    """Overwrite /meta/ce_identity with a bogus value."""
    with h5py.File(path, "a") as f:
        f["meta"].attrs["ce_identity"] = "tampered-not-a-real-ce-identity"


def _make_wl_checkpoint(tmp_path):
    """Tiny single-walker, two-window serial REWL run, saved to disk.

    Returns (path, cluster_expansion). Built with default ensemble_kwargs
    (None), so the on-disk kwargs hash is the clean empty-dict hash.
    """
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl import WangLandauParallelTempering
    from tests._wl_fixtures import make_wl_atoms, make_wl_ce

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    lo, hi = e0 - 50.0, e0 + 50.0
    pt = WangLandauParallelTempering(
        cluster_expansion=ce,
        atoms=[atoms, atoms],
        windows=[(lo, hi), (lo, hi)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
    )
    pt.run(n_cycles=2)
    path = tmp_path / "wl_ckpt.h5"
    pt.save_checkpoint(path)
    return path, ce


# Each case: (label, callable(path, ce, **kw) -> pt, is_process_pool).
def _wl_resume_cases():
    from mchammer_pt.wl import WangLandauParallelTempering as W

    return [
        (
            "resume",
            lambda p, ce, **kw: W.resume(p, cluster_expansion=ce, **kw),
            False,
        ),
        (
            "resume_process_pool",
            lambda p, ce, **kw: W.resume_process_pool(
                p, cluster_expansion=ce, **kw
            ),
            True,
        ),
    ]


_WL_RESUME = _wl_resume_cases()
_WL_RESUME_IDS = [c[0] for c in _WL_RESUME]


@pytest.mark.parametrize("label,call,is_pp", _WL_RESUME, ids=_WL_RESUME_IDS)
def test_wl_resume_default_raises_on_kwargs_hash_mismatch(
    label, call, is_pp, tmp_path
):
    """Without the flag, a real kwargs-hash mismatch still hard-errors."""
    path, ce = _make_wl_checkpoint(tmp_path)
    _tamper_kwargs_hash(path)
    with pytest.raises(ValueError, match="ensemble_kwargs hash mismatch"):
        call(path, ce)


@pytest.mark.parametrize("label,call,is_pp", _WL_RESUME, ids=_WL_RESUME_IDS)
def test_wl_resume_allow_kwargs_mismatch_warns_and_loads(
    label, call, is_pp, tmp_path
):
    """With allow_kwargs_mismatch=True the mismatch warns and the
    orchestrator is reconstructed."""
    path, ce = _make_wl_checkpoint(tmp_path)
    _tamper_kwargs_hash(path)
    with pytest.warns(UserWarning, match="allow_kwargs_mismatch"):
        pt = call(path, ce, allow_kwargs_mismatch=True)
    try:
        assert len(pt.windows) == 2
    finally:
        if is_pp:
            pt.pool.shutdown()


def test_wl_resume_ce_identity_still_guarded_under_allow(tmp_path):
    """allow_kwargs_mismatch must NOT bypass the CE-identity guard."""
    from mchammer_pt.wl import WangLandauParallelTempering

    path, ce = _make_wl_checkpoint(tmp_path)
    _tamper_ce_identity(path)
    with pytest.raises(ValueError, match="CE identity mismatch"):
        WangLandauParallelTempering.resume(
            path, cluster_expansion=ce, allow_kwargs_mismatch=True
        )


# Reuses _make_wl_checkpoint / _tamper_kwargs_hash (defined above in this
# file). measure_from_checkpoint* delegate to resume*/resume_process_pool
# with _frozen=True, forwarding allow_kwargs_mismatch.
def _wl_measure_cases():
    from mchammer_pt.wl import WangLandauParallelTempering as W

    return [
        (
            "measure_from_checkpoint",
            lambda p, ce, **kw: W.measure_from_checkpoint(
                p, cluster_expansion=ce, **kw
            ),
            False,
        ),
        (
            "measure_from_checkpoint_process_pool",
            lambda p, ce, **kw: W.measure_from_checkpoint_process_pool(
                p, cluster_expansion=ce, **kw
            ),
            True,
        ),
    ]


_WL_MEASURE = _wl_measure_cases()
_WL_MEASURE_IDS = [c[0] for c in _WL_MEASURE]


@pytest.mark.parametrize("label,call,is_pp", _WL_MEASURE, ids=_WL_MEASURE_IDS)
def test_wl_measure_default_raises_on_kwargs_hash_mismatch(
    label, call, is_pp, tmp_path
):
    """measure_from_checkpoint* hard-error on a kwargs-hash mismatch by
    default (they delegate to resume*/resume_process_pool)."""
    path, ce = _make_wl_checkpoint(tmp_path)
    _tamper_kwargs_hash(path)
    with pytest.raises(ValueError, match="ensemble_kwargs hash mismatch"):
        call(path, ce)


@pytest.mark.parametrize("label,call,is_pp", _WL_MEASURE, ids=_WL_MEASURE_IDS)
def test_wl_measure_allow_kwargs_mismatch_warns_and_loads(
    label, call, is_pp, tmp_path
):
    """measure_from_checkpoint* forward allow_kwargs_mismatch into the
    resume call: the mismatch warns and a frozen-measurement orchestrator
    is returned."""
    path, ce = _make_wl_checkpoint(tmp_path)
    _tamper_kwargs_hash(path)
    with pytest.warns(UserWarning, match="allow_kwargs_mismatch"):
        pt = call(path, ce, allow_kwargs_mismatch=True)
    try:
        assert len(pt.windows) == 2
    finally:
        if is_pp:
            pt.pool.shutdown()


# Canonical PT shares the _validate_kwargs_hash guard, so it gets the same
# flag. Reuses _tamper_kwargs_hash / _tamper_ce_identity (defined above).
def _make_canonical_checkpoint(tmp_path, toy_ce, toy_atoms):
    """Tiny canonical PT run, saved to disk. Returns the checkpoint path.

    Built with default ensemble_kwargs (None), so the on-disk kwargs hash
    is the clean empty-dict hash.
    """
    from mchammer_pt import CanonicalParallelTempering

    pt = CanonicalParallelTempering(
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperatures=[300.0, 400.0, 500.0],
        block_size=10,
        random_seed=42,
    )
    pt.run(n_cycles=3)
    path = tmp_path / "canonical_ckpt.h5"
    pt.save_checkpoint(path)
    return path


def _canonical_resume_cases():
    from mchammer_pt import CanonicalParallelTempering as C

    return [
        (
            "resume",
            lambda p, ce, **kw: C.resume(p, cluster_expansion=ce, **kw),
            False,
        ),
        (
            "resume_process_pool",
            lambda p, ce, **kw: C.resume_process_pool(
                p, cluster_expansion=ce, **kw
            ),
            True,
        ),
    ]


_CANON_RESUME = _canonical_resume_cases()
_CANON_RESUME_IDS = [c[0] for c in _CANON_RESUME]


@pytest.mark.parametrize(
    "label,call,is_pp", _CANON_RESUME, ids=_CANON_RESUME_IDS
)
def test_canonical_resume_default_raises_on_kwargs_hash_mismatch(
    label, call, is_pp, toy_ce, toy_atoms, tmp_path
):
    """Canonical resume*/resume_process_pool hard-error on a kwargs-hash
    mismatch by default."""
    path = _make_canonical_checkpoint(tmp_path, toy_ce, toy_atoms)
    _tamper_kwargs_hash(path)
    with pytest.raises(ValueError, match="ensemble_kwargs hash mismatch"):
        call(path, toy_ce)


@pytest.mark.parametrize(
    "label,call,is_pp", _CANON_RESUME, ids=_CANON_RESUME_IDS
)
def test_canonical_resume_allow_kwargs_mismatch_warns_and_loads(
    label, call, is_pp, toy_ce, toy_atoms, tmp_path
):
    """allow_kwargs_mismatch=True downgrades the canonical mismatch to a
    warning and reconstructs the orchestrator."""
    path = _make_canonical_checkpoint(tmp_path, toy_ce, toy_atoms)
    _tamper_kwargs_hash(path)
    with pytest.warns(UserWarning, match="allow_kwargs_mismatch"):
        pt = call(path, toy_ce, allow_kwargs_mismatch=True)
    try:
        assert len(pt.temperatures) == 3
    finally:
        if is_pp:
            pt.pool.shutdown()


def test_canonical_resume_ce_identity_still_guarded_under_allow(
    toy_ce, toy_atoms, tmp_path
):
    """allow_kwargs_mismatch must NOT bypass the CE-identity guard on
    canonical resume."""
    from mchammer_pt import CanonicalParallelTempering

    path = _make_canonical_checkpoint(tmp_path, toy_ce, toy_atoms)
    _tamper_ce_identity(path)
    with pytest.raises(ValueError, match="CE identity mismatch"):
        CanonicalParallelTempering.resume(
            path, cluster_expansion=toy_ce, allow_kwargs_mismatch=True
        )
