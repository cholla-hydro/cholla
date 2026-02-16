import enum

import cholla_utils
from cholla_utils._misc import ChollaDataFmt
from cholla_utils._testing import (
    BlockPreset,
    ExtraFmt,
    SyntheticSnapResult,
    assert_equal_particle_data,
    assert_arraydict_equal,
    USE_DEFAULT as DEFAULT,
    SyntheticInputPack,
)

import numpy as np
import pytest


def _c(blocking_preset, f, p, fmt) -> SyntheticInputPack:
    # the function name and argument names are intentionally short
    return SyntheticInputPack(
        blocking_preset=blocking_preset, field=f, particle=p, datafmt=fmt
    )


_CASES_ONLY_PARTICLE = [
    _c(BlockPreset.CubeDomain_1, f=None, p=DEFAULT, fmt=ChollaDataFmt.DISTRIBUTED),
    _c(BlockPreset.IrrDomain_142, f=None, p=DEFAULT, fmt=ChollaDataFmt.DISTRIBUTED),
    _c(BlockPreset.CubeDomain_1, f=None, p=DEFAULT, fmt=ChollaDataFmt.CONCAT),
    _c(BlockPreset.IrrDomain_142, f=None, p=DEFAULT, fmt=ChollaDataFmt.CONCAT),
]

_CASES_ONLY_FIELD = [
    _c(BlockPreset.CubeDomain_1, f=DEFAULT, p=None, fmt=ChollaDataFmt.DISTRIBUTED),
    _c(BlockPreset.IrrDomain_142, f=DEFAULT, p=None, fmt=ChollaDataFmt.DISTRIBUTED),
    _c(BlockPreset.CubeDomain_1, f=DEFAULT, p=None, fmt=ChollaDataFmt.LEGACY_CONCAT),
    _c(BlockPreset.IrrDomain_222, f=DEFAULT, p=None, fmt=ChollaDataFmt.LEGACY_CONCAT),
    _c(BlockPreset.IrrDomain_142, f=DEFAULT, p=None, fmt=ChollaDataFmt.LEGACY_CONCAT),
    _c(BlockPreset.CubeDomain_1, f=DEFAULT, p=None, fmt=ChollaDataFmt.CONCAT),
    _c(BlockPreset.IrrDomain_142, f=DEFAULT, p=None, fmt=ChollaDataFmt.CONCAT),
]

_CASES_WITH_BOTH = [
    _c(BlockPreset.CubeDomain_1, f=DEFAULT, p=DEFAULT, fmt=ChollaDataFmt.DISTRIBUTED),
    _c(BlockPreset.IrrDomain_222, f=DEFAULT, p=DEFAULT, fmt=ChollaDataFmt.DISTRIBUTED),
    _c(BlockPreset.IrrDomain_142, f=DEFAULT, p=DEFAULT, fmt=ChollaDataFmt.DISTRIBUTED),
    _c(BlockPreset.CubeDomain_1, f=DEFAULT, p=DEFAULT, fmt=ChollaDataFmt.CONCAT),
    _c(BlockPreset.IrrDomain_222, f=DEFAULT, p=DEFAULT, fmt=ChollaDataFmt.CONCAT),
    _c(BlockPreset.IrrDomain_142, f=DEFAULT, p=DEFAULT, fmt=ChollaDataFmt.CONCAT),
    _c(BlockPreset.IrrDomain_142, f=DEFAULT, p=DEFAULT, fmt=ChollaDataFmt.CONCAT),
    # cases where we concatenated either particles or fields (but not the other!)
    _c(BlockPreset.CubeDomain_1, f=DEFAULT, p=DEFAULT, fmt=ExtraFmt.DistribF_ConcatP),
    _c(BlockPreset.CubeDomain_1, f=DEFAULT, p=DEFAULT, fmt=ExtraFmt.ConcatF_DistribP),
    _c(BlockPreset.IrrDomain_142, f=DEFAULT, p=DEFAULT, fmt=ExtraFmt.DistribF_ConcatP),
    _c(BlockPreset.IrrDomain_142, f=DEFAULT, p=DEFAULT, fmt=ExtraFmt.ConcatF_DistribP),
    # cases where we concatenated paricles and fields to a single unified file
    # -> at this time, we don't provide tools to do this. But, let's confirm that this
    #    works (the modern concatenation format was explicitly designed to support it)
    _c(BlockPreset.CubeDomain_1, f=DEFAULT, p=DEFAULT, fmt=ExtraFmt.UnifiedConcat),
    _c(BlockPreset.IrrDomain_142, f=DEFAULT, p=DEFAULT, fmt=ExtraFmt.UnifiedConcat),
]

_CASES_WITH_FIELD = _CASES_ONLY_FIELD + _CASES_WITH_BOTH


@pytest.mark.parametrize("synth_snap", _CASES_WITH_FIELD, indirect=True)
def test_get_native_fields(synth_snap: SyntheticSnapResult):
    field_names = sorted(synth_snap.field_data.keys())
    loaded_field_names = cholla_utils.get_native_fields(synth_snap.root_field_fname)
    assert field_names == sorted(loaded_field_names)


@pytest.mark.parametrize("synth_snap", _CASES_WITH_FIELD, indirect=True)
def test_load_field_single(synth_snap: SyntheticSnapResult):
    global_arr = synth_snap.field_data
    field_name = sorted(global_arr.keys())[0]
    loaded = cholla_utils.load_field(synth_snap.root_field_fname, field=field_name)
    np.testing.assert_array_equal(global_arr[field_name], loaded)


@pytest.mark.parametrize("synth_snap", _CASES_WITH_FIELD, indirect=True)
def test_load_field_onebyone(synth_snap: SyntheticSnapResult):
    global_arr = synth_snap.field_data
    loaded = {
        f_name: cholla_utils.load_field(synth_snap.root_field_fname, field=f_name)
        for f_name in global_arr
    }
    assert_arraydict_equal(
        loaded, global_arr, err_msg="comparing individually loaded fields"
    )


@pytest.mark.parametrize("synth_snap", _CASES_WITH_FIELD, indirect=True)
def test_load_field_all(synth_snap: SyntheticSnapResult):
    global_arr = synth_snap.field_data
    loaded = cholla_utils.load_field(
        synth_snap.root_field_fname, field=list(global_arr)
    )
    assert_arraydict_equal(
        loaded, global_arr, err_msg="comparing fields loaded together"
    )


@pytest.mark.parametrize("synth_snap", _CASES_WITH_FIELD, indirect=True)
def test_load_field_single_subarray(synth_snap: SyntheticSnapResult):
    idx = np.s_[1, 2:, -3:-1]

    global_arr = synth_snap.field_data
    field_name = sorted(global_arr.keys())[0]

    loaded = cholla_utils.load_field(
        synth_snap.root_field_fname, field=field_name, idx=idx
    )
    np.testing.assert_equal(global_arr[field_name][idx], loaded)


@pytest.mark.parametrize("synth_snap", _CASES_WITH_FIELD, indirect=True)
def test_load_field_all_subarray(synth_snap: SyntheticSnapResult):
    idx = np.s_[1, 2:, -3:-1]

    global_arr = {k: arr[idx] for k, arr in synth_snap.field_data.items()}

    loaded = cholla_utils.load_field(
        synth_snap.root_field_fname, field=list(global_arr), idx=idx
    )

    assert_arraydict_equal(
        loaded, global_arr, err_msg="comparing fields loaded together"
    )


@pytest.mark.parametrize("synth_snap", _CASES_WITH_FIELD, indirect=True)
def test_load_field_failure(synth_snap):
    with pytest.raises(KeyError):
        cholla_utils.load_field(synth_snap.root_field_fname, field="not-a-field")

    with pytest.raises(FileNotFoundError):
        cholla_utils.load_field("/not/a/file", field="not-a-field")


# we can load particle data from either the root field fname or the root
# particle file.
# -> we are going to parametrize over both approaches


class FnameChoice(enum.Enum):
    ParticleRoot = enum.auto()
    FieldRoot = enum.auto()


_PAIRS = (
    [(case, FnameChoice.ParticleRoot) for case in _CASES_ONLY_PARTICLE]
    + [(case, FnameChoice.ParticleRoot) for case in _CASES_WITH_BOTH]
    + [
        (case, FnameChoice.FieldRoot)
        for case in _CASES_WITH_BOTH
        if case.datafmt is not ExtraFmt.UnifiedConcat
    ]
)


@pytest.mark.parametrize("synth_snap, ch", _PAIRS, indirect=["synth_snap"])
def test_get_native_ptype_properties(synth_snap: SyntheticSnapResult, ch: FnameChoice):
    if ch is FnameChoice.ParticleRoot:
        root_fname = synth_snap.root_particle_fname
    else:
        root_fname = synth_snap.root_field_fname

    ptype_prop_pairs = sorted(synth_snap.particle_data.get_ptype_prop_pairs())

    # Step 3a: loading ptype-property pairs
    loaded_pairs = cholla_utils.get_native_ptype_properties(root_fname)
    assert ptype_prop_pairs == sorted(loaded_pairs)


@pytest.mark.parametrize("synth_snap, ch", _PAIRS, indirect=["synth_snap"])
def test_load_particle_onebyone(synth_snap: SyntheticSnapResult, ch: FnameChoice):
    if ch is FnameChoice.ParticleRoot:
        root_fname = synth_snap.root_particle_fname
    else:
        root_fname = synth_snap.root_field_fname

    particle_data = synth_snap.particle_data
    loaded = {
        ptype_prop_pair: cholla_utils.load_particle(root_fname, ptype_prop_pair)
        for ptype_prop_pair in particle_data.get_ptype_prop_pairs()
    }
    assert_equal_particle_data(
        loaded,
        particle_data,
        exhaustive=True,
        err_msg="comparing individually loaded particle props",
    )


@pytest.mark.parametrize("synth_snap, ch", _PAIRS, indirect=["synth_snap"])
def test_load_particle_all(synth_snap: SyntheticSnapResult, ch: FnameChoice):
    if ch is FnameChoice.ParticleRoot:
        root_fname = synth_snap.root_particle_fname
    else:
        root_fname = synth_snap.root_field_fname

    particle_data = synth_snap.particle_data
    ptype_prop_pairs = particle_data.get_ptype_prop_pairs()
    loaded = cholla_utils.load_particle(root_fname, ptype_prop_pairs)

    assert_equal_particle_data(
        loaded,
        particle_data,
        exhaustive=True,
        err_msg="comparing particle props loaded together",
    )


@pytest.mark.parametrize("synth_snap, ch", _PAIRS, indirect=["synth_snap"])
def test_load_particle_failure(synth_snap: SyntheticSnapResult, ch: FnameChoice):
    if ch is FnameChoice.ParticleRoot:
        root_fname = synth_snap.root_particle_fname
    else:
        root_fname = synth_snap.root_field_fname

    with pytest.raises(KeyError):
        cholla_utils.load_particle(root_fname, ptype_prop_pair=("io", "not-a-prop"))


def test_load_particle_nonfile():
    with pytest.raises(FileNotFoundError):
        cholla_utils.load_particle(
            "/not/a/file", ptype_prop_pair=("io", "particle_IDs")
        )
