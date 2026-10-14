import warnings
from itertools import pairwise

import numpy as np
import pytest

from mercupix import mask
from mercupix.mask import _BITFLAGS_DICT, _MASK_DTYPE

CR = _BITFLAGS_DICT["CR"]
SAT = _BITFLAGS_DICT["SATURATED"]
CM = _BITFLAGS_DICT["CM"]
HOT = _BITFLAGS_DICT["HOT"]
WARM = _BITFLAGS_DICT["WARM"]


def make_ramps(ngroups=20, npix=32, flux=30., dt=2.8, readnoise=10., seed=0):
    """Ramps of shape (ngroups, npix, npix): Poisson flux [ADU/s] + Gaussian read noise."""
    rng = np.random.default_rng(seed)
    counts = np.cumsum(rng.poisson(flux * dt, (ngroups, npix, npix)), axis=0)
    return (counts + rng.normal(0, readnoise, counts.shape)).astype("float32")


# ============== #
#   Bit flags    #
# ============== #
def test_bitflags_are_distinct_powers_of_two():
    """Each flag owns a single bit, and all flags fit in the mask dtype."""
    values = [v for k, v in _BITFLAGS_DICT.items() if k != "GOOD"]
    assert _BITFLAGS_DICT["GOOD"] == 0
    assert len(set(values)) == len(values)
    assert all(v > 0 and (v & (v - 1)) == 0 for v in values)
    # all flags fit in the mask dtype
    assert max(values) <= np.iinfo(_MASK_DTYPE).max


def test_get_boolmask_selects_flags():
    """get_boolmask keeps all flags by default, or a selection with ignore_flags."""
    bitmask = np.array([0, CR, SAT, CR | SAT, HOT], dtype=_MASK_DTYPE)
    assert mask.get_boolmask(bitmask).tolist() == [False, True, True, True, True]
    assert mask.get_boolmask(bitmask, ignore_flags="~CR").tolist() == [False, True, False, True, False]
    assert mask.get_boolmask(bitmask, ignore_flags="CR").tolist() == [False, False, True, True, True]
    assert mask.get_boolmask(bitmask, ignore_flags="~CR,SATURATED").tolist() == [False, True, True, True, False]


# ============== #
#   Saturation   #
# ============== #
def test_get_saturated():
    """Digitally saturated values (65535) get SATURATED; shape and dtype follow the input."""
    data = np.array([[[0, 65_534]], [[65_535, 65_535]]], dtype="uint16")
    sat = mask.get_saturated(data)
    assert sat.shape == data.shape and sat.dtype == np.dtype(_MASK_DTYPE)
    assert sat.tolist() == [[[0, 0]], [[SAT, SAT]]]


def test_get_saturated_threshold():
    """sat_threshold lowers the saturation level to a fraction of 65535."""
    threshold = int(0.9 * 65_535)
    data = np.array([threshold - 1, threshold, 65_535], dtype="uint16")
    assert mask.get_saturated(data, sat_threshold=0.9).tolist() == [0, SAT, SAT]


# ================== #
#  Charge migration  #
# ================== #
def test_get_charge_migration_flags_neighbours():
    """The 8 neighbours of a saturated pixel get CM, within the same group only."""
    sat = np.zeros((2, 5, 5), dtype=_MASK_DTYPE)
    sat[1, 2, 2] = SAT
    cm = mask.get_charge_migration(sat)

    expected = np.zeros((5, 5), dtype=_MASK_DTYPE)
    expected[1:4, 1:4] = CM
    expected[2, 2] = 0                       # the saturated pixel itself is not CM
    assert cm.shape == sat.shape
    assert (cm[1] == expected).all()
    assert not cm[0].any()                   # dilation stays within each group


def test_get_charge_migration_does_not_modify_input():
    """get_charge_migration returns a new mask and leaves its input untouched."""
    sat = np.zeros((1, 5, 5), dtype=_MASK_DTYPE)
    sat[0, 2, 2] = SAT
    before = sat.copy()
    _ = mask.get_charge_migration(sat)
    assert (sat == before).all()


def test_get_charge_migration_frame_edges():
    """A saturated corner pixel only has 3 neighbours."""
    sat = np.zeros((1, 4, 4), dtype=_MASK_DTYPE)
    sat[0, 0, 0] = SAT
    assert (mask.get_charge_migration(sat) == CM).sum() == 3


def test_saturation_and_charge_migration_combine():
    """OR-ing saturation and CM keeps SATURATED and never produces HOT."""
    sat = np.zeros((1, 5, 5), dtype=_MASK_DTYPE)
    sat[0, 2, 2] = SAT
    combined = sat | mask.get_charge_migration(sat)
    assert combined[0, 2, 2] == SAT
    assert (combined[0, 1:4, 1:4] == CM).sum() == 8
    assert not (combined & HOT).any()


def test_get_charge_migration_only_dilates_saturated_bit():
    """Only SATURATED pixels are dilated, not other flags."""
    flags = np.zeros((1, 5, 5), dtype=_MASK_DTYPE)
    flags[0, 2, 2] = CR                      # a cosmic ray, not a saturated pixel
    assert not mask.get_charge_migration(flags).any()


# ================== #
#  Known bad pixels  #
# ================== #
def test_detect_hotpix():
    """Dark slopes in [warm, hot) are WARM, from hot upwards HOT, never both."""
    dark_slope = np.array([0.1, 0.15, 0.3, 0.5, 2.])
    flags = mask.detect_hotpix(dark_slope, warm_threshold=0.15, hot_threshold=0.5)
    assert flags.tolist() == [0, WARM, WARM, HOT, HOT]
    assert not (((flags & WARM) > 0) & ((flags & HOT) > 0)).any()


# ============== #
#  Cosmic rays   #
# ============== #
@pytest.mark.parametrize("group", [1, 2, 5, 10, 18, 19])
def test_get_cosmics_detects_jump(group):
    """A 1000 ADU jump is flagged CR on the group where it appears."""
    data = make_ramps()
    data[group:] += 1_000
    cr_mask, _ = mask.get_cosmics(data, inpaint=False)
    assert cr_mask.shape == data.shape and cr_mask.dtype == np.dtype(_MASK_DTYPE)
    assert (cr_mask[group] & CR).all()


def test_get_cosmics_few_false_positives():
    """Clean ramps produce almost no CR flags."""
    data = make_ramps(npix=64)
    cr_mask, _ = mask.get_cosmics(data, inpaint=False, cr_threshold_mad=7)
    assert (cr_mask & CR).mean() < 1e-3


def test_get_cosmics_threshold_is_monotonic():
    """A higher cr_threshold_mad never flags more groups."""
    data = make_ramps()
    data[8:, :16] += 60                      # faint jumps in half of the pixels
    nflags = [np.count_nonzero(mask.get_cosmics(data, inpaint=False, cr_threshold_mad=t)[0] & CR)
              for t in (5, 7, 9, 12)]
    assert all(a >= b for a, b in pairwise(nflags))


def test_get_cosmics_inpaint_removes_jump():
    """Inpainting brings a ramp hit by a jump back to its clean level."""
    clean = make_ramps()
    data = clean.copy()
    data[10:] += 1_000
    _, inpainted = mask.get_cosmics(data, inpaint=True)
    # the jump is gone: the last group is back to the clean level (within noise).
    assert np.median(np.abs(inpainted[-1] - clean[-1])) < 50


def test_get_cosmics_inpaint_keeps_clean_ramps():
    """Inpainting leaves unflagged ramps unchanged."""
    data = make_ramps()
    cr_mask, inpainted = mask.get_cosmics(data, inpaint=True)
    untouched = ~(cr_mask[1:] & CR).astype(bool).any(axis=0)
    assert np.allclose(inpainted[:, untouched], data[:, untouched], atol=1e-2)


def test_get_cosmics_without_inpaint_returns_input():
    """With inpaint=False, the data are returned unchanged."""
    data = make_ramps()
    data[10:] += 1_000
    _, out = mask.get_cosmics(data, inpaint=False)
    assert np.array_equal(out, data)


def test_get_cosmics_never_flags_group0():
    """Group 0 has no difference to test: it is never flagged."""
    data = make_ramps()
    data[0] += 5_000                         # whatever happens before the first read
    cr_mask, _ = mask.get_cosmics(data, inpaint=False)
    assert not cr_mask[0].any()


def test_get_cosmics_does_not_flag_bright_sources():
    """A bright source without cosmic ray is not flagged CR."""
    data = make_ramps()
    ngroups = data.shape[0]
    data[:, 10:13, 10:13] += np.arange(1, ngroups + 1)[:, None, None] * 500.
    cr_mask, _ = mask.get_cosmics(data, inpaint=False)
    assert not (cr_mask[:, 10:13, 10:13] & CR).any()


def test_get_cosmics_uint16_same_as_float():
    """Raw uint16 input gives the same flags as the same data in float."""
    data = make_ramps() + 1_000
    data[8:] -= 300                          # a negative step, e.g. a reset or RTN drop
    data = np.round(data).astype("uint16")
    mask_uint16, _ = mask.get_cosmics(data, inpaint=False)
    mask_float, _ = mask.get_cosmics(data.astype("float32"), inpaint=False)
    assert np.array_equal(mask_uint16, mask_float)


def test_get_cosmics_returns_float32():
    """The returned datacube is float32, with or without inpainting."""
    data = make_ramps().astype("float64")
    data[10:] += 1_000
    for inpaint in (False, True):
        _, out = mask.get_cosmics(data, inpaint=inpaint)
        assert out.dtype == np.float32 and out.shape == data.shape


@pytest.mark.parametrize("window_length", [1, 2, 3, 5, 9, 100])
def test_get_cosmics_window_length(window_length):
    """Jumps are detected for any window length, even one longer than the ramp."""
    data = make_ramps()
    data[1:, :8] += 1_000                    # first difference
    data[10:, 8:16] += 1_000                 # middle
    data[19:, 16:24] += 1_000                # last difference
    cr_mask, _ = mask.get_cosmics(data, inpaint=False, window_length=window_length)
    assert (cr_mask[1, :8] & CR).all()
    assert (cr_mask[10, 8:16] & CR).all()
    assert (cr_mask[19, 16:24] & CR).all()
    assert not (cr_mask[:, 24:] & CR).any()


def test_get_cosmics_noise_free_ramp():
    """A noise-free ramp: the jump is flagged and nothing else."""
    data = np.cumsum(np.full((10, 2, 2), 84.), axis=0)
    data[6:, 0, 0] += 500
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # zero scatter: 0/0 and x/0
        cr_mask, _ = mask.get_cosmics(data, inpaint=False)
    assert cr_mask[6, 0, 0] == CR
    assert np.count_nonzero(cr_mask & CR) == 1


def test_get_cosmics_saturated():
    """With saturated=, saturating pixels get no false CR, and real jumps are still found."""
    data = make_ramps(npix=8, flux=2_000.) + 10_000    # bright: all pixels saturate around group 9 of 20
    data[4:, 0, 0] += 1_500                            # one cosmic ray (~20 sigma), before saturation
    data = np.minimum(data, 65_535)
    saturated = mask.get_saturated(data) > 0
    assert saturated[-1].all()                         # the test needs saturating pixels

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cr_mask, inpainted = mask.get_cosmics(data, inpaint=True, saturated=saturated)

    flagged = (cr_mask & CR) > 0
    assert flagged[4, 0, 0]                            # the cosmic ray, on the right group
    assert np.count_nonzero(flagged) == 1              # and nothing else
    assert np.array_equal(inpainted[:, 1:], data[:, 1:])   # other pixels left untouched


@pytest.mark.parametrize("ngroups", [2, 3])
def test_get_cosmics_too_few_groups(ngroups):
    """With fewer than 3 differences, a warning is issued and nothing is flagged."""
    data = make_ramps(ngroups=ngroups)
    with pytest.warns(UserWarning):
        cr_mask, out = mask.get_cosmics(data)
    assert not cr_mask.any()
    assert np.array_equal(out, data)


# ======================== #
#  Integration: get_mask   #
# ======================== #
def test_rawramp_get_mask_saturation_and_charge_migration():
    """RawRamp.get_mask flags SATURATED and CM on the active area, with no HOT."""
    from mercupix.ramp import RawRamp

    rng = np.random.default_rng(0)
    ngroups, size, nref = 10, 16, 4
    times = np.arange(1, ngroups + 1, dtype="float32") * 2.8
    raw = 10_000 + rng.normal(0, 5, (ngroups, size + 2 * nref, size + 2 * nref))
    raw[4:, 12, 12] = 65_535                 # full-frame coordinates
    ramp = RawRamp(raw.astype("uint16"), times, nchannels=4)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flags, _ = ramp.get_mask(saturated=True, charge_migration=True, disconnected=False)

    y = x = 12 - nref                        # active-area coordinates
    assert flags.shape == (ngroups, size, size)
    assert flags[-1, y, x] == SAT
    assert (flags[-1, y - 1:y + 2, x - 1:x + 2] == CM).sum() == 8
    assert not (flags[:4] & (SAT | CM)).any()          # before saturation
    assert not (flags & HOT).any()
